//! glTF 2.0 exporter.
//!
//! Exports [`Scene`] data to binary glTF (`.glb`) format.
//! Material instances are collected from meshes via `Arc<CpuMaterialInstance>`
//! and deduplicated using Arc pointer identity, along with textures and samplers.

use std::collections::{BTreeMap, HashMap};
use std::io::Cursor;
use std::sync::Arc;

use gltf_dep::json as gj;

use crate::material::{AlphaMode, CpuMaterialInstance, TextureRef, TextureSource};
use crate::mesh::{
    CpuMesh, IndexFormat, PrimitiveTopology, VertexAttributeFormat, VertexAttributeSemantic,
};
use crate::sampler::{AddressMode, CpuSampler, FilterMode};
use crate::scene::{
    Animation, AnimationProperty, CameraProjection, Interpolation, Scene, SceneNode,
};
use crate::texture::CpuTexture;

use super::error::GltfError;

// ---------------------------------------------------------------------------
// Export context
// ---------------------------------------------------------------------------

pub(super) struct ExportContext {
    root: gj::Root,
    buffer_data: Vec<u8>,

    // Arc pointer → glTF index dedup maps
    instance_list: Vec<Arc<CpuMaterialInstance>>,
    instance_map: HashMap<*const CpuMaterialInstance, u32>,
    cpu_textures: Vec<Arc<CpuTexture>>,
    texture_map: HashMap<*const CpuTexture, u32>,
    sampler_list: Vec<Arc<CpuSampler>>,
    sampler_map: HashMap<*const CpuSampler, u32>,
    named_image_map: HashMap<String, u32>,
    // (image_idx, Option<sampler_idx>) → glTF texture index
    gltf_texture_map: HashMap<(u32, Option<u32>), u32>,
}

impl ExportContext {
    pub(super) fn new() -> Self {
        Self {
            root: gj::Root::default(),
            buffer_data: Vec::new(),
            instance_list: Vec::new(),
            instance_map: HashMap::new(),
            cpu_textures: Vec::new(),
            texture_map: HashMap::new(),
            sampler_list: Vec::new(),
            sampler_map: HashMap::new(),
            named_image_map: HashMap::new(),
            gltf_texture_map: HashMap::new(),
        }
    }

    // -- Step 1: Collect unique resources from scenes -------------------------

    pub(super) fn collect_resources(&mut self, scenes: &[&Scene]) {
        // Collect unique material instances from all meshes
        for scene in scenes {
            for mesh in &scene.meshes {
                if let Some(inst_arc) = mesh.material() {
                    let ptr = Arc::as_ptr(inst_arc);
                    if !self.instance_map.contains_key(&ptr) {
                        let idx = self.instance_list.len() as u32;
                        self.instance_map.insert(ptr, idx);
                        self.instance_list.push(Arc::clone(inst_arc));
                    }
                }
            }
        }

        // Collect unique textures and samplers from collected instances
        for inst in &self.instance_list.clone() {
            for tex_ref in inst.textures() {
                if let TextureSource::Cpu(arc) = &tex_ref.texture {
                    let ptr = Arc::as_ptr(arc);
                    if !self.texture_map.contains_key(&ptr) {
                        let idx = self.cpu_textures.len() as u32;
                        self.texture_map.insert(ptr, idx);
                        self.cpu_textures.push(Arc::clone(arc));
                    }
                }

                if let Some(sampler_arc) = &tex_ref.sampler {
                    let ptr = Arc::as_ptr(sampler_arc);
                    if !self.sampler_map.contains_key(&ptr) {
                        let idx = self.sampler_list.len() as u32;
                        self.sampler_map.insert(ptr, idx);
                        self.sampler_list.push(Arc::clone(sampler_arc));
                    }
                }
            }
        }
    }

    // -- Step 2: Build images ------------------------------------------------

    pub(super) fn build_images(&mut self) -> Result<(), GltfError> {
        // Clone the list to avoid borrowing self immutably while mutating
        let textures = self.cpu_textures.clone();
        for cpu_tex in &textures {
            let png_bytes = encode_texture_to_png(cpu_tex)?;

            let view_idx = self.push_buffer_view(&png_bytes, None);

            self.root.images.push(gj::Image {
                buffer_view: Some(gj::Index::new(view_idx)),
                mime_type: Some(gj::image::MimeType("image/png".into())),
                name: cpu_tex.name.clone(),
                uri: None,
                extensions: None,
                extras: gj::Extras::default(),
            });
        }
        Ok(())
    }

    // -- Step 3: Build samplers ----------------------------------------------

    pub(super) fn build_samplers(&mut self) {
        let samplers = self.sampler_list.clone();
        for sampler in &samplers {
            self.root.samplers.push(gj::texture::Sampler {
                mag_filter: Some(gj::validation::Checked::Valid(map_mag_filter(
                    sampler.mag_filter,
                ))),
                min_filter: Some(gj::validation::Checked::Valid(map_min_filter(
                    sampler.min_filter,
                    sampler.mipmap_filter,
                ))),
                wrap_s: gj::validation::Checked::Valid(map_wrapping(sampler.address_mode_u)),
                wrap_t: gj::validation::Checked::Valid(map_wrapping(sampler.address_mode_v)),
                name: sampler.name.clone(),
                extensions: None,
                extras: gj::Extras::default(),
            });
        }
    }

    /// Get or create a glTF texture index for a TextureRef.
    fn resolve_texture_index(&mut self, tex_ref: &TextureRef) -> Option<u32> {
        let image_idx = match &tex_ref.texture {
            TextureSource::Cpu(arc) => {
                let ptr = Arc::as_ptr(arc);
                *self.texture_map.get(&ptr)?
            }
            TextureSource::Named(name) => {
                if let Some(&idx) = self.named_image_map.get(name) {
                    idx
                } else {
                    let idx = self.root.images.len() as u32;
                    self.named_image_map.insert(name.clone(), idx);
                    self.root.images.push(gj::Image {
                        buffer_view: None,
                        mime_type: None,
                        name: Some(name.clone()),
                        uri: Some(name.clone()),
                        extensions: None,
                        extras: gj::Extras::default(),
                    });
                    idx
                }
            }
        };

        let sampler_idx = tex_ref.sampler.as_ref().map(|s| {
            let ptr = Arc::as_ptr(s);
            *self.sampler_map.get(&ptr).unwrap()
        });

        let key = (image_idx, sampler_idx);
        if let Some(&tex_idx) = self.gltf_texture_map.get(&key) {
            return Some(tex_idx);
        }

        let tex_idx = self.root.textures.len() as u32;
        self.gltf_texture_map.insert(key, tex_idx);

        self.root.textures.push(gj::Texture {
            name: None,
            sampler: sampler_idx.map(gj::Index::new),
            source: gj::Index::new(image_idx),
            extensions: None,
            extras: gj::Extras::default(),
        });

        Some(tex_idx)
    }

    // -- Step 4: Build materials ---------------------------------------------

    pub(super) fn build_materials(&mut self) {
        let instances = self.instance_list.clone();
        for inst in &instances {
            let base_color_factor = inst.get_vec4("base_color").unwrap_or([1.0, 1.0, 1.0, 1.0]);
            let metallic_factor = inst.get_float("metallic").unwrap_or(1.0);
            let roughness_factor = inst.get_float("roughness").unwrap_or(1.0);
            let emissive_factor = inst.get_vec3("emissive").unwrap_or([0.0, 0.0, 0.0]);

            let base_color_texture = self.build_texture_info(inst, "base_color_texture");
            let metallic_roughness_texture =
                self.build_texture_info(inst, "metallic_roughness_texture");
            let emissive_texture = self.build_texture_info(inst, "emissive_texture");

            let normal_texture = self.build_normal_texture_info(inst);
            let occlusion_texture = self.build_occlusion_texture_info(inst);

            let alpha_mode = match inst.material.alpha_mode {
                AlphaMode::Opaque => {
                    gj::validation::Checked::Valid(gj::material::AlphaMode::Opaque)
                }
                AlphaMode::Mask { .. } => {
                    gj::validation::Checked::Valid(gj::material::AlphaMode::Mask)
                }
                AlphaMode::Blend => gj::validation::Checked::Valid(gj::material::AlphaMode::Blend),
            };

            let alpha_cutoff = match inst.material.alpha_mode {
                AlphaMode::Mask { cutoff } => Some(gj::material::AlphaCutoff(cutoff)),
                _ => None,
            };

            let pbr = gj::material::PbrMetallicRoughness {
                base_color_factor: gj::material::PbrBaseColorFactor(base_color_factor),
                base_color_texture,
                metallic_factor: gj::material::StrengthFactor(metallic_factor),
                roughness_factor: gj::material::StrengthFactor(roughness_factor),
                metallic_roughness_texture,
                extensions: None,
                extras: gj::Extras::default(),
            };

            self.root.materials.push(gj::Material {
                name: inst.name.clone(),
                alpha_cutoff,
                alpha_mode,
                double_sided: inst.material.double_sided,
                pbr_metallic_roughness: pbr,
                normal_texture,
                occlusion_texture,
                emissive_texture,
                emissive_factor: gj::material::EmissiveFactor(emissive_factor),
                extensions: None,
                extras: gj::Extras::default(),
            });
        }
    }

    fn build_texture_info(
        &mut self,
        inst: &CpuMaterialInstance,
        name: &str,
    ) -> Option<gj::texture::Info> {
        let tex_ref = inst.get_texture(name)?;
        let tex_idx = self.resolve_texture_index(tex_ref)?;
        Some(gj::texture::Info {
            index: gj::Index::new(tex_idx),
            tex_coord: tex_ref.tex_coord,
            extensions: None,
            extras: gj::Extras::default(),
        })
    }

    fn build_normal_texture_info(
        &mut self,
        inst: &CpuMaterialInstance,
    ) -> Option<gj::material::NormalTexture> {
        let tex_ref = inst.get_texture("normal_texture")?;
        let tex_idx = self.resolve_texture_index(tex_ref)?;
        let scale = inst.get_float("normal_scale").unwrap_or(1.0);
        Some(gj::material::NormalTexture {
            index: gj::Index::new(tex_idx),
            scale,
            tex_coord: tex_ref.tex_coord,
            extensions: None,
            extras: gj::Extras::default(),
        })
    }

    fn build_occlusion_texture_info(
        &mut self,
        inst: &CpuMaterialInstance,
    ) -> Option<gj::material::OcclusionTexture> {
        let tex_ref = inst.get_texture("occlusion_texture")?;
        let tex_idx = self.resolve_texture_index(tex_ref)?;
        let strength = inst.get_float("occlusion_strength").unwrap_or(1.0);
        Some(gj::material::OcclusionTexture {
            index: gj::Index::new(tex_idx),
            strength: gj::material::StrengthFactor(strength),
            tex_coord: tex_ref.tex_coord,
            extensions: None,
            extras: gj::Extras::default(),
        })
    }

    // -- Step 5: Build scenes ------------------------------------------------

    pub(super) fn build_scenes(&mut self, scenes: &[&Scene]) -> Result<(), GltfError> {
        for scene in scenes {
            let scene_mesh_offset = self.root.meshes.len() as u32;
            self.build_meshes(&scene.meshes)?;

            let scene_camera_offset = self.root.cameras.len() as u32;
            self.build_cameras(&scene.cameras);

            let scene_node_offset = self.root.nodes.len() as u32;
            let root_indices =
                self.flatten_scene_nodes(&scene.nodes, scene_mesh_offset, scene_camera_offset);

            let scene_skin_offset = self.root.skins.len() as u32;
            self.build_skins(&scene.skins, scene_node_offset)?;

            self.patch_skin_refs(&scene.nodes, scene_node_offset, scene_skin_offset);

            self.build_animations(&scene.animations, scene_node_offset)?;

            self.root.scenes.push(gj::Scene {
                name: scene.name.clone(),
                nodes: root_indices.into_iter().map(gj::Index::new).collect(),
                extensions: None,
                extras: gj::Extras::default(),
            });
        }

        Ok(())
    }

    fn build_meshes(&mut self, meshes: &[CpuMesh]) -> Result<(), GltfError> {
        for mesh in meshes {
            let layout = mesh.layout();
            let stride = layout.buffer_stride(0);
            let vertex_data = mesh.vertex_buffer_data(0).unwrap_or(&[]);
            let vertex_count = mesh.vertex_count();

            let vtx_view_idx = if !vertex_data.is_empty() {
                Some(self.push_buffer_view_with_stride(
                    vertex_data,
                    stride,
                    Some(gj::buffer::Target::ArrayBuffer),
                ))
            } else {
                None
            };

            let mut attributes = BTreeMap::new();
            for attr in &layout.attributes {
                if attr.buffer_index != 0 {
                    continue;
                }
                let Some(view_idx) = vtx_view_idx else {
                    continue;
                };

                let (component_type, accessor_type) = map_attribute_format(attr.format);
                let (min, max) = if attr.semantic == VertexAttributeSemantic::Position {
                    compute_position_min_max(vertex_data, stride, attr.offset, vertex_count)
                } else {
                    (None, None)
                };

                let acc_idx = self.push_accessor(
                    view_idx,
                    attr.offset,
                    vertex_count,
                    component_type,
                    accessor_type,
                    min,
                    max,
                    false,
                );

                let semantic = map_semantic(attr.semantic);
                attributes.insert(
                    gj::validation::Checked::Valid(semantic),
                    gj::Index::new(acc_idx),
                );
            }

            let indices_accessor = if let (Some(idx_data), Some(idx_format)) =
                (mesh.index_data(), mesh.index_format())
            {
                let view_idx =
                    self.push_buffer_view(idx_data, Some(gj::buffer::Target::ElementArrayBuffer));
                let component_type = match idx_format {
                    IndexFormat::Uint16 => gj::accessor::ComponentType::U16,
                    IndexFormat::Uint32 => gj::accessor::ComponentType::U32,
                };
                let acc_idx = self.push_accessor(
                    view_idx,
                    0,
                    mesh.index_count(),
                    component_type,
                    gj::accessor::Type::Scalar,
                    None,
                    None,
                    false,
                );
                Some(gj::Index::new(acc_idx))
            } else {
                None
            };

            let mode = map_topology(mesh.topology());

            let primitive = gj::mesh::Primitive {
                attributes,
                extensions: None,
                extras: gj::Extras::default(),
                indices: indices_accessor,
                material: mesh
                    .material()
                    .map(|m| gj::Index::new(*self.instance_map.get(&Arc::as_ptr(m)).unwrap())),
                mode: gj::validation::Checked::Valid(mode),
                targets: None,
            };

            self.root.meshes.push(gj::Mesh {
                name: mesh.label().map(String::from),
                primitives: vec![primitive],
                weights: None,
                extensions: None,
                extras: gj::Extras::default(),
            });
        }
        Ok(())
    }

    fn build_cameras(&mut self, cameras: &[crate::scene::SceneCamera]) {
        for cam in cameras {
            let camera = match &cam.projection {
                CameraProjection::Perspective {
                    yfov,
                    aspect,
                    znear,
                    zfar,
                } => gj::Camera {
                    name: cam.name.clone(),
                    type_: gj::validation::Checked::Valid(gj::camera::Type::Perspective),
                    orthographic: None,
                    perspective: Some(gj::camera::Perspective {
                        aspect_ratio: *aspect,
                        yfov: *yfov,
                        zfar: *zfar,
                        znear: *znear,
                        extensions: None,
                        extras: gj::Extras::default(),
                    }),
                    extensions: None,
                    extras: gj::Extras::default(),
                },
                CameraProjection::Orthographic {
                    xmag,
                    ymag,
                    znear,
                    zfar,
                } => gj::Camera {
                    name: cam.name.clone(),
                    type_: gj::validation::Checked::Valid(gj::camera::Type::Orthographic),
                    orthographic: Some(gj::camera::Orthographic {
                        xmag: *xmag,
                        ymag: *ymag,
                        zfar: *zfar,
                        znear: *znear,
                        extensions: None,
                        extras: gj::Extras::default(),
                    }),
                    perspective: None,
                    extensions: None,
                    extras: gj::Extras::default(),
                },
            };
            self.root.cameras.push(camera);
        }
    }

    fn flatten_scene_nodes(
        &mut self,
        roots: &[SceneNode],
        mesh_offset: u32,
        camera_offset: u32,
    ) -> Vec<u32> {
        let mut root_indices = Vec::new();
        for node in roots {
            let idx = self.flatten_node(node, mesh_offset, camera_offset);
            root_indices.push(idx);
        }
        root_indices
    }

    fn flatten_node(&mut self, node: &SceneNode, mesh_offset: u32, camera_offset: u32) -> u32 {
        let my_index = self.root.nodes.len() as u32;
        self.root.nodes.push(gj::Node::default());

        let child_indices: Vec<u32> = node
            .children
            .iter()
            .map(|c| self.flatten_node(c, mesh_offset, camera_offset))
            .collect();

        let mesh = if !node.meshes.is_empty() {
            Some(gj::Index::new(mesh_offset + node.meshes[0] as u32))
        } else {
            None
        };

        let mut all_children: Vec<gj::Index<gj::Node>> =
            child_indices.into_iter().map(gj::Index::new).collect();

        for &mesh_idx in node.meshes.iter().skip(1) {
            let extra_idx = self.root.nodes.len() as u32;
            self.root.nodes.push(gj::Node {
                mesh: Some(gj::Index::new(mesh_offset + mesh_idx as u32)),
                ..gj::Node::default()
            });
            all_children.push(gj::Index::new(extra_idx));
        }

        let t = node.transform;
        let is_identity = t.translation == [0.0, 0.0, 0.0]
            && t.rotation == [0.0, 0.0, 0.0, 1.0]
            && t.scale == [1.0, 1.0, 1.0];

        let gltf_node = gj::Node {
            name: node.name.clone(),
            camera: node
                .camera
                .map(|i| gj::Index::new(camera_offset + i as u32)),
            children: if all_children.is_empty() {
                None
            } else {
                Some(all_children)
            },
            mesh,
            skin: None,
            translation: if !is_identity {
                Some(t.translation)
            } else {
                None
            },
            rotation: if !is_identity {
                Some(gj::scene::UnitQuaternion(t.rotation))
            } else {
                None
            },
            scale: if !is_identity { Some(t.scale) } else { None },
            matrix: None,
            weights: None,
            extensions: None,
            extras: gj::Extras::default(),
        };

        self.root.nodes[my_index as usize] = gltf_node;
        my_index
    }

    fn patch_skin_refs(&mut self, roots: &[SceneNode], node_offset: u32, skin_offset: u32) {
        let mut flat_idx = node_offset;
        for node in roots {
            self.patch_skin_refs_recursive(node, &mut flat_idx, skin_offset);
        }
    }

    fn patch_skin_refs_recursive(
        &mut self,
        node: &SceneNode,
        flat_idx: &mut u32,
        skin_offset: u32,
    ) {
        let my_idx = *flat_idx;
        *flat_idx += 1;

        if let Some(skin_idx) = node.skin {
            self.root.nodes[my_idx as usize].skin =
                Some(gj::Index::new(skin_offset + skin_idx as u32));
        }

        for child in &node.children {
            self.patch_skin_refs_recursive(child, flat_idx, skin_offset);
        }

        // Skip extra child nodes for multi-mesh nodes
        if node.meshes.len() > 1 {
            *flat_idx += (node.meshes.len() - 1) as u32;
        }
    }

    fn build_skins(
        &mut self,
        skins: &[crate::scene::SceneSkin],
        node_offset: u32,
    ) -> Result<(), GltfError> {
        for skin in skins {
            let ibm_accessor = if !skin.inverse_bind_matrices.is_empty() {
                let data: Vec<u8> = skin
                    .inverse_bind_matrices
                    .iter()
                    .flat_map(|m| m.iter().flat_map(|f| f.to_le_bytes()))
                    .collect();
                let view_idx = self.push_buffer_view(&data, None);
                let acc_idx = self.push_accessor(
                    view_idx,
                    0,
                    skin.inverse_bind_matrices.len() as u32,
                    gj::accessor::ComponentType::F32,
                    gj::accessor::Type::Mat4,
                    None,
                    None,
                    false,
                );
                Some(gj::Index::new(acc_idx))
            } else {
                None
            };

            self.root.skins.push(gj::Skin {
                name: skin.name.clone(),
                inverse_bind_matrices: ibm_accessor,
                joints: skin
                    .joints
                    .iter()
                    .map(|&j| gj::Index::new(node_offset + j as u32))
                    .collect(),
                skeleton: skin
                    .skeleton
                    .map(|s| gj::Index::new(node_offset + s as u32)),
                extensions: None,
                extras: gj::Extras::default(),
            });
        }
        Ok(())
    }

    fn build_animations(
        &mut self,
        animations: &[Animation],
        node_offset: u32,
    ) -> Result<(), GltfError> {
        for anim in animations {
            let mut channels = Vec::new();
            let mut samplers = Vec::new();

            for channel in &anim.channels {
                let ts_data: Vec<u8> = channel
                    .timestamps
                    .iter()
                    .flat_map(|f| f.to_le_bytes())
                    .collect();
                let ts_view = self.push_buffer_view(&ts_data, None);
                let ts_min = channel
                    .timestamps
                    .iter()
                    .copied()
                    .reduce(f32::min)
                    .unwrap_or(0.0);
                let ts_max = channel
                    .timestamps
                    .iter()
                    .copied()
                    .reduce(f32::max)
                    .unwrap_or(0.0);
                let ts_acc = self.push_accessor(
                    ts_view,
                    0,
                    channel.timestamps.len() as u32,
                    gj::accessor::ComponentType::F32,
                    gj::accessor::Type::Scalar,
                    Some(json_f32_array(&[ts_min])),
                    Some(json_f32_array(&[ts_max])),
                    false,
                );

                let val_data: Vec<u8> = channel
                    .values
                    .iter()
                    .flat_map(|f| f.to_le_bytes())
                    .collect();
                let val_view = self.push_buffer_view(&val_data, None);
                let (val_type, val_count) = match channel.property {
                    AnimationProperty::Translation | AnimationProperty::Scale => {
                        (gj::accessor::Type::Vec3, channel.values.len() / 3)
                    }
                    AnimationProperty::Rotation => {
                        (gj::accessor::Type::Vec4, channel.values.len() / 4)
                    }
                    AnimationProperty::MorphTargetWeights => {
                        (gj::accessor::Type::Scalar, channel.values.len())
                    }
                };
                let val_acc = self.push_accessor(
                    val_view,
                    0,
                    val_count as u32,
                    gj::accessor::ComponentType::F32,
                    val_type,
                    None,
                    None,
                    false,
                );

                let sampler_idx = samplers.len() as u32;
                samplers.push(gj::animation::Sampler {
                    input: gj::Index::new(ts_acc),
                    output: gj::Index::new(val_acc),
                    interpolation: gj::validation::Checked::Valid(map_interpolation(
                        channel.interpolation,
                    )),
                    extensions: None,
                    extras: gj::Extras::default(),
                });

                let path = match channel.property {
                    AnimationProperty::Translation => gj::animation::Property::Translation,
                    AnimationProperty::Rotation => gj::animation::Property::Rotation,
                    AnimationProperty::Scale => gj::animation::Property::Scale,
                    AnimationProperty::MorphTargetWeights => {
                        gj::animation::Property::MorphTargetWeights
                    }
                };

                channels.push(gj::animation::Channel {
                    sampler: gj::Index::new(sampler_idx),
                    target: gj::animation::Target {
                        node: gj::Index::new(node_offset + channel.target_node as u32),
                        path: gj::validation::Checked::Valid(path),
                        extensions: None,
                        extras: gj::Extras::default(),
                    },
                    extensions: None,
                    extras: gj::Extras::default(),
                });
            }

            self.root.animations.push(gj::Animation {
                name: anim.name.clone(),
                channels,
                samplers,
                extensions: None,
                extras: gj::Extras::default(),
            });
        }
        Ok(())
    }

    pub(super) fn set_default_scene(&mut self, default_scene: Option<usize>) {
        self.root.scene = default_scene.map(|i| gj::Index::new(i as u32));
        self.root.asset = gj::Asset {
            generator: Some("RedLilium Engine".into()),
            version: "2.0".into(),
            ..Default::default()
        };
    }

    // -- Buffer/accessor helpers ---------------------------------------------

    fn align_buffer(&mut self) {
        let padding = (4 - (self.buffer_data.len() % 4)) % 4;
        self.buffer_data.extend(std::iter::repeat_n(0u8, padding));
    }

    fn push_buffer_view(&mut self, data: &[u8], target: Option<gj::buffer::Target>) -> u32 {
        self.align_buffer();
        let offset = self.buffer_data.len();
        self.buffer_data.extend_from_slice(data);

        let view_idx = self.root.buffer_views.len() as u32;
        self.root.buffer_views.push(gj::buffer::View {
            buffer: gj::Index::new(0),
            byte_offset: Some(gj::validation::USize64(offset as u64)),
            byte_length: gj::validation::USize64(data.len() as u64),
            byte_stride: None,
            target: target.map(gj::validation::Checked::Valid),
            name: None,
            extensions: None,
            extras: gj::Extras::default(),
        });

        view_idx
    }

    fn push_buffer_view_with_stride(
        &mut self,
        data: &[u8],
        stride: u32,
        target: Option<gj::buffer::Target>,
    ) -> u32 {
        self.align_buffer();
        let offset = self.buffer_data.len();
        self.buffer_data.extend_from_slice(data);

        let view_idx = self.root.buffer_views.len() as u32;
        self.root.buffer_views.push(gj::buffer::View {
            buffer: gj::Index::new(0),
            byte_offset: Some(gj::validation::USize64(offset as u64)),
            byte_length: gj::validation::USize64(data.len() as u64),
            byte_stride: Some(gj::buffer::Stride(stride as usize)),
            target: target.map(gj::validation::Checked::Valid),
            name: None,
            extensions: None,
            extras: gj::Extras::default(),
        });

        view_idx
    }

    #[allow(clippy::too_many_arguments)]
    fn push_accessor(
        &mut self,
        buffer_view: u32,
        byte_offset: u32,
        count: u32,
        component_type: gj::accessor::ComponentType,
        type_: gj::accessor::Type,
        min: Option<gj::Value>,
        max: Option<gj::Value>,
        normalized: bool,
    ) -> u32 {
        let acc_idx = self.root.accessors.len() as u32;
        self.root.accessors.push(gj::Accessor {
            buffer_view: Some(gj::Index::new(buffer_view)),
            byte_offset: Some(gj::validation::USize64(byte_offset as u64)),
            count: gj::validation::USize64(count as u64),
            component_type: gj::validation::Checked::Valid(gj::accessor::GenericComponentType(
                component_type,
            )),
            type_: gj::validation::Checked::Valid(type_),
            min,
            max,
            normalized,
            name: None,
            sparse: None,
            extensions: None,
            extras: gj::Extras::default(),
        });
        acc_idx
    }

    pub(super) fn finalize_buffer(&mut self) {
        if !self.buffer_data.is_empty() {
            self.root.buffers.push(gj::Buffer {
                byte_length: gj::validation::USize64(self.buffer_data.len() as u64),
                name: None,
                uri: None,
                extensions: None,
                extras: gj::Extras::default(),
            });
        }
    }

    // -- GLB assembly --------------------------------------------------------

    pub(super) fn to_glb(&self) -> Result<Vec<u8>, GltfError> {
        let json_bytes = self
            .root
            .to_vec()
            .map_err(|e| GltfError::ExportError(format!("JSON serialization failed: {e}")))?;

        let json_pad = (4 - (json_bytes.len() % 4)) % 4;
        let json_chunk_len = json_bytes.len() + json_pad;

        let bin_pad = (4 - (self.buffer_data.len() % 4)) % 4;
        let bin_chunk_len = self.buffer_data.len() + bin_pad;

        let has_bin = !self.buffer_data.is_empty();
        let total_length = 12 + 8 + json_chunk_len + if has_bin { 8 + bin_chunk_len } else { 0 };

        let mut glb = Vec::with_capacity(total_length);

        // Header
        glb.extend_from_slice(&0x46546C67u32.to_le_bytes()); // magic "glTF"
        glb.extend_from_slice(&2u32.to_le_bytes()); // version
        glb.extend_from_slice(&(total_length as u32).to_le_bytes());

        // JSON chunk
        glb.extend_from_slice(&(json_chunk_len as u32).to_le_bytes());
        glb.extend_from_slice(&0x4E4F534Au32.to_le_bytes()); // "JSON"
        glb.extend_from_slice(&json_bytes);
        glb.extend(std::iter::repeat_n(b' ', json_pad));

        // BIN chunk
        if has_bin {
            glb.extend_from_slice(&(bin_chunk_len as u32).to_le_bytes());
            glb.extend_from_slice(&0x004E4942u32.to_le_bytes()); // "BIN\0"
            glb.extend_from_slice(&self.buffer_data);
            glb.extend(std::iter::repeat_n(0u8, bin_pad));
        }

        Ok(glb)
    }
}

// ---------------------------------------------------------------------------
// Helper functions
// ---------------------------------------------------------------------------

fn encode_texture_to_png(texture: &CpuTexture) -> Result<Vec<u8>, GltfError> {
    let img = image::ImageBuffer::<image::Rgba<u8>, _>::from_raw(
        texture.width,
        texture.height,
        texture.data.clone(),
    )
    .ok_or_else(|| GltfError::ExportError("invalid texture dimensions for PNG encoding".into()))?;

    let mut png_bytes = Vec::new();
    img.write_to(&mut Cursor::new(&mut png_bytes), image::ImageFormat::Png)
        .map_err(|e| GltfError::ExportError(format!("PNG encoding failed: {e}")))?;

    Ok(png_bytes)
}

fn map_attribute_format(
    format: VertexAttributeFormat,
) -> (gj::accessor::ComponentType, gj::accessor::Type) {
    match format {
        VertexAttributeFormat::Float => {
            (gj::accessor::ComponentType::F32, gj::accessor::Type::Scalar)
        }
        VertexAttributeFormat::Float2 => {
            (gj::accessor::ComponentType::F32, gj::accessor::Type::Vec2)
        }
        VertexAttributeFormat::Float3 => {
            (gj::accessor::ComponentType::F32, gj::accessor::Type::Vec3)
        }
        VertexAttributeFormat::Float4 => {
            (gj::accessor::ComponentType::F32, gj::accessor::Type::Vec4)
        }
        VertexAttributeFormat::Uint4 => {
            (gj::accessor::ComponentType::U16, gj::accessor::Type::Vec4)
        }
        VertexAttributeFormat::Unorm8x4 => {
            (gj::accessor::ComponentType::U8, gj::accessor::Type::Vec4)
        }
        VertexAttributeFormat::Int | VertexAttributeFormat::Uint => {
            (gj::accessor::ComponentType::U32, gj::accessor::Type::Scalar)
        }
        VertexAttributeFormat::Int2 | VertexAttributeFormat::Uint2 => {
            (gj::accessor::ComponentType::U32, gj::accessor::Type::Vec2)
        }
        VertexAttributeFormat::Int3 | VertexAttributeFormat::Uint3 => {
            (gj::accessor::ComponentType::U32, gj::accessor::Type::Vec3)
        }
        VertexAttributeFormat::Int4 => (gj::accessor::ComponentType::U32, gj::accessor::Type::Vec4),
        VertexAttributeFormat::Snorm8x4 => {
            (gj::accessor::ComponentType::I8, gj::accessor::Type::Vec4)
        }
    }
}

fn map_semantic(semantic: VertexAttributeSemantic) -> gj::mesh::Semantic {
    match semantic {
        VertexAttributeSemantic::Position => gj::mesh::Semantic::Positions,
        VertexAttributeSemantic::Normal => gj::mesh::Semantic::Normals,
        VertexAttributeSemantic::Tangent => gj::mesh::Semantic::Tangents,
        VertexAttributeSemantic::TexCoord0 => gj::mesh::Semantic::TexCoords(0),
        VertexAttributeSemantic::TexCoord1 => gj::mesh::Semantic::TexCoords(1),
        VertexAttributeSemantic::Color => gj::mesh::Semantic::Colors(0),
        VertexAttributeSemantic::Joints => gj::mesh::Semantic::Joints(0),
        VertexAttributeSemantic::Weights => gj::mesh::Semantic::Weights(0),
    }
}

fn map_topology(topology: PrimitiveTopology) -> gj::mesh::Mode {
    match topology {
        PrimitiveTopology::PointList => gj::mesh::Mode::Points,
        PrimitiveTopology::LineList => gj::mesh::Mode::Lines,
        PrimitiveTopology::LineStrip => gj::mesh::Mode::LineStrip,
        PrimitiveTopology::TriangleList => gj::mesh::Mode::Triangles,
        PrimitiveTopology::TriangleStrip => gj::mesh::Mode::TriangleStrip,
    }
}

fn map_mag_filter(mode: FilterMode) -> gj::texture::MagFilter {
    match mode {
        FilterMode::Nearest => gj::texture::MagFilter::Nearest,
        FilterMode::Linear => gj::texture::MagFilter::Linear,
    }
}

fn map_min_filter(min: FilterMode, mipmap: FilterMode) -> gj::texture::MinFilter {
    match (min, mipmap) {
        (FilterMode::Nearest, FilterMode::Nearest) => gj::texture::MinFilter::NearestMipmapNearest,
        (FilterMode::Nearest, FilterMode::Linear) => gj::texture::MinFilter::NearestMipmapLinear,
        (FilterMode::Linear, FilterMode::Nearest) => gj::texture::MinFilter::LinearMipmapNearest,
        (FilterMode::Linear, FilterMode::Linear) => gj::texture::MinFilter::LinearMipmapLinear,
    }
}

fn map_wrapping(mode: AddressMode) -> gj::texture::WrappingMode {
    match mode {
        AddressMode::ClampToEdge => gj::texture::WrappingMode::ClampToEdge,
        AddressMode::Repeat => gj::texture::WrappingMode::Repeat,
        AddressMode::MirrorRepeat => gj::texture::WrappingMode::MirroredRepeat,
        AddressMode::ClampToBorder => gj::texture::WrappingMode::ClampToEdge,
    }
}

fn map_interpolation(interp: Interpolation) -> gj::animation::Interpolation {
    match interp {
        Interpolation::Linear => gj::animation::Interpolation::Linear,
        Interpolation::Step => gj::animation::Interpolation::Step,
        Interpolation::CubicSpline => gj::animation::Interpolation::CubicSpline,
    }
}

/// Build a JSON array of f32 values (for accessor min/max).
fn json_f32_array(values: &[f32]) -> gj::Value {
    gj::Value::Array(values.iter().map(|&v| gj::Value::from(v as f64)).collect())
}

/// Compute min/max for a POSITION attribute from interleaved vertex data.
fn compute_position_min_max(
    vertex_data: &[u8],
    stride: u32,
    offset: u32,
    vertex_count: u32,
) -> (Option<gj::Value>, Option<gj::Value>) {
    if vertex_count == 0 || vertex_data.is_empty() {
        return (None, None);
    }

    let mut min = [f32::MAX; 3];
    let mut max = [f32::MIN; 3];

    for v in 0..vertex_count as usize {
        let base = v * stride as usize + offset as usize;
        if base + 12 > vertex_data.len() {
            break;
        }
        for c in 0..3 {
            let byte_offset = base + c * 4;
            let val = f32::from_le_bytes([
                vertex_data[byte_offset],
                vertex_data[byte_offset + 1],
                vertex_data[byte_offset + 2],
                vertex_data[byte_offset + 3],
            ]);
            if val < min[c] {
                min[c] = val;
            }
            if val > max[c] {
                max[c] = val;
            }
        }
    }

    (Some(json_f32_array(&min)), Some(json_f32_array(&max)))
}
