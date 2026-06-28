//! The mesh loader. `Asset = Mesh` (the GPU resource), `Deps = Arc<VertexLayout>`
//! (the shared layout to bind — resolved and injected by the mesh manager).
//!
//! `pipeline` composes stages per source:
//! - `File`      → `[read, deserialize, upload]` — a serialized [`CpuMeshData`]
//!   blob (bincode); the vertex layout is *not* in the blob, it's a separate
//!   shared asset referenced by the mesh's DB record.
//! - `Generated` → `[generate, upload]` — procedural geometry, no IO.
//!
//! Both end in the same upload stage: combine the layout-less [`CpuMeshData`] with
//! the injected shared `Arc<VertexLayout>` into a `CpuMesh`, then upload it to a
//! resident [`Mesh`] through the frame graph. Because the layout `Arc` is the
//! shared one, the GPU mesh binds the same layout as everything else (which is
//! what the renderer batches on).

use std::sync::Arc;

use redlilium_assets::{
    AnyAsset, AssetError, AssetLoader, AssetPath, AssetSource, AssetStage, Executor, GpuValue,
    Guid, LoadEnv, StageFuture,
};
use redlilium_core::mesh::{CpuMeshData, VertexLayout, generators};
use redlilium_graphics::{GraphicsDevice, Mesh};
use redlilium_vfs::Vfs;

/// Procedural mesh generators. Float params are stored as IEEE-754 bits so the
/// identity (a cache key) is exact `Hash + Eq` and deterministic across runs.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum MeshGenerator {
    Cube {
        half_bits: u32,
    },
    Sphere {
        radius_bits: u32,
        segments: u32,
        rings: u32,
    },
    Quad {
        half_w_bits: u32,
        half_h_bits: u32,
    },
}

impl MeshGenerator {
    pub fn cube(half: f32) -> Self {
        Self::Cube {
            half_bits: half.to_bits(),
        }
    }
    pub fn sphere(radius: f32, segments: u32, rings: u32) -> Self {
        Self::Sphere {
            radius_bits: radius.to_bits(),
            segments,
            rings,
        }
    }
    pub fn quad(half_w: f32, half_h: f32) -> Self {
        Self::Quad {
            half_w_bits: half_w.to_bits(),
            half_h_bits: half_h.to_bits(),
        }
    }

    /// The vertex layout this generator's geometry uses — cheap (no geometry is
    /// built). The mesh manager interns this to bind the shared layout `Arc`.
    pub fn layout(&self) -> Arc<VertexLayout> {
        match self {
            Self::Cube { .. } => VertexLayout::position_normal(),
            Self::Sphere { .. } => VertexLayout::position_normal_uv(),
            Self::Quad { .. } => VertexLayout::position_uv(),
        }
    }

    /// Synthesize the geometry as layout-less mesh data.
    pub fn build(&self) -> CpuMeshData {
        let cpu = match *self {
            Self::Cube { half_bits } => generators::generate_cube(f32::from_bits(half_bits)),
            Self::Sphere {
                radius_bits,
                segments,
                rings,
            } => generators::generate_sphere(f32::from_bits(radius_bits), segments, rings),
            Self::Quad {
                half_w_bits,
                half_h_bits,
            } => {
                generators::generate_quad(f32::from_bits(half_w_bits), f32::from_bits(half_h_bits))
            }
        };
        CpuMeshData::from_cpu_mesh(&cpu)
    }
}

/// Identity of a mesh asset. `File` resolves a serialized blob from `guid` via the
/// DB (its layout is a separate referenced asset); `Generated` is procedural (its
/// layout comes from the generator). Serialized in `MeshRenderer` / prefabs.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum MeshSource {
    File(Guid),
    Generated(MeshGenerator),
}

impl AssetSource for MeshSource {
    fn file_guid(&self) -> Option<Guid> {
        match self {
            Self::File(guid) => Some(*guid),
            Self::Generated(_) => None,
        }
    }
}

/// Loads a mesh (file blob or procedural) and uploads it as a GPU [`Mesh`] bound
/// to the injected shared layout.
pub struct MeshLoader;

impl AssetLoader for MeshLoader {
    const NAME: &'static str = "mesh";
    const EXTENSIONS: &'static [&'static str] = &["rmesh"];
    type Source = MeshSource;
    type Asset = Mesh;
    /// The shared vertex layout to bind (resolved by the mesh manager).
    type Deps = Arc<VertexLayout>;

    fn pipeline(
        source: &MeshSource,
        deps: &Arc<VertexLayout>,
        env: &LoadEnv,
    ) -> Vec<Box<dyn AssetStage>> {
        let mut stages: Vec<Box<dyn AssetStage>> = Vec::new();
        match source {
            // No IO — generate on the CPU, then upload.
            MeshSource::Generated(generator) => {
                stages.push(Box::new(GenerateStage {
                    generator: generator.clone(),
                }));
            }
            // Read the blob, then deserialize it. (Read omitted only if the path
            // didn't resolve — left to fail in deserialize.)
            MeshSource::File(_) => {
                if let Some(path) = &env.path {
                    stages.push(Box::new(ReadBlobStage {
                        path: path.clone(),
                        vfs: env.vfs.clone(),
                    }));
                }
                stages.push(Box::new(DeserializeMeshStage));
            }
        }
        stages.push(Box::new(UploadMeshStage {
            device: env.device.clone(),
            layout: deps.clone(),
        }));
        stages
    }
}

/// IO stage: read the mesh blob's bytes.
struct ReadBlobStage {
    path: AssetPath,
    vfs: Vfs,
}

impl AssetStage for ReadBlobStage {
    fn executor(&self) -> Executor {
        Executor::Io
    }
    fn run_async(&self, _input: AnyAsset) -> StageFuture {
        let path = self.path.clone();
        let vfs = self.vfs.clone();
        Box::pin(async move {
            let raw = format!("{}/{}", path.mount, path.path);
            let bytes = vfs
                .read(&raw)
                .await
                .map_err(|e| AssetError::Io(e.to_string()))?;
            Ok(Box::new(bytes) as AnyAsset)
        })
    }
}

/// CPU stage: deserialize the bincode blob into [`CpuMeshData`] (no layout yet).
struct DeserializeMeshStage;

impl AssetStage for DeserializeMeshStage {
    fn executor(&self) -> Executor {
        Executor::Cpu
    }
    fn run_async(&self, input: AnyAsset) -> StageFuture {
        Box::pin(async move {
            let bytes = input
                .downcast::<Vec<u8>>()
                .map_err(|_| AssetError::Decode("mesh: expected file bytes".into()))?;
            let data: CpuMeshData = bincode::deserialize(&bytes)
                .map_err(|e| AssetError::Decode(format!("mesh: bincode: {e}")))?;
            Ok(Box::new(data) as AnyAsset)
        })
    }
}

/// CPU stage: synthesize procedural geometry as [`CpuMeshData`] (no input).
struct GenerateStage {
    generator: MeshGenerator,
}

impl AssetStage for GenerateStage {
    fn executor(&self) -> Executor {
        Executor::Cpu
    }
    fn run_async(&self, _input: AnyAsset) -> StageFuture {
        let generator = self.generator.clone();
        // Heavy procedural geometry would `yield_now().await` periodically.
        Box::pin(async move { Ok(Box::new(generator.build()) as AnyAsset) })
    }
}

/// GPU stage: combine the data with the injected shared layout into a `CpuMesh`,
/// then upload it to a resident [`Mesh`] through the frame graph.
struct UploadMeshStage {
    device: Arc<GraphicsDevice>,
    layout: Arc<VertexLayout>,
}

impl AssetStage for UploadMeshStage {
    fn executor(&self) -> Executor {
        Executor::Gpu
    }
    fn run_gpu(
        &self,
        input: AnyAsset,
    ) -> Result<(GpuValue, Vec<redlilium_graphics::TransferOperation>), AssetError> {
        let data = *input
            .downcast::<CpuMeshData>()
            .map_err(|_| AssetError::Decode("mesh: upload stage expected CpuMeshData".into()))?;
        let cpu = data.into_cpu_mesh(Arc::clone(&self.layout));
        let (mesh, ops) = self.device.create_mesh_deferred(&cpu)?;
        Ok((Box::new(mesh) as GpuValue, ops))
    }
}
