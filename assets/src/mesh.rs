//! The built-in mesh loader and its stages. `pipeline` composes the stages per
//! source: `File -> [read, decode, upload]`, `Generated -> [generate, upload]`.

use std::sync::Arc;

use redlilium_core::mesh::{CpuMesh, generators};
use redlilium_graphics::{GraphicsDevice, Mesh};
use redlilium_vfs::Vfs;
use serde::{Deserialize, Serialize};

use crate::db::AssetPath;
use crate::error::AssetError;
use crate::loader::AssetLoader;
use crate::source::{AssetSource, Guid};
use crate::stage::{AnyAsset, AssetStage, Executor, GpuValue, LoadEnv, StageFuture};

// ---------------------------------------------------------------------------
// Source
// ---------------------------------------------------------------------------

/// Procedural mesh generators. Float params stored as IEEE-754 bits so identity
/// (the cache key) is exact `Hash + Eq` and deterministic across runs.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
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

    /// Synthesize the CPU mesh.
    pub fn build(&self) -> CpuMesh {
        match *self {
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
        }
    }
}

/// Where a mesh comes from. Serialized in `MeshRenderer` and used as identity.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MeshSource {
    /// A primitive of a file asset; the path resolves via the DB from `guid`.
    File { guid: Guid, primitive: u32 },
    /// Procedurally generated.
    Generated(MeshGenerator),
}

impl AssetSource for MeshSource {
    fn file_guid(&self) -> Option<Guid> {
        match self {
            Self::File { guid, .. } => Some(*guid),
            Self::Generated(_) => None,
        }
    }
}

// ---------------------------------------------------------------------------
// Loader
// ---------------------------------------------------------------------------

/// The mesh loader. `Asset = Mesh` (the GPU resource).
pub struct MeshLoader;

impl AssetLoader for MeshLoader {
    const NAME: &'static str = "mesh";
    type Source = MeshSource;
    type Asset = Mesh;

    fn pipeline(&self, source: &MeshSource, env: &LoadEnv) -> Vec<Box<dyn AssetStage>> {
        match source {
            // No IO — generate on the CPU, then upload.
            MeshSource::Generated(generator) => vec![
                Box::new(GenerateStage {
                    generator: generator.clone(),
                }) as Box<dyn AssetStage>,
                Box::new(UploadMeshStage {
                    device: env.device.clone(),
                }),
            ],
            // Read the file, decode it, then upload. (Read omitted only if the
            // path didn't resolve — left to fail in decode.)
            MeshSource::File { primitive, .. } => {
                let mut stages: Vec<Box<dyn AssetStage>> = Vec::new();
                if let Some(path) = &env.path {
                    stages.push(Box::new(ReadFileStage {
                        path: path.clone(),
                        vfs: env.vfs.clone(),
                    }));
                }
                stages.push(Box::new(DecodeGltfStage {
                    primitive: *primitive,
                }));
                stages.push(Box::new(UploadMeshStage {
                    device: env.device.clone(),
                }));
                stages
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Stages
// ---------------------------------------------------------------------------

/// IO stage: read the primary file bytes.
struct ReadFileStage {
    path: AssetPath,
    vfs: Vfs,
}

impl AssetStage for ReadFileStage {
    fn executor(&self) -> Executor {
        Executor::Io
    }
    fn run_async(&self, _input: AnyAsset) -> StageFuture {
        let path = self.path.clone();
        let vfs = self.vfs.clone();
        Box::pin(async move {
            // Real glTF read would also fetch referenced .bin/image files here.
            let raw = format!("{}/{}", path.mount, path.path);
            let bytes = vfs
                .read(&raw)
                .await
                .map_err(|e| AssetError::Io(e.to_string()))?;
            Ok(Box::new(bytes) as AnyAsset)
        })
    }
}

/// CPU stage: decode glTF bytes into a `CpuMesh` (stubbed).
struct DecodeGltfStage {
    primitive: u32,
}

impl AssetStage for DecodeGltfStage {
    fn executor(&self) -> Executor {
        Executor::Cpu
    }
    fn run_async(&self, input: AnyAsset) -> StageFuture {
        let primitive = self.primitive;
        Box::pin(async move {
            let bytes = input
                .downcast::<Vec<u8>>()
                .map_err(|_| AssetError::Decode("decode stage expected file bytes".into()))?;
            let _ = (bytes, primitive);
            // TODO: decode glTF and pick `primitive`.
            Err(AssetError::Decode(
                "glTF mesh decoding not yet wired".into(),
            ))
        })
    }
}

/// CPU stage: synthesize a procedural `CpuMesh` (no input).
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

/// GPU stage: upload a `CpuMesh` to a resident `Mesh` through the frame graph.
struct UploadMeshStage {
    device: Arc<GraphicsDevice>,
}

impl AssetStage for UploadMeshStage {
    fn executor(&self) -> Executor {
        Executor::Gpu
    }
    fn run_gpu(
        &self,
        input: AnyAsset,
    ) -> Result<(GpuValue, Vec<redlilium_graphics::TransferOperation>), AssetError> {
        let cpu = input
            .downcast::<CpuMesh>()
            .map_err(|_| AssetError::Decode("upload stage expected CpuMesh".into()))?;
        let (mesh, ops) = self.device.create_mesh_deferred(&cpu)?;
        Ok((Box::new(mesh) as GpuValue, ops))
    }
}
