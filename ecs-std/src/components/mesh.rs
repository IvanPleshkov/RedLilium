use std::sync::Arc;

use redlilium_core::material::CpuMaterialInstance;
use redlilium_core::mesh::CpuMesh;

/// Links an entity to renderable mesh and material data.
///
/// Stores Arc-wrapped references to CPU-side mesh and material data.
/// The rendering system reads these to create or update GPU resources.
pub struct MeshRenderer {
    /// The mesh data (shared via Arc to avoid expensive clones).
    pub mesh: Arc<CpuMesh>,
    /// The material instance for this mesh.
    pub material: Arc<CpuMaterialInstance>,
}

impl MeshRenderer {
    /// Create a new mesh renderer with the given mesh and material.
    pub fn new(mesh: Arc<CpuMesh>, material: Arc<CpuMaterialInstance>) -> Self {
        Self { mesh, material }
    }
}

impl Clone for MeshRenderer {
    fn clone(&self) -> Self {
        Self {
            mesh: Arc::clone(&self.mesh),
            material: Arc::clone(&self.material),
        }
    }
}

impl std::fmt::Debug for MeshRenderer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MeshRenderer")
            .field("mesh", &format_args!("Arc<CpuMesh>"))
            .field("material", &self.material)
            .finish()
    }
}
