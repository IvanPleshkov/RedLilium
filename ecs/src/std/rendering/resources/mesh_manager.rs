//! GPU mesh management.

use std::collections::HashMap;
use std::sync::Arc;

use redlilium_graphics::{CpuMesh, GraphicsDevice, GraphicsError, Mesh};

/// Resource for managing GPU meshes by name.
///
/// Holds a reference to the [`GraphicsDevice`] and caches created meshes
/// by name for reuse. Also enables serialization of `Arc<Mesh>` references
/// by mapping between mesh names and GPU mesh handles.
pub struct MeshManager {
    device: Arc<GraphicsDevice>,
    meshes: HashMap<String, Arc<Mesh>>,
    /// Cached local-space AABBs keyed by mesh name.
    aabbs: HashMap<String, redlilium_core::math::Aabb>,
}

impl MeshManager {
    /// Create a new mesh manager for the given device.
    pub fn new(device: Arc<GraphicsDevice>) -> Self {
        Self {
            device,
            meshes: HashMap::new(),
            aabbs: HashMap::new(),
        }
    }

    /// Get the graphics device.
    pub fn device(&self) -> &Arc<GraphicsDevice> {
        &self.device
    }

    // --- Mesh creation & lookup ---

    /// Create a GPU mesh from CPU data.
    pub fn create_mesh(&mut self, cpu_mesh: &CpuMesh) -> Result<Arc<Mesh>, GraphicsError> {
        let aabb = cpu_mesh.compute_aabb();
        let mesh = self.device.create_mesh_from_cpu(cpu_mesh)?;
        if let Some(label) = mesh.label() {
            self.meshes.insert(label.to_owned(), Arc::clone(&mesh));
            if let Some(aabb) = aabb {
                self.aabbs.insert(label.to_owned(), aabb);
            }
        }
        Ok(mesh)
    }

    /// Look up a previously created mesh by name.
    pub fn get_mesh(&self, name: &str) -> Option<&Arc<Mesh>> {
        self.meshes.get(name)
    }

    /// Insert a mesh into the cache under a given name.
    pub fn insert_mesh(&mut self, name: impl Into<String>, mesh: Arc<Mesh>) {
        self.meshes.insert(name.into(), mesh);
    }

    /// Remove a mesh from the cache by name, returning it if present.
    pub fn remove_mesh(&mut self, name: &str) -> Option<Arc<Mesh>> {
        self.meshes.remove(name)
    }

    /// Find the registered name for a mesh by Arc pointer identity.
    pub fn find_name(&self, mesh: &Arc<Mesh>) -> Option<&str> {
        self.meshes
            .iter()
            .find(|(_, v)| Arc::ptr_eq(v, mesh))
            .map(|(k, _)| k.as_str())
    }

    // --- AABB ---

    /// Look up the cached local-space AABB for a mesh by Arc pointer identity.
    pub fn get_aabb_by_mesh(&self, mesh: &Arc<Mesh>) -> Option<redlilium_core::math::Aabb> {
        let name = self.find_name(mesh)?;
        self.aabbs.get(name).copied()
    }

    // --- Iteration ---

    /// Get a reference to all cached meshes.
    pub fn meshes(&self) -> &HashMap<String, Arc<Mesh>> {
        &self.meshes
    }

    /// Iterate over all cached mesh names.
    pub fn mesh_names(&self) -> impl Iterator<Item = &str> {
        self.meshes.keys().map(|s| s.as_str())
    }

    /// Returns the number of cached meshes.
    pub fn mesh_count(&self) -> usize {
        self.meshes.len()
    }
}
