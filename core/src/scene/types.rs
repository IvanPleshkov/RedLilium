//! Scene graph data types.
//!
//! All types use plain arrays (`[f32; 3]`, `[f32; 4]`, etc.) instead of
//! math library types to keep the core crate dependency-free of `glam`.

use crate::mesh::CpuMesh;

/// Node transform decomposed into translation, rotation, and scale.
///
/// Uses plain arrays for portability. Convert to `glam` types as needed:
/// `Vec3::from(t.translation)`, `Quat::from_array(t.rotation)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NodeTransform {
    /// Translation [x, y, z].
    pub translation: [f32; 3],
    /// Rotation quaternion [x, y, z, w].
    pub rotation: [f32; 4],
    /// Scale [x, y, z].
    pub scale: [f32; 3],
}

impl NodeTransform {
    /// Identity transform: no translation, identity rotation, unit scale.
    pub const IDENTITY: Self = Self {
        translation: [0.0, 0.0, 0.0],
        rotation: [0.0, 0.0, 0.0, 1.0],
        scale: [1.0, 1.0, 1.0],
    };

    /// Returns this transform with a different translation.
    #[must_use]
    pub const fn with_translation(mut self, translation: [f32; 3]) -> Self {
        self.translation = translation;
        self
    }

    /// Returns this transform with a different rotation.
    #[must_use]
    pub const fn with_rotation(mut self, rotation: [f32; 4]) -> Self {
        self.rotation = rotation;
        self
    }

    /// Returns this transform with a different scale.
    #[must_use]
    pub const fn with_scale(mut self, scale: [f32; 3]) -> Self {
        self.scale = scale;
        self
    }
}

impl Default for NodeTransform {
    fn default() -> Self {
        Self::IDENTITY
    }
}

/// A node in a scene graph tree.
///
/// Nodes form a recursive tree structure. Each node has a local transform,
/// optional references to resources (meshes, cameras, skins), and child nodes.
/// Resource references use indices into the owning [`Scene`]'s arrays.
#[derive(Debug)]
pub struct SceneNode {
    /// Node name, if any.
    pub name: Option<String>,
    /// Local transform relative to parent.
    pub transform: NodeTransform,
    /// Indices into [`Scene::meshes`].
    /// Empty if the node carries no mesh.
    pub meshes: Vec<usize>,
    /// Index into [`Scene::cameras`], if this node has a camera.
    pub camera: Option<usize>,
    /// Index into [`Scene::skins`], if this node has a skin.
    pub skin: Option<usize>,
    /// Child nodes forming the sub-tree.
    pub children: Vec<SceneNode>,
}

impl SceneNode {
    /// Creates a new node with default (identity) transform and no attachments.
    pub fn new() -> Self {
        Self {
            name: None,
            transform: NodeTransform::IDENTITY,
            meshes: Vec::new(),
            camera: None,
            skin: None,
            children: Vec::new(),
        }
    }

    /// Set the node name.
    #[must_use]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Set the local transform.
    #[must_use]
    pub fn with_transform(mut self, transform: NodeTransform) -> Self {
        self.transform = transform;
        self
    }

    /// Set the mesh indices.
    #[must_use]
    pub fn with_meshes(mut self, meshes: Vec<usize>) -> Self {
        self.meshes = meshes;
        self
    }

    /// Set the camera index.
    #[must_use]
    pub fn with_camera(mut self, camera: usize) -> Self {
        self.camera = Some(camera);
        self
    }

    /// Set the skin index.
    #[must_use]
    pub fn with_skin(mut self, skin: usize) -> Self {
        self.skin = Some(skin);
        self
    }

    /// Set the child nodes.
    #[must_use]
    pub fn with_children(mut self, children: Vec<SceneNode>) -> Self {
        self.children = children;
        self
    }
}

impl Default for SceneNode {
    fn default() -> Self {
        Self::new()
    }
}

/// A scene containing a tree of nodes and all resources they reference.
///
/// Represents a single scene (one of potentially many in a document).
/// Nodes are organized as a forest of trees (multiple root nodes).
/// Resource arrays (meshes, cameras, skins) are owned by the scene
/// so that node indices resolve locally.
#[derive(Debug)]
pub struct Scene {
    /// Scene name, if any.
    pub name: Option<String>,
    /// Root nodes of the scene.
    pub nodes: Vec<SceneNode>,
    /// All meshes referenced by nodes in this scene.
    pub meshes: Vec<CpuMesh>,
    /// All cameras referenced by nodes in this scene.
    pub cameras: Vec<SceneCamera>,
    /// All skins referenced by nodes in this scene.
    pub skins: Vec<SceneSkin>,
}

impl Scene {
    /// Creates a new empty scene.
    pub fn new() -> Self {
        Self {
            name: None,
            nodes: Vec::new(),
            meshes: Vec::new(),
            cameras: Vec::new(),
            skins: Vec::new(),
        }
    }

    /// Set the scene name.
    #[must_use]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Set the root nodes.
    #[must_use]
    pub fn with_nodes(mut self, nodes: Vec<SceneNode>) -> Self {
        self.nodes = nodes;
        self
    }

    /// Set the meshes.
    #[must_use]
    pub fn with_meshes(mut self, meshes: Vec<CpuMesh>) -> Self {
        self.meshes = meshes;
        self
    }

    /// Set the cameras.
    #[must_use]
    pub fn with_cameras(mut self, cameras: Vec<SceneCamera>) -> Self {
        self.cameras = cameras;
        self
    }

    /// Set the skins.
    #[must_use]
    pub fn with_skins(mut self, skins: Vec<SceneSkin>) -> Self {
        self.skins = skins;
        self
    }
}

impl Default for Scene {
    fn default() -> Self {
        Self::new()
    }
}

// -- Cameras --

/// A camera definition.
#[derive(Debug, Clone)]
pub struct SceneCamera {
    /// Camera name.
    pub name: Option<String>,
    /// Projection type and parameters.
    pub projection: CameraProjection,
}

/// Camera projection parameters.
#[derive(Debug, Clone)]
pub enum CameraProjection {
    /// Perspective projection.
    Perspective {
        /// Vertical field of view in radians.
        yfov: f32,
        /// Aspect ratio (width/height), if specified.
        aspect: Option<f32>,
        /// Near clipping plane distance.
        znear: f32,
        /// Far clipping plane distance, if specified.
        zfar: Option<f32>,
    },
    /// Orthographic projection.
    Orthographic {
        /// Horizontal magnification.
        xmag: f32,
        /// Vertical magnification.
        ymag: f32,
        /// Near clipping plane distance.
        znear: f32,
        /// Far clipping plane distance.
        zfar: f32,
    },
}

// -- Skins --

/// A skin for skeletal animation.
#[derive(Debug, Clone)]
pub struct SceneSkin {
    /// Skin name.
    pub name: Option<String>,
    /// Joint node indices (referencing nodes in the scene).
    pub joints: Vec<usize>,
    /// Inverse bind matrices (column-major 4x4, one per joint).
    pub inverse_bind_matrices: Vec<[f32; 16]>,
    /// Root skeleton node index, if specified.
    pub skeleton: Option<usize>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn node_transform_default_is_identity() {
        let t = NodeTransform::default();
        assert_eq!(t, NodeTransform::IDENTITY);
        assert_eq!(t.translation, [0.0, 0.0, 0.0]);
        assert_eq!(t.rotation, [0.0, 0.0, 0.0, 1.0]);
        assert_eq!(t.scale, [1.0, 1.0, 1.0]);
    }

    #[test]
    fn node_transform_builder() {
        let t = NodeTransform::IDENTITY
            .with_translation([1.0, 2.0, 3.0])
            .with_rotation([0.0, 0.707, 0.0, 0.707])
            .with_scale([2.0, 2.0, 2.0]);
        assert_eq!(t.translation, [1.0, 2.0, 3.0]);
        assert_eq!(t.rotation, [0.0, 0.707, 0.0, 0.707]);
        assert_eq!(t.scale, [2.0, 2.0, 2.0]);
    }

    #[test]
    fn scene_node_default() {
        let node = SceneNode::new();
        assert!(node.name.is_none());
        assert_eq!(node.transform, NodeTransform::IDENTITY);
        assert!(node.meshes.is_empty());
        assert!(node.camera.is_none());
        assert!(node.skin.is_none());
        assert!(node.children.is_empty());
    }

    #[test]
    fn scene_node_builder() {
        let child = SceneNode::new().with_name("child");
        let node = SceneNode::new()
            .with_name("root")
            .with_meshes(vec![0, 1])
            .with_camera(0)
            .with_children(vec![child]);
        assert_eq!(node.name.as_deref(), Some("root"));
        assert_eq!(node.meshes, vec![0, 1]);
        assert_eq!(node.camera, Some(0));
        assert_eq!(node.children.len(), 1);
        assert_eq!(node.children[0].name.as_deref(), Some("child"));
    }

    #[test]
    fn scene_builder() {
        let scene = Scene::new()
            .with_name("My Scene")
            .with_nodes(vec![SceneNode::new()]);
        assert_eq!(scene.name.as_deref(), Some("My Scene"));
        assert_eq!(scene.nodes.len(), 1);
    }
}
