//! Built-in asset loaders. These are the loaders whose dependencies live at or
//! below this crate (core + graphics); loaders needing higher-level crates (a
//! prefab loader needing `ecs`, an audio loader, etc.) live in those crates and
//! register into the processor builder the same way.

mod mesh;

pub use mesh::{MeshGenerator, MeshLoader, MeshSource};
