//! Resource handles for the render graph.

use std::sync::atomic::{AtomicU32, Ordering};

/// Generic handle to a resource in the render graph.
///
/// Handles use a generation counter to detect stale references.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ResourceHandle {
    /// Index into the resource array.
    index: u32,
    /// Generation counter for validation.
    generation: u32,
}

impl ResourceHandle {
    /// Create a new resource handle.
    pub(crate) fn new(index: u32) -> Self {
        static GENERATION: AtomicU32 = AtomicU32::new(0);
        Self {
            index,
            generation: GENERATION.fetch_add(1, Ordering::Relaxed),
        }
    }

    /// Get the index of this resource.
    pub fn index(&self) -> u32 {
        self.index
    }

    /// Get the generation of this handle.
    pub fn generation(&self) -> u32 {
        self.generation
    }
}

/// Resource access type for dependency tracking.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[allow(dead_code)] // Part of planned API
pub enum ResourceAccess {
    /// Read-only access.
    Read,
    /// Write-only access.
    Write,
    /// Read and write access.
    ReadWrite,
}

#[allow(dead_code)] // Part of planned API
impl ResourceAccess {
    /// Check if this access includes reading.
    pub fn reads(&self) -> bool {
        matches!(self, Self::Read | Self::ReadWrite)
    }

    /// Check if this access includes writing.
    pub fn writes(&self) -> bool {
        matches!(self, Self::Write | Self::ReadWrite)
    }
}
