use std::future::Future;
use std::pin::Pin;

use crate::VfsError;

/// A boxed, `Send` future returning a `Result`.
///
/// All [`VfsProvider`] methods return this type. The futures are `Send + 'static`
/// so they can be spawned on any async runtime (e.g. via `IoRunner::run()`).
pub type VfsFuture<T> = Pin<Box<dyn Future<Output = Result<T, VfsError>> + Send>>;

/// Trait for virtual file system backends.
///
/// Providers implement byte-level I/O operations. The returned futures do NOT
/// drive themselves — the caller is responsible for running them on an async
/// runtime (e.g. `IoRuntime` from the ECS crate).
///
/// # Read vs Write
///
/// All providers must implement read operations (`read`, `exists`, `list_dir`).
/// Write operations (`write`, `delete`, `create_dir`) have default implementations
/// that return [`VfsError::ReadOnly`]. Providers that support writes (e.g.
/// filesystem, memory) override these methods and return `false` from
/// [`is_read_only()`](VfsProvider::is_read_only).
///
/// # Path Contract
///
/// Paths passed to provider methods are already normalized by the [`Vfs`](crate::Vfs)
/// router: forward slashes, no leading/trailing slashes, no `..` or `.` segments.
/// The path is relative to the provider's root (the source prefix has been stripped).
pub trait VfsProvider: Send + Sync + 'static {
    // --- Read operations (required) ---

    /// Read the entire contents of a file at the given path.
    fn read(&self, path: &str) -> VfsFuture<Vec<u8>>;

    /// Check whether a file exists at the given path.
    fn exists(&self, path: &str) -> VfsFuture<bool>;

    /// List the immediate children of a directory.
    ///
    /// Returns file and directory names (not full paths).
    /// Returns an empty vec for non-existent directories.
    fn list_dir(&self, path: &str) -> VfsFuture<Vec<String>>;

    // --- Write operations (optional, default returns ReadOnly) ---

    /// Whether this provider is read-only.
    ///
    /// Returns `true` by default. Providers that support writes should
    /// override this to return `false`.
    fn is_read_only(&self) -> bool {
        true
    }

    /// Write data to a file, creating or overwriting it.
    fn write(&self, _path: &str, _data: Vec<u8>) -> VfsFuture<()> {
        Box::pin(async { Err(VfsError::ReadOnly) })
    }

    /// Delete a file at the given path.
    fn delete(&self, _path: &str) -> VfsFuture<()> {
        Box::pin(async { Err(VfsError::ReadOnly) })
    }

    /// Create a directory at the given path.
    fn create_dir(&self, _path: &str) -> VfsFuture<()> {
        Box::pin(async { Err(VfsError::ReadOnly) })
    }
}
