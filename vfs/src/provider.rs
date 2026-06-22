use std::future::Future;
use std::pin::Pin;

use crate::VfsError;

/// A boxed, `Send` future returning a `Result`.
///
/// All [`VfsProvider`] methods return this type. The futures are `Send + 'static`
/// so they can be spawned on any async runtime (e.g. via `IoRunner::run()`).
pub type VfsFuture<T> = Pin<Box<dyn Future<Output = Result<T, VfsError>> + Send>>;

/// A single directory entry with its kind, returned by
/// [`list_dir_entries`](VfsProvider::list_dir_entries).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VfsDirEntry {
    /// The entry name (not a full path).
    pub name: String,
    /// Whether the entry is a directory (`true`) or a file (`false`).
    pub is_dir: bool,
}

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

    /// List the immediate children of a directory **with their kind**
    /// (file vs directory).
    ///
    /// The default implementation falls back to [`list_dir`](Self::list_dir)
    /// and guesses the kind from whether the name contains a `.` — providers
    /// that can report real metadata (filesystem, in-memory) override this so
    /// names like `my.assets` (a directory) or `LICENSE` (a file) classify
    /// correctly.
    fn list_dir_entries(&self, path: &str) -> VfsFuture<Vec<VfsDirEntry>> {
        let names = self.list_dir(path);
        Box::pin(async move {
            Ok(names
                .await?
                .into_iter()
                .map(|name| {
                    let is_dir = !name.contains('.');
                    VfsDirEntry { name, is_dir }
                })
                .collect())
        })
    }

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
