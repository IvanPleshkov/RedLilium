use std::collections::HashMap;
use std::sync::Arc;

use crate::error::VfsError;
use crate::path;
use crate::provider::{VfsFuture, VfsProvider};

/// Virtual file system that routes paths to mounted providers.
///
/// Paths are structured as `"source_name/rest/of/path"`. The first path
/// segment selects the provider. If no source name matches, the default
/// source (if set) is tried with the full path.
///
/// `Clone` is cheap (Arc internals). Thread-safe (`Send + Sync`).
///
/// # Example
///
/// ```ignore
/// let mut vfs = Vfs::new();
/// vfs.mount("assets", FileSystemProvider::new("./assets"));
/// vfs.mount("builtin", MemoryProvider::new());
/// vfs.set_default("assets");
///
/// // Reads from FileSystemProvider at "./assets/textures/brick.png"
/// let bytes = io.run(vfs.read("assets/textures/brick.png")).await;
///
/// // With default source, also reads from assets:
/// let bytes = io.run(vfs.read("textures/brick.png")).await;
/// ```
#[derive(Clone)]
pub struct Vfs {
    inner: Arc<VfsInner>,
}

struct VfsInner {
    sources: HashMap<String, Box<dyn VfsProvider>>,
    default_source: Option<String>,
}

impl Vfs {
    /// Create an empty VFS with no mounted sources.
    pub fn new() -> Self {
        Self {
            inner: Arc::new(VfsInner {
                sources: HashMap::new(),
                default_source: None,
            }),
        }
    }

    /// Mount a provider under the given source name.
    ///
    /// Replaces any previously mounted provider with the same name.
    ///
    /// # Panics
    ///
    /// Panics if the `Vfs` has already been cloned. All mounting must
    /// happen during the configuration phase before sharing the `Vfs`.
    pub fn mount(&mut self, name: impl Into<String>, provider: impl VfsProvider) {
        let inner = Arc::get_mut(&mut self.inner).expect("cannot mount after Vfs has been cloned");
        inner.sources.insert(name.into(), Box::new(provider));
    }

    /// Set the default source name used when a path does not match any mount.
    ///
    /// # Panics
    ///
    /// Panics if the `Vfs` has already been cloned.
    pub fn set_default(&mut self, name: impl Into<String>) {
        let inner =
            Arc::get_mut(&mut self.inner).expect("cannot set default after Vfs has been cloned");
        inner.default_source = Some(name.into());
    }

    /// Read the entire contents of a file.
    ///
    /// The first path segment selects the source provider. Falls back
    /// to the default source if no mount matches.
    pub fn read(&self, raw_path: &str) -> VfsFuture<Vec<u8>> {
        let (provider, resolved_path) = match self.resolve(raw_path) {
            Ok(v) => v,
            Err(e) => return Box::pin(async move { Err(e) }),
        };
        provider.read(&resolved_path)
    }

    /// Check whether a file exists.
    pub fn exists(&self, raw_path: &str) -> VfsFuture<bool> {
        let (provider, resolved_path) = match self.resolve(raw_path) {
            Ok(v) => v,
            Err(e) => return Box::pin(async move { Err(e) }),
        };
        provider.exists(&resolved_path)
    }

    /// List the immediate children of a directory.
    pub fn list_dir(&self, raw_path: &str) -> VfsFuture<Vec<String>> {
        let (provider, resolved_path) = match self.resolve(raw_path) {
            Ok(v) => v,
            Err(e) => return Box::pin(async move { Err(e) }),
        };
        provider.list_dir(&resolved_path)
    }

    /// Write data to a file.
    ///
    /// Returns [`VfsError::ReadOnly`] if the resolved provider does not
    /// support writes.
    pub fn write(&self, raw_path: &str, data: Vec<u8>) -> VfsFuture<()> {
        let (provider, resolved_path) = match self.resolve(raw_path) {
            Ok(v) => v,
            Err(e) => return Box::pin(async move { Err(e) }),
        };
        provider.write(&resolved_path, data)
    }

    /// Delete a file.
    pub fn delete(&self, raw_path: &str) -> VfsFuture<()> {
        let (provider, resolved_path) = match self.resolve(raw_path) {
            Ok(v) => v,
            Err(e) => return Box::pin(async move { Err(e) }),
        };
        provider.delete(&resolved_path)
    }

    /// Create a directory.
    pub fn create_dir(&self, raw_path: &str) -> VfsFuture<()> {
        let (provider, resolved_path) = match self.resolve(raw_path) {
            Ok(v) => v,
            Err(e) => return Box::pin(async move { Err(e) }),
        };
        provider.create_dir(&resolved_path)
    }

    /// Check if the provider for a given path is read-only.
    ///
    /// Returns `Err` if the path cannot be resolved to a provider.
    pub fn is_read_only(&self, raw_path: &str) -> Result<bool, VfsError> {
        let (provider, _) = self.resolve(raw_path)?;
        Ok(provider.is_read_only())
    }

    /// Resolve a raw path to a provider reference and the path within that provider.
    fn resolve(&self, raw_path: &str) -> Result<(&dyn VfsProvider, String), VfsError> {
        let normalized = path::normalize(raw_path)?;
        let (source, rest) = path::split_source(&normalized);

        // Try matching the first segment as a source name
        if let Some(provider) = self.inner.sources.get(source) {
            return Ok((provider.as_ref(), rest.to_owned()));
        }

        // Fall back to default source with the full path
        if let Some(default_name) = &self.inner.default_source
            && let Some(provider) = self.inner.sources.get(default_name)
        {
            return Ok((provider.as_ref(), normalized));
        }

        Err(VfsError::NoSuchSource(source.to_owned()))
    }
}

impl Default for Vfs {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MemoryProvider;
    use std::pin::Pin;
    use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};

    fn poll_ready<T>(mut fut: VfsFuture<T>) -> Result<T, VfsError> {
        let waker = noop_waker();
        let mut cx = Context::from_waker(&waker);
        match Pin::new(&mut fut).poll(&mut cx) {
            Poll::Ready(val) => val,
            Poll::Pending => panic!("expected future to be immediately ready"),
        }
    }

    fn noop_waker() -> Waker {
        fn noop(_: *const ()) {}
        fn clone(p: *const ()) -> RawWaker {
            RawWaker::new(p, &VTABLE)
        }
        static VTABLE: RawWakerVTable = RawWakerVTable::new(clone, noop, noop, noop);
        unsafe { Waker::from_raw(RawWaker::new(std::ptr::null(), &VTABLE)) }
    }

    #[test]
    fn mount_and_read() {
        let mem = MemoryProvider::new();
        mem.insert("hello.txt", b"world".to_vec());

        let mut vfs = Vfs::new();
        vfs.mount("data", mem);

        let result = poll_ready(vfs.read("data/hello.txt")).unwrap();
        assert_eq!(result, b"world");
    }

    #[test]
    fn default_source_fallback() {
        let mem = MemoryProvider::new();
        mem.insert("hello.txt", b"world".to_vec());

        let mut vfs = Vfs::new();
        vfs.mount("data", mem);
        vfs.set_default("data");

        // "hello.txt" doesn't match any source, falls back to "data"
        let result = poll_ready(vfs.read("hello.txt")).unwrap();
        assert_eq!(result, b"world");
    }

    #[test]
    fn no_source_error() {
        let vfs = Vfs::new();
        let result = poll_ready(vfs.read("unknown/file.txt"));
        assert!(matches!(result, Err(VfsError::NoSuchSource(_))));
    }

    #[test]
    fn path_normalization() {
        let mem = MemoryProvider::new();
        mem.insert("a/b.txt", b"ok".to_vec());

        let mut vfs = Vfs::new();
        vfs.mount("data", mem);

        let result = poll_ready(vfs.read("data//a/./b.txt")).unwrap();
        assert_eq!(result, b"ok");
    }

    #[test]
    fn invalid_path_rejected() {
        let mut vfs = Vfs::new();
        vfs.mount("data", MemoryProvider::new());

        let result = poll_ready(vfs.read("data/../secret.txt"));
        assert!(matches!(result, Err(VfsError::InvalidPath(_))));
    }

    #[test]
    fn exists_via_vfs() {
        let mem = MemoryProvider::new();
        mem.insert("file.txt", vec![]);

        let mut vfs = Vfs::new();
        vfs.mount("m", mem);

        assert!(poll_ready(vfs.exists("m/file.txt")).unwrap());
        assert!(!poll_ready(vfs.exists("m/nope.txt")).unwrap());
    }

    #[test]
    fn list_dir_via_vfs() {
        let mem = MemoryProvider::new();
        mem.insert("a.txt", vec![]);
        mem.insert("sub/b.txt", vec![]);

        let mut vfs = Vfs::new();
        vfs.mount("m", mem);

        let entries = poll_ready(vfs.list_dir("m")).unwrap();
        assert!(entries.contains(&"a.txt".to_owned()));
        assert!(entries.contains(&"sub".to_owned()));
    }

    #[test]
    fn write_via_vfs() {
        let mem = MemoryProvider::new();

        let mut vfs = Vfs::new();
        vfs.mount("m", mem);

        poll_ready(vfs.write("m/new.txt", b"hello".to_vec())).unwrap();
        let data = poll_ready(vfs.read("m/new.txt")).unwrap();
        assert_eq!(data, b"hello");
    }

    #[test]
    fn delete_via_vfs() {
        let mem = MemoryProvider::new();
        mem.insert("file.txt", b"data".to_vec());

        let mut vfs = Vfs::new();
        vfs.mount("m", mem);

        poll_ready(vfs.delete("m/file.txt")).unwrap();
        assert!(!poll_ready(vfs.exists("m/file.txt")).unwrap());
    }

    #[test]
    fn is_read_only_check() {
        let mut vfs = Vfs::new();
        vfs.mount("mem", MemoryProvider::new());

        assert!(!vfs.is_read_only("mem/anything").unwrap());
    }

    #[test]
    fn multiple_sources() {
        let mem1 = MemoryProvider::new();
        mem1.insert("a.txt", b"from_1".to_vec());

        let mem2 = MemoryProvider::new();
        mem2.insert("b.txt", b"from_2".to_vec());

        let mut vfs = Vfs::new();
        vfs.mount("src1", mem1);
        vfs.mount("src2", mem2);

        assert_eq!(poll_ready(vfs.read("src1/a.txt")).unwrap(), b"from_1");
        assert_eq!(poll_ready(vfs.read("src2/b.txt")).unwrap(), b"from_2");
    }

    #[test]
    fn clone_is_cheap() {
        let mut vfs = Vfs::new();
        vfs.mount("m", MemoryProvider::new());

        let vfs2 = vfs.clone();
        // Both share the same inner data
        poll_ready(vfs2.exists("m/anything")).unwrap();
    }
}
