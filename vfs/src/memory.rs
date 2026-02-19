use std::collections::{HashMap, HashSet};
use std::sync::{Arc, RwLock};

use crate::error::VfsError;
use crate::provider::{VfsFuture, VfsProvider};

/// In-memory VFS provider for tests and embedded assets.
///
/// Thread-safe and mutable even after being mounted in a [`Vfs`](crate::Vfs).
/// Supports both read and write operations.
///
/// Directories are implicit — they exist whenever a file path contains
/// that directory prefix.
///
/// # Example
///
/// ```ignore
/// let mem = MemoryProvider::new();
/// mem.insert("config/settings.json", b"{}".to_vec());
/// mem.insert("shaders/basic.wgsl", shader_bytes);
///
/// let mut vfs = Vfs::new();
/// vfs.mount("builtin", mem);
/// ```
#[derive(Clone)]
pub struct MemoryProvider {
    files: Arc<RwLock<HashMap<String, Vec<u8>>>>,
}

impl MemoryProvider {
    /// Create an empty in-memory provider.
    pub fn new() -> Self {
        Self {
            files: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Insert a file at the given path.
    ///
    /// The path should use forward slashes and have no leading slash.
    /// Overwrites any existing file at the same path.
    pub fn insert(&self, path: impl Into<String>, data: Vec<u8>) {
        self.files.write().unwrap().insert(path.into(), data);
    }

    /// Remove a file at the given path, returning its data if it existed.
    pub fn remove(&self, path: &str) -> Option<Vec<u8>> {
        self.files.write().unwrap().remove(path)
    }
}

impl Default for MemoryProvider {
    fn default() -> Self {
        Self::new()
    }
}

impl VfsProvider for MemoryProvider {
    fn read(&self, path: &str) -> VfsFuture<Vec<u8>> {
        let files = self.files.clone();
        let path = path.to_owned();
        Box::pin(async move {
            let map = files.read().unwrap();
            map.get(&path).cloned().ok_or(VfsError::NotFound(path))
        })
    }

    fn exists(&self, path: &str) -> VfsFuture<bool> {
        let files = self.files.clone();
        let path = path.to_owned();
        Box::pin(async move {
            let map = files.read().unwrap();
            Ok(map.contains_key(&path))
        })
    }

    fn list_dir(&self, path: &str) -> VfsFuture<Vec<String>> {
        let files = self.files.clone();
        let path = path.to_owned();
        Box::pin(async move {
            let map = files.read().unwrap();
            let mut children = HashSet::new();

            let prefix = if path.is_empty() {
                String::new()
            } else {
                format!("{path}/")
            };

            for key in map.keys() {
                if let Some(rest) = key.strip_prefix(&prefix) {
                    // Extract the immediate child name (first segment)
                    let child = match rest.find('/') {
                        Some(pos) => &rest[..pos],
                        None => rest,
                    };
                    if !child.is_empty() {
                        children.insert(child.to_owned());
                    }
                }
            }

            let mut result: Vec<String> = children.into_iter().collect();
            result.sort();
            Ok(result)
        })
    }

    fn is_read_only(&self) -> bool {
        false
    }

    fn write(&self, path: &str, data: Vec<u8>) -> VfsFuture<()> {
        let files = self.files.clone();
        let path = path.to_owned();
        Box::pin(async move {
            files.write().unwrap().insert(path, data);
            Ok(())
        })
    }

    fn delete(&self, path: &str) -> VfsFuture<()> {
        let files = self.files.clone();
        let path = path.to_owned();
        Box::pin(async move {
            files
                .write()
                .unwrap()
                .remove(&path)
                .ok_or(VfsError::NotFound(path))?;
            Ok(())
        })
    }

    fn create_dir(&self, _path: &str) -> VfsFuture<()> {
        // Directories are implicit in MemoryProvider
        Box::pin(async { Ok(()) })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::pin::Pin;
    use std::task::{Context, Poll, RawWaker, RawWakerVTable, Waker};

    /// Poll an immediately-ready future. Panics if the future is Pending.
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
    fn read_existing_file() {
        let mem = MemoryProvider::new();
        mem.insert("config.json", b"{}".to_vec());
        let result = poll_ready(mem.read("config.json")).unwrap();
        assert_eq!(result, b"{}");
    }

    #[test]
    fn read_missing_file() {
        let mem = MemoryProvider::new();
        let result = poll_ready(mem.read("nope.txt"));
        assert!(result.is_err());
    }

    #[test]
    fn exists_true() {
        let mem = MemoryProvider::new();
        mem.insert("file.txt", vec![]);
        assert!(poll_ready(mem.exists("file.txt")).unwrap());
    }

    #[test]
    fn exists_false() {
        let mem = MemoryProvider::new();
        assert!(!poll_ready(mem.exists("nope.txt")).unwrap());
    }

    #[test]
    fn list_dir_root() {
        let mem = MemoryProvider::new();
        mem.insert("a.txt", vec![]);
        mem.insert("b/c.txt", vec![]);
        mem.insert("b/d.txt", vec![]);

        let mut entries = poll_ready(mem.list_dir("")).unwrap();
        entries.sort();
        assert_eq!(entries, vec!["a.txt", "b"]);
    }

    #[test]
    fn list_dir_nested() {
        let mem = MemoryProvider::new();
        mem.insert("dir/a.txt", vec![]);
        mem.insert("dir/sub/b.txt", vec![]);

        let mut entries = poll_ready(mem.list_dir("dir")).unwrap();
        entries.sort();
        assert_eq!(entries, vec!["a.txt", "sub"]);
    }

    #[test]
    fn list_dir_empty() {
        let mem = MemoryProvider::new();
        let entries = poll_ready(mem.list_dir("nonexistent")).unwrap();
        assert!(entries.is_empty());
    }

    #[test]
    fn write_and_read() {
        let mem = MemoryProvider::new();
        poll_ready(mem.write("new.txt", b"hello".to_vec())).unwrap();
        let data = poll_ready(mem.read("new.txt")).unwrap();
        assert_eq!(data, b"hello");
    }

    #[test]
    fn delete_existing() {
        let mem = MemoryProvider::new();
        mem.insert("file.txt", b"data".to_vec());
        poll_ready(mem.delete("file.txt")).unwrap();
        assert!(!poll_ready(mem.exists("file.txt")).unwrap());
    }

    #[test]
    fn delete_missing() {
        let mem = MemoryProvider::new();
        let result = poll_ready(mem.delete("nope.txt"));
        assert!(result.is_err());
    }

    #[test]
    fn is_not_read_only() {
        let mem = MemoryProvider::new();
        assert!(!mem.is_read_only());
    }

    #[test]
    fn remove_returns_data() {
        let mem = MemoryProvider::new();
        mem.insert("file.txt", b"data".to_vec());
        let data = mem.remove("file.txt");
        assert_eq!(data, Some(b"data".to_vec()));
        assert!(mem.remove("file.txt").is_none());
    }
}
