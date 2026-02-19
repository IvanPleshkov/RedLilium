use std::path::PathBuf;

use crate::provider::{VfsFuture, VfsProvider};

/// File system VFS provider for reading and writing assets on disk.
///
/// The root path is joined with the VFS path to form the actual filesystem
/// path. All I/O is blocking (`std::fs`) inside the returned futures. The
/// caller must run these futures on a thread pool (via `IoRunner::run()`).
///
/// Path traversal is prevented by the VFS path normalization which rejects
/// `..` segments before they reach the provider.
///
/// # Example
///
/// ```ignore
/// let mut vfs = Vfs::new();
/// vfs.mount("assets", FileSystemProvider::new("./assets"));
///
/// // Reads ./assets/textures/brick.png
/// let bytes = io.run(vfs.read("assets/textures/brick.png")).await;
/// ```
pub struct FileSystemProvider {
    root: PathBuf,
}

impl FileSystemProvider {
    /// Create a provider rooted at the given directory.
    ///
    /// The directory does not need to exist yet — it will be checked
    /// at read/write time.
    pub fn new(root: impl Into<PathBuf>) -> Self {
        Self { root: root.into() }
    }

    /// Resolve a VFS path to a full filesystem path.
    fn resolve(&self, path: &str) -> PathBuf {
        self.root.join(path)
    }
}

impl VfsProvider for FileSystemProvider {
    fn read(&self, path: &str) -> VfsFuture<Vec<u8>> {
        let full_path = self.resolve(path);
        Box::pin(async move { Ok(std::fs::read(full_path)?) })
    }

    fn exists(&self, path: &str) -> VfsFuture<bool> {
        let full_path = self.resolve(path);
        Box::pin(async move { Ok(full_path.exists()) })
    }

    fn list_dir(&self, path: &str) -> VfsFuture<Vec<String>> {
        let full_path = self.resolve(path);
        Box::pin(async move {
            if !full_path.is_dir() {
                return Ok(Vec::new());
            }
            let mut entries = Vec::new();
            for entry in std::fs::read_dir(full_path)? {
                let entry = entry?;
                if let Some(name) = entry.file_name().to_str() {
                    entries.push(name.to_owned());
                }
            }
            entries.sort();
            Ok(entries)
        })
    }

    fn is_read_only(&self) -> bool {
        false
    }

    fn write(&self, path: &str, data: Vec<u8>) -> VfsFuture<()> {
        let full_path = self.resolve(path);
        Box::pin(async move {
            if let Some(parent) = full_path.parent() {
                std::fs::create_dir_all(parent)?;
            }
            std::fs::write(full_path, data)?;
            Ok(())
        })
    }

    fn delete(&self, path: &str) -> VfsFuture<()> {
        let full_path = self.resolve(path);
        Box::pin(async move {
            std::fs::remove_file(full_path)?;
            Ok(())
        })
    }

    fn create_dir(&self, path: &str) -> VfsFuture<()> {
        let full_path = self.resolve(path);
        Box::pin(async move {
            std::fs::create_dir_all(full_path)?;
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::VfsError;
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

    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("redlilium_vfs_test_{name}"));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn read_existing_file() {
        let dir = temp_dir("read");
        std::fs::write(dir.join("test.txt"), b"hello").unwrap();

        let provider = FileSystemProvider::new(&dir);
        let data = poll_ready(provider.read("test.txt")).unwrap();
        assert_eq!(data, b"hello");

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn read_missing_file() {
        let dir = temp_dir("read_missing");
        let provider = FileSystemProvider::new(&dir);
        let result = poll_ready(provider.read("nope.txt"));
        assert!(result.is_err());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn exists_check() {
        let dir = temp_dir("exists");
        std::fs::write(dir.join("file.txt"), b"").unwrap();

        let provider = FileSystemProvider::new(&dir);
        assert!(poll_ready(provider.exists("file.txt")).unwrap());
        assert!(!poll_ready(provider.exists("nope.txt")).unwrap());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn list_dir_entries() {
        let dir = temp_dir("list");
        std::fs::write(dir.join("a.txt"), b"").unwrap();
        std::fs::write(dir.join("b.txt"), b"").unwrap();
        std::fs::create_dir_all(dir.join("sub")).unwrap();

        let provider = FileSystemProvider::new(&dir);
        let entries = poll_ready(provider.list_dir("")).unwrap();
        assert!(entries.contains(&"a.txt".to_owned()));
        assert!(entries.contains(&"b.txt".to_owned()));
        assert!(entries.contains(&"sub".to_owned()));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn write_creates_file_and_parents() {
        let dir = temp_dir("write");
        let provider = FileSystemProvider::new(&dir);

        poll_ready(provider.write("sub/dir/file.txt", b"data".to_vec())).unwrap();
        assert_eq!(
            std::fs::read(dir.join("sub/dir/file.txt")).unwrap(),
            b"data"
        );

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn delete_file() {
        let dir = temp_dir("delete");
        std::fs::write(dir.join("file.txt"), b"data").unwrap();

        let provider = FileSystemProvider::new(&dir);
        poll_ready(provider.delete("file.txt")).unwrap();
        assert!(!dir.join("file.txt").exists());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn create_dir_nested() {
        let dir = temp_dir("mkdir");
        let provider = FileSystemProvider::new(&dir);

        poll_ready(provider.create_dir("a/b/c")).unwrap();
        assert!(dir.join("a/b/c").is_dir());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn is_not_read_only() {
        let provider = FileSystemProvider::new("/tmp");
        assert!(!provider.is_read_only());
    }
}
