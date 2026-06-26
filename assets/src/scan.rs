//! Mount import/scan. Walks a mount (or a subtree), routes files by extension to
//! a kind, hashes their content, and registers them in the DB.
//!
//! **Idempotent + watch-ready:** re-scanning is a no-op for unchanged files and
//! returns a [`ScanReport`] delta (added / modified / removed guids) so a reload
//! layer (and, later, a file-watcher feeding single-path scans) can react. The
//! scan is the *mechanism*; the trigger (startup / manual refresh / watch event)
//! is someone else's concern.
//!
//! Prototype note: every routed file is read in full to hash its content. A real
//! importer would gate on cheap metadata (mtime + size) and hash only suspects.

use std::collections::HashSet;

use redlilium_vfs::Vfs;

use crate::db::{AssetDb, AssetPath};
use crate::error::AssetError;
use crate::source::Guid;

/// What a scan changed in the DB — the delta a reload layer consumes.
#[derive(Debug, Default)]
pub struct ScanReport {
    /// Newly registered (path not previously known).
    pub added: Vec<Guid>,
    /// Content hash changed since last scan.
    pub modified: Vec<Guid>,
    /// Previously registered under the scanned scope, now gone from disk.
    pub removed: Vec<Guid>,
    /// Routed files whose content was unchanged.
    pub unchanged: usize,
    /// Files seen but with no extension route (skipped).
    pub unrouted: usize,
}

/// Scan `mount` (or the subtree `scope` within it) through `vfs`, routing files
/// to kinds via `route` (`extension -> kind`) and registering them in `db`.
/// Returns the delta. `scope = None` scans the whole mount.
pub async fn scan_mount(
    vfs: &Vfs,
    db: &mut AssetDb,
    mount: &str,
    scope: Option<&str>,
    route: impl Fn(&str) -> Option<String>,
) -> Result<ScanReport, AssetError> {
    let mut report = ScanReport::default();
    let mut seen: HashSet<String> = HashSet::new();

    let scope = scope.map(|s| s.trim_matches('/'));
    let start = match scope {
        Some(s) => format!("{mount}/{s}"),
        None => mount.to_string(),
    };
    let strip = format!("{mount}/");

    // Iterative DFS over directories (async recursion would need boxing).
    let mut stack = vec![start];
    while let Some(dir) = stack.pop() {
        let entries = vfs
            .list_dir_entries(&dir)
            .await
            .map_err(|e| AssetError::Io(e.to_string()))?;
        for entry in entries {
            let raw = format!("{dir}/{}", entry.name);
            if entry.is_dir {
                stack.push(raw);
                continue;
            }
            let Some(kind) = extension_of(&entry.name).and_then(&route) else {
                report.unrouted += 1;
                continue;
            };
            let rel = raw.strip_prefix(&strip).unwrap_or(&raw).to_string();
            let bytes = vfs
                .read(&raw)
                .await
                .map_err(|e| AssetError::Io(e.to_string()))?;
            let hash = fnv1a(&bytes);

            let path = AssetPath::new(mount, rel.clone());
            let prev = db
                .guid_of(&path)
                .and_then(|g| db.record(&g).map(|r| r.source_hash));
            let guid = db.register_path(path, &kind, hash);
            match prev {
                None => report.added.push(guid),
                Some(h) if h != hash => report.modified.push(guid),
                Some(_) => report.unchanged += 1,
            }
            seen.insert(rel);
        }
    }

    // Deletions: records under the scanned scope no longer on disk.
    let prefix = scope.map(|s| format!("{s}/")).unwrap_or_default();
    for (guid, rel) in db.entries_under(mount, &prefix) {
        if !seen.contains(&rel) {
            db.remove(&guid);
            report.removed.push(guid);
        }
    }
    Ok(report)
}

/// The extension after the last dot (no dot), or `None` if the name has none.
fn extension_of(name: &str) -> Option<&str> {
    name.rsplit_once('.').map(|(_, ext)| ext)
}

/// FNV-1a 64-bit — small, deterministic, stable across runs (so persisted
/// `source_hash`es compare correctly). Change-detection only, not security.
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;
    use redlilium_vfs::MemoryProvider;
    use std::future::Future;
    use std::task::{Context, Poll, Waker};

    /// Drive a future whose awaits are all immediately ready (MemoryProvider).
    fn block<T>(fut: impl Future<Output = T>) -> T {
        let mut fut = std::pin::pin!(fut);
        let mut cx = Context::from_waker(Waker::noop());
        loop {
            if let Poll::Ready(v) = fut.as_mut().poll(&mut cx) {
                return v;
            }
        }
    }

    fn vfs_with(files: &[(&str, &[u8])]) -> Vfs {
        let mem = MemoryProvider::new();
        for (p, b) in files {
            mem.insert(*p, b.to_vec());
        }
        let mut vfs = Vfs::new();
        vfs.mount("assets", mem);
        vfs
    }

    fn route(ext: &str) -> Option<String> {
        matches!(ext, "glb" | "gltf").then(|| "mesh".to_string())
    }

    #[test]
    fn scan_reports_added_modified_removed() {
        let mut db = AssetDb::new();

        // Initial import: two routed meshes + one unrouted file.
        let v1 = vfs_with(&[
            ("mesh/a.glb", b"AAA"),
            ("mesh/b.gltf", b"BBB"),
            ("readme.txt", b"x"),
        ]);
        let r = block(scan_mount(&v1, &mut db, "assets", None, route)).unwrap();
        assert_eq!(r.added.len(), 2);
        assert_eq!(r.unrouted, 1);
        assert_eq!(db.len(), 2);

        // Re-scan, nothing changed → all unchanged, idempotent.
        let r = block(scan_mount(&v1, &mut db, "assets", None, route)).unwrap();
        assert!(r.added.is_empty());
        assert_eq!(r.unchanged, 2);

        // a.glb content changed (same guid, bumped hash).
        let v2 = vfs_with(&[("mesh/a.glb", b"ZZZ"), ("mesh/b.gltf", b"BBB")]);
        let r = block(scan_mount(&v2, &mut db, "assets", None, route)).unwrap();
        assert_eq!(r.modified.len(), 1);
        assert_eq!(r.unchanged, 1);

        // a.glb deleted from disk.
        let v3 = vfs_with(&[("mesh/b.gltf", b"BBB")]);
        let r = block(scan_mount(&v3, &mut db, "assets", None, route)).unwrap();
        assert_eq!(r.removed.len(), 1);
        assert_eq!(db.len(), 1);
    }

    #[test]
    fn scope_limits_walk_and_deletions() {
        let mut db = AssetDb::new();
        let v = vfs_with(&[("mesh/a.glb", b"A"), ("other/c.glb", b"C")]);
        block(scan_mount(&v, &mut db, "assets", None, route)).unwrap();
        assert_eq!(db.len(), 2);

        // Re-scan only `mesh/`, with mesh/a.glb gone: only the in-scope record is
        // removed; other/c.glb (out of scope) is untouched.
        let v2 = vfs_with(&[("other/c.glb", b"C")]);
        let r = block(scan_mount(&v2, &mut db, "assets", Some("mesh"), route)).unwrap();
        assert_eq!(r.removed.len(), 1);
        assert_eq!(db.len(), 1); // other/c.glb survived
    }
}
