use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::mpsc;

use notify::{EventKind, RecommendedWatcher, RecursiveMode, Watcher};

use crate::project::ProjectConfig;

/// Watches local filesystem mounts for changes and reports affected VFS paths.
pub struct FsWatcher {
    /// The underlying file watcher (kept alive).
    _watcher: RecommendedWatcher,
    /// Receives raw notify events from the background thread.
    event_rx: mpsc::Receiver<notify::Event>,
    /// Maps local absolute directory prefixes to VFS mount names.
    mount_roots: Vec<(PathBuf, String)>,
}

impl FsWatcher {
    /// Create a new watcher for all filesystem mounts in the project config.
    /// Returns `None` if there are no filesystem mounts to watch.
    pub fn new(config: &ProjectConfig) -> Option<Self> {
        let (tx, rx) = mpsc::channel::<notify::Event>();

        let mut watcher = notify::recommended_watcher(move |res: notify::Result<notify::Event>| {
            if let Ok(event) = res {
                let _ = tx.send(event);
            }
        })
        .ok()?;

        let mut mount_roots = Vec::new();

        for mount in &config.mount {
            if mount.r#type != "filesystem" {
                continue;
            }
            let local_path = Path::new(&mount.path);
            let abs_path = if local_path.is_absolute() {
                local_path.to_path_buf()
            } else {
                std::env::current_dir()
                    .ok()?
                    .join(local_path)
                    .canonicalize()
                    .ok()?
            };

            if abs_path.is_dir() {
                if let Err(e) = watcher.watch(&abs_path, RecursiveMode::Recursive) {
                    log::warn!("Failed to watch {:?}: {e}", abs_path);
                    continue;
                }
                log::info!(
                    "Watching filesystem mount: \"{}\" -> {:?}",
                    mount.name,
                    abs_path
                );
                mount_roots.push((abs_path, mount.name.clone()));
            }
        }

        if mount_roots.is_empty() {
            return None;
        }

        Some(Self {
            _watcher: watcher,
            event_rx: rx,
            mount_roots,
        })
    }

    /// Drain filesystem change events and return affected VFS directory paths.
    pub fn poll_changes(&self) -> Vec<String> {
        let mut changed: HashMap<String, ()> = HashMap::new();

        while let Ok(event) = self.event_rx.try_recv() {
            if !matches!(
                event.kind,
                EventKind::Create(_) | EventKind::Modify(_) | EventKind::Remove(_)
            ) {
                continue;
            }
            for path in &event.paths {
                if let Some(vfs_dir) = self.to_vfs_dir(path) {
                    changed.insert(vfs_dir, ());
                }
            }
        }

        changed.into_keys().collect()
    }

    /// Map a local filesystem path to its VFS directory path (e.g. "assets/textures").
    fn to_vfs_dir(&self, local_path: &Path) -> Option<String> {
        for (root, mount_name) in &self.mount_roots {
            if let Ok(relative) = local_path.strip_prefix(root) {
                // Get the parent directory (we want the dir containing the changed file)
                let dir = if local_path.is_dir() {
                    relative
                } else {
                    relative.parent().unwrap_or(Path::new(""))
                };

                let dir_str = dir.to_string_lossy().replace('\\', "/");
                return if dir_str.is_empty() {
                    Some(mount_name.clone())
                } else {
                    Some(format!("{mount_name}/{dir_str}"))
                };
            }
        }
        None
    }
}
