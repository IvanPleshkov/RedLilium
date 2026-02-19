use std::collections::{HashMap, HashSet};

use redlilium_vfs::Vfs;

use crate::background_vfs::{BackgroundVfs, VfsRequestId, VfsResult};
use crate::project::ProjectConfig;

/// A directory entry in the asset browser.
struct DirEntry {
    name: String,
    is_dir: bool,
}

/// Asset browser panel showing VFS contents as a directory tree + file list.
pub struct AssetBrowser {
    /// Mount names from the project config (used as tree roots).
    mount_names: Vec<String>,
    /// Currently selected path: (source_name, directory_path_within_source).
    selected: Option<(String, String)>,
    /// Tree nodes that are currently expanded (keys: "source/dir/subdir").
    expanded: HashSet<String>,
    /// Cached file listing for the right panel.
    cached_entries: Vec<DirEntry>,
    /// The (source, dir) that `cached_entries` corresponds to.
    cached_key: Option<(String, String)>,

    // Async VFS support
    bg_vfs: BackgroundVfs,
    /// Cached directory listings by VFS path.
    dir_cache: HashMap<String, Vec<String>>,
    /// In-flight listing requests: vfs_path -> request_id.
    pending_requests: HashMap<String, VfsRequestId>,
}

impl AssetBrowser {
    /// Create a new asset browser from the project config.
    pub fn new(config: &ProjectConfig) -> Self {
        Self {
            mount_names: config.mount.iter().map(|m| m.name.clone()).collect(),
            selected: None,
            expanded: HashSet::new(),
            cached_entries: Vec::new(),
            cached_key: None,
            bg_vfs: BackgroundVfs::new(),
            dir_cache: HashMap::new(),
            pending_requests: HashMap::new(),
        }
    }

    /// Poll completed background VFS results. Call once per frame.
    pub fn poll(&mut self) {
        for (id, result) in self.bg_vfs.poll_results() {
            match result {
                VfsResult::ListDir(Ok(entries)) => {
                    if let Some((path, _)) =
                        self.pending_requests.iter().find(|(_, rid)| **rid == id)
                    {
                        let path = path.clone();
                        self.dir_cache.insert(path.clone(), entries);
                        self.pending_requests.remove(&path);
                    }
                }
                VfsResult::ListDir(Err(e)) => {
                    log::warn!("VFS list_dir failed: {e}");
                    self.pending_requests.retain(|_, rid| *rid != id);
                }
            }
        }
    }

    /// Request a directory listing. Returns cached result if available,
    /// otherwise dispatches a background request and returns `None`.
    fn request_list_dir(&mut self, vfs: &Vfs, vfs_path: &str) -> Option<Vec<String>> {
        if let Some(entries) = self.dir_cache.get(vfs_path) {
            return Some(entries.clone());
        }
        if !self.pending_requests.contains_key(vfs_path) {
            let id = self.bg_vfs.list_dir(vfs, vfs_path);
            self.pending_requests.insert(vfs_path.to_owned(), id);
        }
        None
    }

    /// Draw the asset browser UI.
    pub fn show(&mut self, ui: &mut egui::Ui, vfs: &Vfs) {
        ui.horizontal(|ui| {
            // Left panel: directory tree (fixed width)
            let tree_width = ui.available_width() * 0.3;
            ui.allocate_ui_with_layout(
                egui::vec2(tree_width, ui.available_height()),
                egui::Layout::top_down(egui::Align::LEFT),
                |ui| {
                    egui::ScrollArea::both()
                        .id_salt("asset_tree")
                        .auto_shrink([false, false])
                        .show(ui, |ui| {
                            self.draw_tree(ui, vfs);
                        });
                },
            );

            ui.separator();

            // Right panel: file listing
            ui.allocate_ui_with_layout(
                egui::vec2(ui.available_width(), ui.available_height()),
                egui::Layout::top_down(egui::Align::LEFT),
                |ui| {
                    egui::ScrollArea::both()
                        .id_salt("asset_files")
                        .auto_shrink([false, false])
                        .show(ui, |ui| {
                            self.draw_file_list(ui, vfs);
                        });
                },
            );
        });
    }

    /// Draw the directory tree (left panel).
    fn draw_tree(&mut self, ui: &mut egui::Ui, vfs: &Vfs) {
        let mount_names = self.mount_names.clone();
        for source in &mount_names {
            self.draw_tree_node(ui, vfs, source, "");
        }
    }

    /// Draw a single tree node (source root or subdirectory).
    fn draw_tree_node(&mut self, ui: &mut egui::Ui, vfs: &Vfs, source: &str, dir_path: &str) {
        let tree_key = if dir_path.is_empty() {
            source.to_owned()
        } else {
            format!("{source}/{dir_path}")
        };

        let display_name = if dir_path.is_empty() {
            source
        } else {
            dir_path.rsplit('/').next().unwrap_or(dir_path)
        };

        let is_expanded = self.expanded.contains(&tree_key);
        let is_selected = self.selected.as_ref() == Some(&(source.to_owned(), dir_path.to_owned()));

        let header = egui::CollapsingHeader::new(display_name)
            .id_salt(&tree_key)
            .open(if is_expanded { Some(true) } else { None })
            .show_background(is_selected)
            .show(ui, |ui| {
                let children = self.list_subdirs(vfs, source, dir_path);
                match children {
                    Some(names) => {
                        for child_name in names {
                            let child_path = if dir_path.is_empty() {
                                child_name.clone()
                            } else {
                                format!("{dir_path}/{child_name}")
                            };
                            self.draw_tree_node(ui, vfs, source, &child_path);
                        }
                    }
                    None => {
                        ui.weak("Loading...");
                    }
                }
            });

        // Track expand/collapse state
        if header.fully_open() {
            self.expanded.insert(tree_key.clone());
        } else if header.openness == 0.0 {
            self.expanded.remove(&tree_key);
        }

        // Select on click
        if header.header_response.clicked() {
            let new_sel = (source.to_owned(), dir_path.to_owned());
            if self.cached_key.as_ref() != Some(&new_sel) {
                self.cached_key = None;
            }
            self.selected = Some(new_sel);
        }
    }

    /// List only subdirectories under a given path.
    /// Returns `None` while loading.
    fn list_subdirs(&mut self, vfs: &Vfs, source: &str, dir_path: &str) -> Option<Vec<String>> {
        let vfs_path = if dir_path.is_empty() {
            source.to_owned()
        } else {
            format!("{source}/{dir_path}")
        };

        let entries = self.request_list_dir(vfs, &vfs_path)?;
        Some(entries.into_iter().filter(|e| !e.contains('.')).collect())
    }

    /// Draw the file listing (right panel).
    fn draw_file_list(&mut self, ui: &mut egui::Ui, vfs: &Vfs) {
        let Some((source, dir_path)) = &self.selected else {
            ui.weak("Select a directory");
            return;
        };
        let source = source.clone();
        let dir_path = dir_path.clone();

        // Refresh cache if needed
        if self.cached_key.as_ref() != Some(&(source.clone(), dir_path.clone())) {
            let vfs_path = if dir_path.is_empty() {
                source.clone()
            } else {
                format!("{source}/{dir_path}")
            };

            match self.request_list_dir(vfs, &vfs_path) {
                Some(names) => {
                    self.cached_entries = names
                        .into_iter()
                        .map(|name| {
                            let is_dir = !name.contains('.');
                            DirEntry { name, is_dir }
                        })
                        .collect();

                    self.cached_entries
                        .sort_by(|a, b| b.is_dir.cmp(&a.is_dir).then(a.name.cmp(&b.name)));

                    self.cached_key = Some((source.clone(), dir_path.clone()));
                }
                None => {
                    // Show loading if no previous cache exists
                    if self.cached_entries.is_empty() {
                        ui.weak("Loading...");
                        return;
                    }
                    // Otherwise keep showing stale data until refresh completes
                }
            }
        }

        // Breadcrumb path
        let display_path = if dir_path.is_empty() {
            source.clone()
        } else {
            format!("{source}/{dir_path}")
        };
        ui.strong(&display_path);
        ui.separator();

        if self.cached_entries.is_empty() {
            ui.weak("(empty)");
            return;
        }

        // File listing
        for entry in &self.cached_entries {
            let icon = if entry.is_dir {
                "\u{1F4C1}"
            } else {
                "\u{1F4C4}"
            };
            let label = format!("{icon} {}", entry.name);

            let response = ui.selectable_label(false, &label);

            if response.double_clicked() && entry.is_dir {
                let new_dir = if dir_path.is_empty() {
                    entry.name.clone()
                } else {
                    format!("{dir_path}/{}", entry.name)
                };
                let tree_key = format!("{source}/{new_dir}");
                self.expanded.insert(tree_key);
                self.selected = Some((source.clone(), new_dir));
                self.cached_key = None;
                break;
            }
        }
    }
}
