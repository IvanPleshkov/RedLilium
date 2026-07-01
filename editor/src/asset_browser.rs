use std::collections::{HashMap, HashSet};

use redlilium_ecs::Entity;
use redlilium_ecs::ui::{ComponentDragPayload, ComponentFileDragPayload, PrefabFileDragPayload};
use redlilium_vfs::{Vfs, VfsDirEntry};

use crate::background_vfs::{BackgroundVfs, VfsRequestId, VfsResult};
use crate::fs_watcher::FsWatcher;
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
    /// Currently selected file as an asset: (source_name, file_path_within_source).
    /// Drives the asset inspector.
    selected_file: Option<(String, String)>,
    /// A mount whose in-memory DB was edited and needs persisting (set by the
    /// asset inspector, drained by the editor).
    db_dirty: Option<String>,
    /// Tree nodes that are currently expanded (keys: "source/dir/subdir").
    expanded: HashSet<String>,
    /// Cached file listing for the right panel.
    cached_entries: Vec<DirEntry>,
    /// The (source, dir) that `cached_entries` corresponds to.
    cached_key: Option<(String, String)>,

    // Async VFS support
    bg_vfs: BackgroundVfs,
    /// Cached directory listings (with entry kinds) by VFS path.
    dir_cache: HashMap<String, Vec<VfsDirEntry>>,
    /// In-flight listing requests: vfs_path -> request_id.
    pending_requests: HashMap<String, VfsRequestId>,
    /// In-flight write requests: vfs_path -> request_id.
    pending_writes: HashMap<String, VfsRequestId>,
    /// In-flight read requests: request_id -> vfs_path.
    pending_reads: HashMap<VfsRequestId, String>,
    /// Completed reads waiting to be consumed by the editor.
    pub completed_reads: Vec<(String, Vec<u8>)>,
    /// Watches local filesystem mounts for external changes.
    fs_watcher: Option<FsWatcher>,
    /// Pending component export: (entity, comp_name, target_vfs_dir).
    /// Set when a component is dropped from inspector onto the file list.
    pub pending_component_export: Option<(Entity, &'static str, String)>,
    /// Pending prefab export: (root_entity, target_vfs_dir).
    /// Set when an entity is dropped from world inspector onto the file list.
    pub pending_prefab_export: Option<(Entity, String)>,
    /// Pending asset creation from the "New" context menu: (source, dir, kind).
    /// Drained by the editor, which writes the file + DB record.
    pending_new: Option<(String, String, String)>,
    /// Active inline rename: (source, file_path, edit_buffer).
    renaming: Option<(String, String, String)>,
    /// Committed rename to apply: (source, old_path, new_name). Drained by editor.
    pending_rename: Option<(String, String, String)>,
    /// Committed move to apply: (source, old_path, new_dir). Drained by editor.
    pending_move: Option<(String, String, String)>,
    /// Committed delete to apply: (source, path). Drained by editor.
    pending_delete: Option<(String, String)>,
}

/// Drag payload for moving an asset file between directories.
#[derive(Clone)]
struct AssetFileDrag {
    source: String,
    path: String,
}

/// Split a full VFS path `source/dir/name` into `(source, path-within-source)`.
fn split_source_path(vfs_path: &str) -> Option<(String, String)> {
    vfs_path
        .split_once('/')
        .map(|(s, rel)| (s.to_owned(), rel.to_owned()))
}

impl AssetBrowser {
    /// Create a new asset browser from the project config.
    pub fn new(config: &ProjectConfig) -> Self {
        Self {
            mount_names: config.mount.iter().map(|m| m.name.clone()).collect(),
            selected: None,
            selected_file: None,
            db_dirty: None,
            expanded: HashSet::new(),
            cached_entries: Vec::new(),
            cached_key: None,
            bg_vfs: BackgroundVfs::new(),
            dir_cache: HashMap::new(),
            pending_requests: HashMap::new(),
            pending_writes: HashMap::new(),
            pending_reads: HashMap::new(),
            completed_reads: Vec::new(),
            fs_watcher: FsWatcher::new(config),
            pending_component_export: None,
            pending_prefab_export: None,
            pending_new: None,
            renaming: None,
            pending_rename: None,
            pending_move: None,
            pending_delete: None,
        }
    }

    /// Take the pending delete intent: `(source, path)`.
    pub fn take_pending_delete(&mut self) -> Option<(String, String)> {
        self.pending_delete.take()
    }

    /// After a delete, refresh the directory and clear the selection if it pointed
    /// at the deleted file.
    pub fn notify_asset_deleted(&mut self, source: &str, path: &str) {
        self.cached_key = None;
        if self.selected_file.as_ref() == Some(&(source.to_owned(), path.to_owned())) {
            self.selected_file = None;
        }
    }

    /// Dispatch an async VFS delete.
    pub fn dispatch_delete(&mut self, vfs: &Vfs, vfs_path: &str) {
        self.bg_vfs.delete(vfs, vfs_path);
    }

    /// Take the pending rename intent: `(source, old_path, new_name)`.
    pub fn take_pending_rename(&mut self) -> Option<(String, String, String)> {
        self.pending_rename.take()
    }

    /// Take the pending move intent: `(source, old_path, new_dir)`.
    pub fn take_pending_move(&mut self) -> Option<(String, String, String)> {
        self.pending_move.take()
    }

    /// After a rename/move, refresh the affected directories and update the
    /// selection to the new path.
    pub fn notify_asset_moved(&mut self, source: &str, new_path: &str) {
        self.cached_key = None;
        self.selected_file = Some((source.to_owned(), new_path.to_owned()));
    }

    /// Take the pending "New asset" intent: `(source, dir, kind)`, if any.
    pub fn take_pending_new(&mut self) -> Option<(String, String, String)> {
        self.pending_new.take()
    }

    /// After the editor creates an asset, refresh the directory listing and select
    /// the new file (drives the inspector).
    pub fn notify_asset_created(&mut self, source: &str, dir: &str, file_path: &str) {
        let vfs_dir = if dir.is_empty() {
            source.to_owned()
        } else {
            format!("{source}/{dir}")
        };
        self.dir_cache.remove(&vfs_dir);
        self.cached_key = None;
        self.selected_file = Some((source.to_owned(), file_path.to_owned()));
    }

    /// The file currently selected as an asset: `(source, path_within_source)`.
    pub fn selected_file(&self) -> Option<&(String, String)> {
        self.selected_file.as_ref()
    }

    /// Clear the asset selection (e.g. when an entity is selected instead).
    pub fn clear_selected_file(&mut self) {
        self.selected_file = None;
    }

    /// Mark a mount's in-memory DB as edited (needs persisting).
    pub fn mark_db_dirty(&mut self, mount: impl Into<String>) {
        self.db_dirty = Some(mount.into());
    }

    /// Take the mount whose DB needs persisting, if any (clears the flag).
    pub fn take_db_dirty(&mut self) -> Option<String> {
        self.db_dirty.take()
    }

    /// Register an extra mount as a browser tree root (e.g. the engine `std`
    /// mount, which is added to the VFS programmatically rather than via config).
    pub fn add_mount(&mut self, name: impl Into<String>) {
        let name = name.into();
        if !self.mount_names.contains(&name) {
            self.mount_names.push(name);
        }
    }

    /// Poll completed background VFS results and filesystem changes. Call once per frame.
    pub fn poll(&mut self) {
        // Check for external filesystem changes
        if let Some(watcher) = &self.fs_watcher {
            for vfs_dir in watcher.poll_changes() {
                log::debug!("Filesystem change detected: {vfs_dir}");
                self.dir_cache.remove(&vfs_dir);
                self.cached_key = None;
            }
        }

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
                VfsResult::Write(Ok(())) => {
                    if let Some((path, _)) = self.pending_writes.iter().find(|(_, rid)| **rid == id)
                    {
                        let path = path.clone();
                        log::info!("File imported: {path}");
                        // Invalidate parent directory cache to trigger refresh
                        if let Some((parent, _)) = path.rsplit_once('/') {
                            self.dir_cache.remove(parent);
                        }
                        self.pending_writes.remove(&path);
                        self.cached_key = None;
                    }
                }
                VfsResult::Write(Err(e)) => {
                    log::error!("VFS write failed: {e}");
                    self.pending_writes.retain(|_, rid| *rid != id);
                }
                VfsResult::Read(Ok(data)) => {
                    if let Some(path) = self.pending_reads.remove(&id) {
                        self.completed_reads.push((path, data));
                    }
                }
                VfsResult::Read(Err(e)) => {
                    log::error!("VFS read failed: {e}");
                    self.pending_reads.remove(&id);
                }
                VfsResult::Move(Ok(()), from, to) => {
                    log::info!("Moved: {from} -> {to}");
                    for p in [&from, &to] {
                        if let Some((parent, _)) = p.rsplit_once('/') {
                            self.dir_cache.remove(parent);
                        }
                    }
                    self.cached_key = None;
                }
                VfsResult::Move(Err(e), from, to) => {
                    log::error!("VFS move {from} -> {to} failed: {e}");
                }
                VfsResult::Delete(Ok(()), path) => {
                    log::info!("Deleted: {path}");
                    if let Some((parent, _)) = path.rsplit_once('/') {
                        self.dir_cache.remove(parent);
                    }
                    self.cached_key = None;
                }
                VfsResult::Delete(Err(e), path) => {
                    log::error!("VFS delete {path} failed: {e}");
                }
            }
        }
    }

    /// Request a directory listing. Returns cached result if available,
    /// otherwise dispatches a background request and returns `None`.
    fn request_list_dir(&mut self, vfs: &Vfs, vfs_path: &str) -> Option<Vec<VfsDirEntry>> {
        if let Some(entries) = self.dir_cache.get(vfs_path) {
            return Some(entries.clone());
        }
        if !self.pending_requests.contains_key(vfs_path) {
            let id = self.bg_vfs.list_dir(vfs, vfs_path);
            self.pending_requests.insert(vfs_path.to_owned(), id);
        }
        None
    }

    /// Dispatch an async VFS read. The result will appear in `completed_reads` on a future poll.
    pub fn dispatch_read(&mut self, vfs: &Vfs, vfs_path: &str) -> VfsRequestId {
        let id = self.bg_vfs.read(vfs, vfs_path);
        self.pending_reads.insert(id, vfs_path.to_owned());
        id
    }

    /// Dispatch an async VFS write (e.g. for component export).
    pub fn dispatch_write(&mut self, vfs: &Vfs, vfs_path: &str, data: Vec<u8>) {
        let id = self.bg_vfs.write(vfs, vfs_path, data);
        self.pending_writes.insert(vfs_path.to_owned(), id);
    }

    /// Dispatch an async VFS file move (rename or move-between-dirs).
    pub fn dispatch_move(&mut self, vfs: &Vfs, from: &str, to: &str) {
        self.bg_vfs.move_file(vfs, from, to);
    }

    /// Draw the asset browser UI.
    pub fn show(&mut self, ui: &mut egui::Ui, vfs: &Vfs) {
        // Handle files dropped from external apps (Finder, Explorer, etc.)
        self.handle_dropped_files(ui, vfs);

        // Show drop target overlay when files are being hovered
        let hovering = ui.ctx().input(|i| !i.raw.hovered_files.is_empty());

        // Left panel: resizable directory tree (fills full height)
        egui::SidePanel::left(ui.id().with("asset_tree_panel"))
            .resizable(true)
            .default_width(ui.available_width() * 0.3)
            .show_inside(ui, |ui| {
                egui::ScrollArea::both()
                    .id_salt("asset_tree")
                    .auto_shrink([false, false])
                    .show(ui, |ui| {
                        self.draw_tree(ui, vfs);
                    });
            });

        // Right panel: file listing (fills remaining space)
        egui::ScrollArea::both()
            .id_salt("asset_files")
            .auto_shrink([false, false])
            .show(ui, |ui| {
                self.draw_file_list(ui, vfs);
            });

        if hovering && self.selected.is_some() {
            let rect = ui.min_rect();
            ui.painter().rect_stroke(
                rect,
                4.0,
                egui::Stroke::new(2.0, crate::theme::ACCENT),
                egui::StrokeKind::Outside,
            );
        }
    }

    /// Import files dropped from external applications into the selected VFS directory.
    fn handle_dropped_files(&mut self, ui: &egui::Ui, vfs: &Vfs) {
        let Some((source, dir_path)) = &self.selected else {
            return;
        };
        let source = source.clone();
        let dir_path = dir_path.clone();

        let dropped: Vec<_> = ui.ctx().input(|i| i.raw.dropped_files.clone());
        for file in dropped {
            let Some(path) = &file.path else { continue };
            let Ok(data) = std::fs::read(path) else {
                log::error!("Failed to read dropped file: {}", path.display());
                continue;
            };
            let file_name = path.file_name().unwrap_or_default().to_string_lossy();
            let vfs_path = if dir_path.is_empty() {
                format!("{source}/{file_name}")
            } else {
                format!("{source}/{dir_path}/{file_name}")
            };
            log::info!("Importing: {} ({} bytes)", vfs_path, data.len());
            let id = self.bg_vfs.write(vfs, &vfs_path, data);
            self.pending_writes.insert(vfs_path, id);
        }
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

        // Accept a file dropped onto this directory → move it here (same-mount).
        // Any of the file drag payloads is accepted: AssetFileDrag (materials,
        // layouts, meshes, …) plus the .component / .prefab import payloads, which
        // carry a `vfs_path` we can move by. Hover-check each type first
        // (non-destructive), then release only the hovered one (release is
        // destructive regardless of the downcast).
        let hdr = &header.header_response;
        let hover_asset = hdr.dnd_hover_payload::<AssetFileDrag>().is_some();
        let hover_comp = hdr.dnd_hover_payload::<ComponentFileDragPayload>().is_some();
        let hover_prefab = hdr.dnd_hover_payload::<PrefabFileDragPayload>().is_some();
        if hover_asset || hover_comp || hover_prefab {
            ui.painter().rect_stroke(
                hdr.rect,
                2.0,
                egui::Stroke::new(2.0, crate::theme::ACCENT),
                egui::StrokeKind::Outside,
            );
        }
        // Resolve the dropped file to (source, path-within-source).
        let moved: Option<(String, String)> = if hover_asset {
            hdr.dnd_release_payload::<AssetFileDrag>()
                .map(|p| (p.source.clone(), p.path.clone()))
        } else if hover_comp {
            hdr.dnd_release_payload::<ComponentFileDragPayload>()
                .and_then(|p| split_source_path(&p.vfs_path))
        } else if hover_prefab {
            hdr.dnd_release_payload::<PrefabFileDragPayload>()
                .and_then(|p| split_source_path(&p.vfs_path))
        } else {
            None
        };
        if let Some((s, rel)) = moved
            && s == source
        {
            self.pending_move = Some((s, rel, dir_path.to_owned()));
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
        Some(
            entries
                .into_iter()
                .filter(|e| e.is_dir)
                .map(|e| e.name)
                .collect(),
        )
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
                Some(entries) => {
                    self.cached_entries = entries
                        .into_iter()
                        .map(|e| DirEntry {
                            name: e.name,
                            is_dir: e.is_dir,
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

        // Panel-background context menu for "New" — created BEFORE the file entries
        // so their clicks land on the entries (a later/top interact would steal
        // them). Also covers an empty directory.
        let bg = ui.interact(
            ui.max_rect(),
            ui.id().with("browser_new_ctx"),
            egui::Sense::click(),
        );
        let mut new_kind: Option<&'static str> = None;
        bg.context_menu(|ui| {
            ui.menu_button("New", |ui| {
                for (label, kind) in [
                    ("Vertex Layout", "vertex_layout"),
                    ("Material", "material"),
                    ("Material Instance", "material_instance"),
                ] {
                    if ui.button(label).clicked() {
                        new_kind = Some(kind);
                        ui.close();
                    }
                }
            });
        });
        if let Some(kind) = new_kind {
            self.pending_new = Some((source.clone(), dir_path.clone(), kind.to_owned()));
        }

        if self.cached_entries.is_empty() {
            ui.weak("(empty)");
            return;
        }

        // File listing
        let mut rename_start: Option<(String, String, String)> = None;
        let mut delete_start: Option<(String, String)> = None;
        for entry in &self.cached_entries {
            // path within the source (dir_path/name), used for both files + dirs.
            let file_path = if dir_path.is_empty() {
                entry.name.clone()
            } else {
                format!("{dir_path}/{}", entry.name)
            };

            // Inline rename mode for this file (Enter commits, Esc cancels).
            let is_renaming = !entry.is_dir
                && self
                    .renaming
                    .as_ref()
                    .is_some_and(|(s, p, _)| s == &source && p == &file_path);
            if is_renaming {
                let buf = &mut self.renaming.as_mut().unwrap().2;
                ui.text_edit_singleline(buf).request_focus();
                if ui.input(|i| i.key_pressed(egui::Key::Enter)) {
                    let (s, p, new_name) = self.renaming.take().unwrap();
                    if !new_name.is_empty() && new_name != entry.name {
                        self.pending_rename = Some((s, p, new_name));
                    }
                } else if ui.input(|i| i.key_pressed(egui::Key::Escape)) {
                    self.renaming = None;
                }
                continue;
            }

            let icon = if entry.is_dir {
                "\u{1F4C1}"
            } else {
                "\u{1F4C4}"
            };
            let label = format!("{icon} {}", entry.name);

            // Use Button with click_and_drag sense so file entries can
            // initiate drag-and-drop (selectable_label only has click sense).
            let response = ui.add(
                egui::Button::new(&label)
                    .frame(false)
                    .sense(egui::Sense::click_and_drag()),
            );

            // Drag payloads: .component / .prefab keep their import payloads; any
            // other file becomes movable between directories (AssetFileDrag).
            if !entry.is_dir {
                let vfs_path = format!("{source}/{file_path}");
                if entry.name.ends_with(".component") {
                    response.dnd_set_drag_payload(ComponentFileDragPayload { vfs_path });
                } else if entry.name.ends_with(".prefab") {
                    response.dnd_set_drag_payload(PrefabFileDragPayload { vfs_path });
                } else {
                    response.dnd_set_drag_payload(AssetFileDrag {
                        source: source.clone(),
                        path: file_path.clone(),
                    });
                }

                // Right-click a file → Rename / Delete.
                response.context_menu(|ui| {
                    if ui.button("Rename").clicked() {
                        rename_start =
                            Some((source.clone(), file_path.clone(), entry.name.clone()));
                        ui.close();
                    }
                    if ui.button("Delete").clicked() {
                        delete_start = Some((source.clone(), file_path.clone()));
                        ui.close();
                    }
                });
            }

            // Select a file as an asset (drives the asset inspector).
            if !entry.is_dir && response.clicked() {
                self.selected_file = Some((source.clone(), file_path.clone()));
            }

            if response.double_clicked() && entry.is_dir {
                let tree_key = format!("{source}/{file_path}");
                self.expanded.insert(tree_key);
                self.selected = Some((source.clone(), file_path.clone()));
                self.cached_key = None;
                break;
            }
        }
        if let Some(r) = rename_start {
            self.renaming = Some(r);
        }
        if let Some(d) = delete_start {
            self.pending_delete = Some(d);
        }

        // Drop target: accept payloads dragged from inspector or world inspector.
        //
        // IMPORTANT: Check payload type with dnd_hover_payload (non-destructive)
        // BEFORE calling dnd_release_payload (destructive — removes payload from
        // egui context regardless of downcast success). Only call release for the
        // matching type.
        let drop_resp = ui.interact(
            ui.max_rect(),
            ui.id().with("comp_export_drop"),
            egui::Sense::hover(),
        );
        let hovering_component = drop_resp
            .dnd_hover_payload::<ComponentDragPayload>()
            .is_some();
        let hovering_entity = drop_resp.dnd_hover_payload::<Entity>().is_some();

        if hovering_component
            && let Some(payload) = drop_resp.dnd_release_payload::<ComponentDragPayload>()
        {
            let vfs_dir = if dir_path.is_empty() {
                source.clone()
            } else {
                format!("{source}/{dir_path}")
            };
            self.pending_component_export = Some((payload.entity, payload.name, vfs_dir));
        } else if hovering_entity && let Some(payload) = drop_resp.dnd_release_payload::<Entity>() {
            let vfs_dir = if dir_path.is_empty() {
                source.clone()
            } else {
                format!("{source}/{dir_path}")
            };
            self.pending_prefab_export = Some((*payload, vfs_dir));
        }

        // Visual hover feedback for any droppable payload
        if hovering_component || hovering_entity {
            ui.painter().rect_stroke(
                drop_resp.rect,
                4.0,
                egui::Stroke::new(2.0, crate::theme::ACCENT),
                egui::StrokeKind::Outside,
            );
        }

    }
}
