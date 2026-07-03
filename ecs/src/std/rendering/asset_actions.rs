//! Undoable edits of asset DB records.
//!
//! The editor guarantee is that UI never mutates state directly — every change
//! flows through [`EditAction`]s so it lands in the undo history. Asset-record
//! edits (inspector settings, reference rebinds) are world mutations too (the
//! [`AssetDb`] is a world resource), so they get action types here instead of
//! ad-hoc `resource_mut` writes.
//!
//! Both `apply` and `undo` feed the ordinary hot-reload chain
//! ([`ChangedAssets`](super::ChangedAssets) → invalidate → reload), so undoing
//! a material edit visibly reverts the scene, and mark the record's mount in
//! [`DirtyMounts`] so the editor persists its `assets.db`.

use redlilium_assets::{AssetDb, Guid};
use redlilium_core::abstract_editor::{EditAction, EditActionError, EditActionResult};

use crate::World;

/// Mounts whose asset DB has pending un-persisted edits. Written by the asset
/// edit actions; the editor drains it once per frame and writes each mount's
/// `assets.db`.
#[derive(Default)]
pub struct DirtyMounts {
    mounts: Vec<String>,
}

impl DirtyMounts {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn push(&mut self, mount: impl Into<String>) {
        let mount = mount.into();
        if !self.mounts.contains(&mount) {
            self.mounts.push(mount);
        }
    }

    pub fn drain(&mut self) -> Vec<String> {
        std::mem::take(&mut self.mounts)
    }
}

/// Feed an applied/undone record edit into hot reload + persistence.
fn touch(world: &mut World, guid: Guid) {
    let mount = world
        .resource::<AssetDb>()
        .record(&guid)
        .map(|r| r.path.mount.clone());
    if world.has_resource::<super::ChangedAssets>() {
        world.resource_mut::<super::ChangedAssets>().push(guid);
    }
    if let Some(mount) = mount
        && world.has_resource::<DirtyMounts>()
    {
        world.resource_mut::<DirtyMounts>().push(mount);
    }
}

/// Reversible edit of a record's per-kind `settings` (material properties,
/// texture import settings, layout parameters, …).
pub struct SetAssetSettingsAction {
    pub guid: Guid,
    pub old: Option<String>,
    pub new: Option<String>,
}

impl std::fmt::Debug for SetAssetSettingsAction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SetAssetSettingsAction")
            .field("guid", &self.guid)
            .finish()
    }
}

impl SetAssetSettingsAction {
    fn set(&self, world: &mut World, value: &Option<String>) -> EditActionResult {
        let ok = world
            .resource_mut::<AssetDb>()
            .set_settings(&self.guid, value.clone());
        if !ok {
            return Err(EditActionError::TargetNotFound(format!(
                "asset record {:?} not in DB",
                self.guid
            )));
        }
        touch(world, self.guid);
        Ok(())
    }
}

impl EditAction<World> for SetAssetSettingsAction {
    fn apply(&mut self, world: &mut World) -> EditActionResult {
        self.set(world, &self.new.clone())
    }

    fn undo(&mut self, world: &mut World) -> EditActionResult {
        self.set(world, &self.old.clone())
    }

    fn description(&self) -> &str {
        "Edit Asset Settings"
    }

    /// Continuous widgets (color pickers, drag values) emit an edit per frame;
    /// consecutive edits of the same record coalesce into one undo entry.
    fn merge(&mut self, other: Box<dyn EditAction<World>>) -> Option<Box<dyn EditAction<World>>> {
        if let Some(other) = other.as_any().downcast_ref::<Self>()
            && self.guid == other.guid
        {
            self.new = other.new.clone();
            return None; // consumed — keep first old, use latest new
        }
        Some(other)
    }
}

/// Reversible edit of a record's named reference (e.g. a mesh's `"layout"`).
/// `None` removes the role.
pub struct SetAssetReferenceAction {
    pub guid: Guid,
    pub role: String,
    pub old: Option<Guid>,
    pub new: Option<Guid>,
}

impl std::fmt::Debug for SetAssetReferenceAction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SetAssetReferenceAction")
            .field("guid", &self.guid)
            .field("role", &self.role)
            .finish()
    }
}

impl SetAssetReferenceAction {
    fn set(&self, world: &mut World, value: Option<Guid>) -> EditActionResult {
        let ok = world
            .resource_mut::<AssetDb>()
            .set_reference(&self.guid, &self.role, value);
        if !ok {
            return Err(EditActionError::TargetNotFound(format!(
                "asset record {:?} not in DB",
                self.guid
            )));
        }
        touch(world, self.guid);
        Ok(())
    }
}

impl EditAction<World> for SetAssetReferenceAction {
    fn apply(&mut self, world: &mut World) -> EditActionResult {
        self.set(world, self.new)
    }

    fn undo(&mut self, world: &mut World) -> EditActionResult {
        self.set(world, self.old)
    }

    fn description(&self) -> &str {
        "Edit Asset Reference"
    }
}
