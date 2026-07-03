//! Central asset DB (prototype).
//!
//! The single source of truth for file-backed identity: `guid -> (mount, path)`,
//! per-asset import settings, and a content hash for reload detection. It is an
//! in-memory index, (de)serialized as ONE file through the VFS (not a DB
//! engine — see the design notes; SQLite/sled fight the whole-blob VFS model).
//!
//! Solo workflow: this central record is the source of truth (no per-file
//! `.meta` sidecars); a file move is detected by matching `source_hash`.
//!
//! ## Identity invariant
//!
//! The DB enforces a **bijection `guid <-> path`** by construction: two indices
//! kept consistent by every mutation, and checked operations that *report* a
//! typed [`DbError`] rather than silently overwriting. Random v4 collisions are
//! ignored (≈`10^-25` at a million assets — unhittable, untestable). The real
//! hazard is *non-random duplication* (a bad git merge of the DB file, importing
//! another project that reuses guids); [`AssetDb::from_records`] validates the
//! bijection on load and surfaces every conflict so a policy layer can resolve
//! it (reassign / prompt) — the DB never lets the invariant break quietly.
//!
//! Persistence (RON in editor, baked binary at runtime) goes through
//! [`to_records`](AssetDb::to_records) / [`from_records`](AssetDb::from_records).

use std::collections::{BTreeMap, HashMap};

use serde::{Deserialize, Serialize};

use crate::source::Guid;

/// A `(mount, path)` location — paths aren't unique across VFS mounts.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct AssetPath {
    pub mount: String,
    pub path: String,
}

impl AssetPath {
    pub fn new(mount: impl Into<String>, path: impl Into<String>) -> Self {
        Self {
            mount: mount.into(),
            path: path.into(),
        }
    }
}

impl std::fmt::Display for AssetPath {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}:{}", self.mount, self.path)
    }
}

/// One DB record for a file-backed asset.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AssetRecord {
    /// Current location of the asset.
    pub path: AssetPath,
    /// Owning asset kind (`AssetKind::NAME`) — routes reload to the right cache.
    pub kind: String,
    /// Content hash of the last import (rename/reimport detection).
    pub source_hash: u64,
    /// Serialized per-kind import settings (deserialized into `K::Settings`).
    #[serde(default)]
    pub settings: Option<String>,
    /// References to other assets this one depends on, keyed by role — e.g. a
    /// mesh's `"layout"` -> the vertex-layout asset's guid. Lets a consumer (a
    /// manager) resolve a dependency's shared resource without the data living in
    /// this asset's blob, and gives reload a dependency edge to follow.
    #[serde(default)]
    pub references: BTreeMap<String, Guid>,
}

impl AssetRecord {
    /// The guid this asset references under `role`, if any (e.g. `"layout"`).
    pub fn reference(&self, role: &str) -> Option<Guid> {
        self.references.get(role).copied()
    }
}

/// A violation (attempt) of the `guid <-> path` bijection.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DbError {
    /// `guid` is already bound to a *different* asset.
    GuidConflict {
        guid: Guid,
        existing: AssetPath,
        incoming: AssetPath,
    },
    /// `path` is already registered under a *different* guid.
    PathConflict {
        path: AssetPath,
        existing: Guid,
        incoming: Guid,
    },
    /// Operation referenced a guid the DB doesn't know.
    UnknownGuid(Guid),
}

impl std::fmt::Display for DbError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::GuidConflict {
                guid,
                existing,
                incoming,
            } => write!(
                f,
                "guid {guid} already maps to {existing}, cannot also map to {incoming}"
            ),
            Self::PathConflict {
                path,
                existing,
                incoming,
            } => write!(
                f,
                "path {path} already owned by guid {existing}, cannot rebind to {incoming}"
            ),
            Self::UnknownGuid(g) => write!(f, "unknown guid {g}"),
        }
    }
}

impl std::error::Error for DbError {}

/// In-memory asset registry with an enforced `guid <-> path` bijection.
#[derive(Debug, Default)]
pub struct AssetDb {
    by_guid: HashMap<Guid, AssetRecord>,
    by_path: HashMap<AssetPath, Guid>,
}

impl AssetDb {
    /// Create an empty DB.
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of registered assets.
    pub fn len(&self) -> usize {
        self.by_guid.len()
    }

    /// Whether the DB is empty.
    pub fn is_empty(&self) -> bool {
        self.by_guid.is_empty()
    }

    /// Resolve a guid to its record (mount/path/settings).
    pub fn record(&self, guid: &Guid) -> Option<&AssetRecord> {
        self.by_guid.get(guid)
    }

    /// Reverse lookup: the guid currently bound to `path`.
    pub fn guid_of(&self, path: &AssetPath) -> Option<Guid> {
        self.by_path.get(path).copied()
    }

    /// Replace a record's per-kind `settings` (the asset's editable data — e.g. a
    /// vertex layout's parameters). Returns `false` if the guid is unknown. Does
    /// not touch the `guid <-> path` bijection.
    pub fn set_settings(&mut self, guid: &Guid, settings: Option<String>) -> bool {
        match self.by_guid.get_mut(guid) {
            Some(record) => {
                record.settings = settings;
                true
            }
            None => false,
        }
    }

    /// Set (or replace) the record's named reference — e.g. rebinding a mesh's
    /// `"layout"`. Returns `false` if `guid` is unknown.
    pub fn set_reference(&mut self, guid: &Guid, role: &str, target: Guid) -> bool {
        match self.by_guid.get_mut(guid) {
            Some(record) => {
                record.references.insert(role.to_owned(), target);
                true
            }
            None => false,
        }
    }

    /// Mint a fresh guid not already present. The loop guards the *invariant*
    /// (uniqueness), not random collisions — it effectively never iterates.
    fn mint(&self) -> Guid {
        loop {
            let g = Guid::new();
            if !self.by_guid.contains_key(&g) {
                return g;
            }
        }
    }

    /// Register a freshly-scanned file. **Idempotent by path:** a known path
    /// keeps its guid (re-scan, content hash refreshed); a new path mints a
    /// fresh guid. This is the normal editor import entry point.
    pub fn register_path(&mut self, path: AssetPath, kind: &str, source_hash: u64) -> Guid {
        if let Some(&guid) = self.by_path.get(&path) {
            if let Some(rec) = self.by_guid.get_mut(&guid) {
                rec.source_hash = source_hash; // content may have changed
            }
            return guid;
        }
        let guid = self.mint();
        self.by_guid.insert(
            guid,
            AssetRecord {
                path: path.clone(),
                kind: kind.to_owned(),
                source_hash,
                settings: None,
                references: BTreeMap::new(),
            },
        );
        self.by_path.insert(path, guid);
        guid
    }

    /// Insert a record under a *specific* guid (loading the DB, importing
    /// another project). Enforces the bijection and reports the precise
    /// conflict instead of overwriting:
    /// - same guid + same path → idempotent field update;
    /// - same guid + different path → [`DbError::GuidConflict`];
    /// - different guid + same path → [`DbError::PathConflict`].
    pub fn insert(&mut self, guid: Guid, record: AssetRecord) -> Result<(), DbError> {
        if let Some(existing) = self.by_guid.get(&guid) {
            if existing.path != record.path {
                return Err(DbError::GuidConflict {
                    guid,
                    existing: existing.path.clone(),
                    incoming: record.path,
                });
            }
            // same asset: update fields, bijection unchanged
            self.by_guid.insert(guid, record);
            return Ok(());
        }
        if let Some(&owner) = self.by_path.get(&record.path)
            && owner != guid
        {
            return Err(DbError::PathConflict {
                path: record.path,
                existing: owner,
                incoming: guid,
            });
        }
        self.by_path.insert(record.path.clone(), guid);
        self.by_guid.insert(guid, record);
        Ok(())
    }

    /// Rename/move: rebind an existing guid to a new path (identity preserved).
    /// Fails if the guid is unknown or the new path is owned by another guid.
    pub fn rebind(&mut self, guid: Guid, new_path: AssetPath) -> Result<(), DbError> {
        let old_path = self
            .by_guid
            .get(&guid)
            .ok_or(DbError::UnknownGuid(guid))?
            .path
            .clone();
        if old_path == new_path {
            return Ok(());
        }
        if let Some(&owner) = self.by_path.get(&new_path)
            && owner != guid
        {
            return Err(DbError::PathConflict {
                path: new_path,
                existing: owner,
                incoming: guid,
            });
        }
        self.by_path.remove(&old_path);
        self.by_path.insert(new_path.clone(), guid);
        self.by_guid.get_mut(&guid).unwrap().path = new_path;
        Ok(())
    }

    /// Forget an asset (e.g. file deleted). Returns the removed record.
    pub fn remove(&mut self, guid: &Guid) -> Option<AssetRecord> {
        let rec = self.by_guid.remove(guid)?;
        self.by_path.remove(&rec.path);
        Some(rec)
    }

    /// Records for persistence (serialize this).
    pub fn to_records(&self) -> Vec<(Guid, AssetRecord)> {
        self.by_guid.iter().map(|(g, r)| (*g, r.clone())).collect()
    }

    /// `(guid, relative-path)` for every record under `mount` whose path starts
    /// with `prefix` (mount-relative). Used by scan to detect deletions.
    pub fn entries_under(&self, mount: &str, prefix: &str) -> Vec<(Guid, String)> {
        self.by_guid
            .iter()
            .filter(|(_, r)| r.path.mount == mount && r.path.path.starts_with(prefix))
            .map(|(g, r)| (*g, r.path.path.clone()))
            .collect()
    }

    /// Build a DB from deserialized records, **validating the bijection**. A
    /// hand-merged DB file can contain duplicate guids/paths; every offending
    /// record is skipped and its conflict returned for the caller to resolve.
    pub fn from_records(
        records: impl IntoIterator<Item = (Guid, AssetRecord)>,
    ) -> (Self, Vec<DbError>) {
        let mut db = Self::new();
        let mut conflicts = Vec::new();
        for (guid, record) in records {
            if let Err(e) = db.insert(guid, record) {
                conflicts.push(e);
            }
        }
        (db, conflicts)
    }

    /// Serialize one mount's records to RON (the editor format — human-readable,
    /// diffable). Paths are stored **mount-relative**: the `mount` is dropped and
    /// re-stamped on load, so a pack is portable across mount names. This file
    /// lives at the mount root and ships with the pack.
    pub fn to_ron_for_mount(&self, mount: &str) -> Result<String, ron::Error> {
        let entries: Vec<DbEntry> = self
            .by_guid
            .iter()
            .filter(|(_, r)| r.path.mount == mount)
            .map(|(guid, r)| DbEntry {
                guid: *guid,
                path: r.path.path.clone(),
                kind: r.kind.clone(),
                source_hash: r.source_hash,
                settings: r.settings.clone(),
                references: r.references.clone(),
            })
            .collect();
        let file = DbFile {
            version: DB_VERSION,
            entries,
        };
        ron::ser::to_string_pretty(&file, ron::ser::PrettyConfig::default())
    }

    /// Merge a mount's RON DB file into this (global) registry, stamping `mount`
    /// onto every path. The bijection guard reports cross-pack guid/path
    /// conflicts (returned, skipped) rather than overwriting. Re-merging the same
    /// mount is idempotent.
    pub fn merge_ron(
        &mut self,
        mount: &str,
        text: &str,
    ) -> Result<Vec<DbError>, ron::error::SpannedError> {
        let file: DbFile = ron::from_str(text)?;
        let mut conflicts = Vec::new();
        for entry in file.entries {
            let record = AssetRecord {
                path: AssetPath::new(mount, entry.path),
                kind: entry.kind,
                source_hash: entry.source_hash,
                settings: entry.settings,
                references: entry.references,
            };
            if let Err(e) = self.insert(entry.guid, record) {
                conflicts.push(e);
            }
        }
        Ok(conflicts)
    }
}

/// On-disk form of one mount's DB, versioned for forward-compatible migrations.
#[derive(Serialize, Deserialize)]
struct DbFile {
    version: u32,
    entries: Vec<DbEntry>,
}

/// One serialized record. `path` is mount-relative (no mount segment).
#[derive(Serialize, Deserialize)]
struct DbEntry {
    guid: Guid,
    path: String,
    kind: String,
    source_hash: u64,
    #[serde(default)]
    settings: Option<String>,
    #[serde(default)]
    references: BTreeMap<String, Guid>,
}

const DB_VERSION: u32 = 1;

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(mount: &str, path: &str) -> AssetRecord {
        AssetRecord {
            path: AssetPath::new(mount, path),
            kind: "mesh".into(),
            source_hash: 0,
            settings: None,
            references: BTreeMap::new(),
        }
    }

    #[test]
    fn register_path_is_idempotent() {
        let mut db = AssetDb::new();
        let p = AssetPath::new("assets", "a.glb");
        let g1 = db.register_path(p.clone(), "mesh", 1);
        let g2 = db.register_path(p.clone(), "mesh", 2); // re-scan, hash changed
        assert_eq!(g1, g2);
        assert_eq!(db.len(), 1);
        assert_eq!(db.record(&g1).unwrap().source_hash, 2);
    }

    #[test]
    fn insert_same_guid_same_path_updates() {
        let mut db = AssetDb::new();
        let g = Guid::new();
        db.insert(g, rec("assets", "a.glb")).unwrap();
        let mut r = rec("assets", "a.glb");
        r.source_hash = 9;
        db.insert(g, r).unwrap(); // idempotent update
        assert_eq!(db.len(), 1);
        assert_eq!(db.record(&g).unwrap().source_hash, 9);
    }

    #[test]
    fn duplicate_guid_is_a_conflict() {
        let mut db = AssetDb::new();
        let g = Guid::new();
        db.insert(g, rec("assets", "a.glb")).unwrap();
        let err = db.insert(g, rec("assets", "b.glb")).unwrap_err();
        assert!(matches!(err, DbError::GuidConflict { .. }));
        // invariant intact: g still points at the original
        assert_eq!(
            db.record(&g).unwrap().path,
            AssetPath::new("assets", "a.glb")
        );
    }

    #[test]
    fn path_taken_by_another_guid_is_a_conflict() {
        let mut db = AssetDb::new();
        db.insert(Guid::new(), rec("assets", "a.glb")).unwrap();
        let err = db.insert(Guid::new(), rec("assets", "a.glb")).unwrap_err();
        assert!(matches!(err, DbError::PathConflict { .. }));
    }

    #[test]
    fn rebind_moves_and_guards() {
        let mut db = AssetDb::new();
        let g = db.register_path(AssetPath::new("assets", "old.glb"), "mesh", 0);
        db.rebind(g, AssetPath::new("assets", "new.glb")).unwrap();
        assert_eq!(db.guid_of(&AssetPath::new("assets", "new.glb")), Some(g));
        assert_eq!(db.guid_of(&AssetPath::new("assets", "old.glb")), None);

        // rebind onto a path owned by another guid → conflict
        let other = db.register_path(AssetPath::new("assets", "taken.glb"), "mesh", 0);
        let err = db
            .rebind(g, AssetPath::new("assets", "taken.glb"))
            .unwrap_err();
        assert!(matches!(err, DbError::PathConflict { .. }));
        let _ = other;

        // unknown guid → typed error
        assert!(matches!(
            db.rebind(Guid::new(), AssetPath::new("assets", "x.glb")),
            Err(DbError::UnknownGuid(_))
        ));
    }

    #[test]
    fn per_mount_ron_is_relative_and_restampable() {
        let mut db = AssetDb::new();
        let g = db.register_path(AssetPath::new("packA", "mesh/a.glb"), "mesh", 7);
        db.register_path(AssetPath::new("packB", "b.png"), "texture", 0); // other mount, excluded

        let text = db.to_ron_for_mount("packA").unwrap();

        // Mount the same pack file under a *different* name → mount is re-stamped.
        let mut global = AssetDb::new();
        let conflicts = global.merge_ron("mounted", &text).unwrap();
        assert!(conflicts.is_empty());
        assert_eq!(global.len(), 1); // only packA's record was in the file
        let rec = global.record(&g).unwrap();
        assert_eq!(rec.path, AssetPath::new("mounted", "mesh/a.glb"));
        assert_eq!(rec.source_hash, 7);
    }

    #[test]
    fn merge_reports_cross_pack_guid_conflict() {
        let g = Guid::new();
        let mut packa = AssetDb::new();
        packa.insert(g, rec("packA", "a.glb")).unwrap();
        let text = packa.to_ron_for_mount("packA").unwrap();

        // A second pack reusing the same guid (a fork) → merge surfaces it.
        let mut global = AssetDb::new();
        global.insert(g, rec("packB", "other.glb")).unwrap();
        let conflicts = global.merge_ron("packA", &text).unwrap();
        assert_eq!(conflicts.len(), 1);
        assert!(matches!(conflicts[0], DbError::GuidConflict { .. }));
    }

    #[test]
    fn from_records_validates_and_reports() {
        let g = Guid::new();
        let records = vec![
            (g, rec("assets", "a.glb")),
            (g, rec("assets", "b.glb")), // duplicate guid → conflict, skipped
        ];
        let (db, conflicts) = AssetDb::from_records(records);
        assert_eq!(db.len(), 1);
        assert_eq!(conflicts.len(), 1);
        assert!(matches!(conflicts[0], DbError::GuidConflict { .. }));
    }
}
