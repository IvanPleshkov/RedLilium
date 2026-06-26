//! Per-mount DB persistence through the VFS. Each mount (asset pack) carries its
//! own DB file at its root (`<mount>/assets.db`, RON) — paths inside are
//! mount-relative so the pack is portable. The editor's in-memory [`AssetDb`] is
//! the merged view of every mounted file.
//!
//! These are thin async wrappers over [`AssetDb::to_ron_for_mount`] /
//! [`AssetDb::merge_ron`]. For spawning on the IO runtime, read the bytes there
//! and call `merge_ron` on the main thread (it borrows the DB mutably).

use redlilium_vfs::Vfs;

use crate::db::{AssetDb, DbError};
use crate::error::AssetError;

/// File name of a mount's DB, at the mount root.
pub const DB_FILE_NAME: &str = "assets.db";

/// Write `mount`'s records to `<mount>/assets.db` (RON) through `vfs`.
pub async fn save_mount_db(db: &AssetDb, vfs: &Vfs, mount: &str) -> Result<(), AssetError> {
    let text = db
        .to_ron_for_mount(mount)
        .map_err(|e| AssetError::Io(e.to_string()))?;
    let path = format!("{mount}/{DB_FILE_NAME}");
    vfs.write(&path, text.into_bytes())
        .await
        .map_err(|e| AssetError::Io(e.to_string()))
}

/// Read `<mount>/assets.db` through `vfs` and merge it into `db`, stamping
/// `mount` onto every path. Returns any duplicate-record conflicts skipped on
/// merge (the bijection guard).
pub async fn load_mount_db(
    db: &mut AssetDb,
    vfs: &Vfs,
    mount: &str,
) -> Result<Vec<DbError>, AssetError> {
    let path = format!("{mount}/{DB_FILE_NAME}");
    let bytes = vfs
        .read(&path)
        .await
        .map_err(|e| AssetError::Io(e.to_string()))?;
    let text = String::from_utf8(bytes).map_err(|e| AssetError::Decode(e.to_string()))?;
    db.merge_ron(mount, &text)
        .map_err(|e| AssetError::Decode(e.to_string()))
}
