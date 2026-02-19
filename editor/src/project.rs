use std::path::Path;

use redlilium_vfs::{FileSystemProvider, SftpConfig, SftpProvider, Vfs};
use serde::Deserialize;

/// Top-level project configuration loaded from `project.toml`.
#[derive(Debug, Clone, Deserialize)]
pub struct ProjectConfig {
    pub project: ProjectInfo,
    #[serde(default)]
    pub mount: Vec<MountConfig>,
}

/// General project information.
#[derive(Debug, Clone, Deserialize)]
pub struct ProjectInfo {
    pub name: String,
}

/// A single VFS mount point definition.
///
/// The `type` field selects the provider: `"filesystem"` (default) or `"sftp"`.
/// SFTP mounts use the `host`, `port`, `username`, and `key` fields.
#[derive(Debug, Clone, Deserialize)]
pub struct MountConfig {
    pub name: String,
    pub path: String,
    #[serde(default)]
    pub default: bool,
    /// `"filesystem"` (default) or `"sftp"`.
    #[serde(default = "default_mount_type")]
    pub r#type: String,
    // SFTP-specific fields (ignored for filesystem mounts).
    pub host: Option<String>,
    pub port: Option<u16>,
    pub username: Option<String>,
    pub key: Option<String>,
}

fn default_mount_type() -> String {
    "filesystem".into()
}

/// Load a project config from a TOML file.
///
/// Returns `Err` with a human-readable message if the file cannot be read
/// or parsed.
pub fn load_project(path: &Path) -> Result<ProjectConfig, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("failed to read {}: {e}", path.display()))?;
    toml::from_str(&content).map_err(|e| format!("failed to parse {}: {e}", path.display()))
}

/// Build a [`Vfs`] from a project config.
///
/// Creates the appropriate provider for each mount and sets the default source
/// if one is marked with `default = true`.
pub fn build_vfs(config: &ProjectConfig) -> Vfs {
    let mut vfs = Vfs::new();

    for mount in &config.mount {
        match mount.r#type.as_str() {
            "filesystem" => {
                log::info!(
                    "VFS mount: \"{}\" -> filesystem {:?}",
                    mount.name,
                    mount.path
                );
                vfs.mount(&mount.name, FileSystemProvider::new(&mount.path));
            }
            "sftp" => {
                let sftp_config = SftpConfig {
                    host: mount.host.clone().unwrap_or_else(|| "localhost".into()),
                    port: mount.port.unwrap_or(22),
                    username: mount.username.clone().unwrap_or_else(|| "root".into()),
                    key_path: mount
                        .key
                        .clone()
                        .unwrap_or_else(|| "~/.ssh/id_ed25519".into()),
                    remote_root: mount.path.clone(),
                };
                log::info!(
                    "VFS mount: \"{}\" -> sftp {}@{}:{}:{}",
                    mount.name,
                    sftp_config.username,
                    sftp_config.host,
                    sftp_config.port,
                    sftp_config.remote_root,
                );
                match SftpProvider::connect(sftp_config) {
                    Ok(provider) => {
                        vfs.mount(&mount.name, provider);
                    }
                    Err(e) => {
                        log::error!("Failed to connect SFTP mount \"{}\": {e}", mount.name);
                    }
                }
            }
            other => {
                log::warn!("Unknown mount type \"{}\" for \"{}\"", other, mount.name);
            }
        }
    }

    if let Some(default_mount) = config.mount.iter().find(|m| m.default) {
        vfs.set_default(&default_mount.name);
    }

    vfs
}

/// Load project config, falling back to a default if the file doesn't exist.
pub fn load_or_default(path: &Path) -> (ProjectConfig, Vfs) {
    let config = match load_project(path) {
        Ok(config) => {
            log::info!(
                "Loaded project: {} ({} mounts)",
                config.project.name,
                config.mount.len()
            );
            config
        }
        Err(e) => {
            log::warn!("No project file ({e}), using defaults");
            ProjectConfig {
                project: ProjectInfo {
                    name: "Untitled".into(),
                },
                mount: vec![MountConfig {
                    name: "assets".into(),
                    path: "./assets".into(),
                    default: true,
                    r#type: "filesystem".into(),
                    host: None,
                    port: None,
                    username: None,
                    key: None,
                }],
            }
        }
    };

    let vfs = build_vfs(&config);
    (config, vfs)
}
