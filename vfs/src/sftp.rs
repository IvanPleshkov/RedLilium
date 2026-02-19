use std::path::Path;
use std::sync::Arc;
use std::sync::mpsc;
use std::thread;

use crate::error::VfsError;
use crate::provider::{VfsFuture, VfsProvider};

/// Configuration for connecting to an SFTP server.
pub struct SftpConfig {
    pub host: String,
    pub port: u16,
    pub username: String,
    pub key_path: String,
    pub remote_root: String,
}

/// VFS provider for remote SFTP access.
///
/// Spawns a dedicated background thread with a tokio runtime.
/// The SSH/SFTP session lives on that thread. Commands are dispatched
/// via [`std::sync::mpsc`] and results arrive through per-operation
/// [`tokio::sync::oneshot`] channels.
///
/// # Example `project.toml`
///
/// ```toml
/// [[mount]]
/// name = "remote"
/// type = "sftp"
/// host = "192.168.1.100"
/// port = 22
/// username = "deploy"
/// key = "~/.ssh/id_ed25519"
/// path = "/data/assets"
/// ```
pub struct SftpProvider {
    sender: mpsc::Sender<SftpCommand>,
    _thread: thread::JoinHandle<()>,
}

enum SftpCommand {
    Read {
        path: String,
        reply: tokio::sync::oneshot::Sender<Result<Vec<u8>, VfsError>>,
    },
    Exists {
        path: String,
        reply: tokio::sync::oneshot::Sender<Result<bool, VfsError>>,
    },
    ListDir {
        path: String,
        reply: tokio::sync::oneshot::Sender<Result<Vec<String>, VfsError>>,
    },
    Write {
        path: String,
        data: Vec<u8>,
        reply: tokio::sync::oneshot::Sender<Result<(), VfsError>>,
    },
    Delete {
        path: String,
        reply: tokio::sync::oneshot::Sender<Result<(), VfsError>>,
    },
    CreateDir {
        path: String,
        reply: tokio::sync::oneshot::Sender<Result<(), VfsError>>,
    },
    Shutdown,
}

fn sftp_err(msg: impl std::fmt::Display) -> VfsError {
    VfsError::Io(std::io::Error::other(msg.to_string()))
}

/// Expand leading `~` to `$HOME`.
fn expand_tilde(path: &str) -> String {
    if let Some(rest) = path.strip_prefix('~')
        && let Ok(home) = std::env::var("HOME")
    {
        return format!("{home}{rest}");
    }
    path.to_owned()
}

impl SftpProvider {
    /// Connect to an SFTP server.
    ///
    /// This blocks the calling thread until the SSH connection and SFTP
    /// subsystem are established. Call during startup before the event loop.
    pub fn connect(config: SftpConfig) -> Result<Self, VfsError> {
        let (cmd_tx, cmd_rx) = mpsc::channel::<SftpCommand>();
        let (ready_tx, ready_rx) = mpsc::channel::<Result<(), VfsError>>();

        let handle = thread::Builder::new()
            .name("sftp-worker".into())
            .spawn(move || {
                let rt = tokio::runtime::Builder::new_current_thread()
                    .enable_all()
                    .build()
                    .expect("failed to create SFTP tokio runtime");

                rt.block_on(async move {
                    match establish_session(&config).await {
                        Ok(sftp) => {
                            let _ = ready_tx.send(Ok(()));
                            run_command_loop(sftp, &config.remote_root, cmd_rx).await;
                        }
                        Err(e) => {
                            let _ = ready_tx.send(Err(e));
                        }
                    }
                });
            })
            .map_err(VfsError::Io)?;

        ready_rx
            .recv()
            .map_err(|_| sftp_err("SFTP thread died during connect"))??;

        Ok(Self {
            sender: cmd_tx,
            _thread: handle,
        })
    }
}

impl Drop for SftpProvider {
    fn drop(&mut self) {
        let _ = self.sender.send(SftpCommand::Shutdown);
    }
}

impl VfsProvider for SftpProvider {
    fn read(&self, path: &str) -> VfsFuture<Vec<u8>> {
        let (tx, rx) = tokio::sync::oneshot::channel();
        let _ = self.sender.send(SftpCommand::Read {
            path: path.to_owned(),
            reply: tx,
        });
        Box::pin(async move { rx.await.map_err(|_| sftp_err("SFTP connection closed"))? })
    }

    fn exists(&self, path: &str) -> VfsFuture<bool> {
        let (tx, rx) = tokio::sync::oneshot::channel();
        let _ = self.sender.send(SftpCommand::Exists {
            path: path.to_owned(),
            reply: tx,
        });
        Box::pin(async move { rx.await.map_err(|_| sftp_err("SFTP connection closed"))? })
    }

    fn list_dir(&self, path: &str) -> VfsFuture<Vec<String>> {
        let (tx, rx) = tokio::sync::oneshot::channel();
        let _ = self.sender.send(SftpCommand::ListDir {
            path: path.to_owned(),
            reply: tx,
        });
        Box::pin(async move { rx.await.map_err(|_| sftp_err("SFTP connection closed"))? })
    }

    fn is_read_only(&self) -> bool {
        false
    }

    fn write(&self, path: &str, data: Vec<u8>) -> VfsFuture<()> {
        let (tx, rx) = tokio::sync::oneshot::channel();
        let _ = self.sender.send(SftpCommand::Write {
            path: path.to_owned(),
            data,
            reply: tx,
        });
        Box::pin(async move { rx.await.map_err(|_| sftp_err("SFTP connection closed"))? })
    }

    fn delete(&self, path: &str) -> VfsFuture<()> {
        let (tx, rx) = tokio::sync::oneshot::channel();
        let _ = self.sender.send(SftpCommand::Delete {
            path: path.to_owned(),
            reply: tx,
        });
        Box::pin(async move { rx.await.map_err(|_| sftp_err("SFTP connection closed"))? })
    }

    fn create_dir(&self, path: &str) -> VfsFuture<()> {
        let (tx, rx) = tokio::sync::oneshot::channel();
        let _ = self.sender.send(SftpCommand::CreateDir {
            path: path.to_owned(),
            reply: tx,
        });
        Box::pin(async move { rx.await.map_err(|_| sftp_err("SFTP connection closed"))? })
    }
}

// ---------------------------------------------------------------------------
// Background thread implementation
// ---------------------------------------------------------------------------

struct SshHandler;

#[async_trait::async_trait]
impl russh::client::Handler for SshHandler {
    type Error = russh::Error;

    async fn check_server_key(
        &mut self,
        _server_public_key: &russh_keys::PublicKey,
    ) -> Result<bool, Self::Error> {
        // Accept all host keys for now.
        Ok(true)
    }
}

async fn establish_session(
    config: &SftpConfig,
) -> Result<russh_sftp::client::SftpSession, VfsError> {
    let key_path = expand_tilde(&config.key_path);
    let key = russh_keys::load_secret_key(Path::new(&key_path), None)
        .map_err(|e| sftp_err(format!("failed to load SSH key {key_path}: {e}")))?;

    let ssh_config = russh::client::Config::default();
    let mut handle = russh::client::connect(
        Arc::new(ssh_config),
        (&*config.host, config.port),
        SshHandler,
    )
    .await
    .map_err(|e| {
        sftp_err(format!(
            "SSH connect to {}:{}: {e}",
            config.host, config.port
        ))
    })?;

    let authenticated = handle
        .authenticate_publickey(&config.username, Arc::new(key))
        .await
        .map_err(|e| sftp_err(format!("SSH auth as {}: {e}", config.username)))?;

    if !authenticated {
        return Err(sftp_err(format!(
            "SSH authentication failed for {}@{}",
            config.username, config.host
        )));
    }

    let channel = handle
        .channel_open_session()
        .await
        .map_err(|e| sftp_err(format!("SSH channel open: {e}")))?;

    channel
        .request_subsystem(true, "sftp")
        .await
        .map_err(|e| sftp_err(format!("SFTP subsystem: {e}")))?;

    let sftp = russh_sftp::client::SftpSession::new(channel.into_stream())
        .await
        .map_err(|e| sftp_err(format!("SFTP session init: {e}")))?;

    log::info!(
        "SFTP connected: {}@{}:{}",
        config.username,
        config.host,
        config.port
    );

    Ok(sftp)
}

async fn run_command_loop(
    sftp: russh_sftp::client::SftpSession,
    remote_root: &str,
    cmd_rx: mpsc::Receiver<SftpCommand>,
) {
    let root = remote_root.trim_end_matches('/');

    while let Ok(cmd) = cmd_rx.recv() {
        match cmd {
            SftpCommand::Read { path, reply } => {
                let full = format!("{root}/{path}");
                let result = sftp.read(&full).await.map_err(sftp_err);
                let _ = reply.send(result);
            }
            SftpCommand::Exists { path, reply } => {
                let full = format!("{root}/{path}");
                let result = sftp.metadata(&full).await;
                let _ = reply.send(Ok(result.is_ok()));
            }
            SftpCommand::ListDir { path, reply } => {
                let full = if path.is_empty() {
                    root.to_owned()
                } else {
                    format!("{root}/{path}")
                };
                let result = async {
                    let entries = sftp.read_dir(&full).await.map_err(sftp_err)?;
                    let mut names: Vec<String> = entries
                        .into_iter()
                        .filter_map(|entry| {
                            let name = entry.file_name();
                            if name == "." || name == ".." {
                                None
                            } else {
                                Some(name)
                            }
                        })
                        .collect();
                    names.sort();
                    Ok(names)
                }
                .await;
                let _ = reply.send(result);
            }
            SftpCommand::Write { path, data, reply } => {
                let full = format!("{root}/{path}");
                let result = sftp.write(&full, &data).await.map_err(sftp_err);
                let _ = reply.send(result);
            }
            SftpCommand::Delete { path, reply } => {
                let full = format!("{root}/{path}");
                let result = sftp.remove_file(&full).await.map_err(sftp_err);
                let _ = reply.send(result);
            }
            SftpCommand::CreateDir { path, reply } => {
                let full = format!("{root}/{path}");
                let result = sftp.create_dir(&full).await.map_err(sftp_err);
                let _ = reply.send(result);
            }
            SftpCommand::Shutdown => break,
        }
    }

    let _ = sftp.close().await;
    log::info!("SFTP worker shut down");
}
