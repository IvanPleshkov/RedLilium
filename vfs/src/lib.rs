//! Virtual file system abstraction for the RedLilium engine.
//!
//! Provides a unified API for reading and writing assets from multiple storage
//! backends through the [`VfsProvider`] trait and the [`Vfs`] router.
//!
//! # Architecture
//!
//! The VFS returns boxed futures (`Pin<Box<dyn Future + Send>>`) from all
//! operations. These futures are not self-driving — they must be spawned on
//! an async runtime. In the RedLilium engine, use `IoRunner::run()`:
//!
//! ```ignore
//! // From an ECS compute task:
//! let bytes = ctx.io().run(vfs.read("assets/textures/brick.png")).await;
//!
//! // From blocking startup code:
//! let bytes = io.run(vfs.read("config.json")).recv().unwrap();
//! ```
//!
//! # Providers
//!
//! - [`MemoryProvider`] — In-memory storage for tests and embedded assets (read-write)
//! - [`FileSystemProvider`] — Native filesystem access (read-write, native only)
//! - [`SftpProvider`] — Remote SSH/SFTP access (read-write, requires `sftp` feature)
//!
//! Custom providers can implement the [`VfsProvider`] trait for packed archives,
//! HTTP fetch, or other storage backends.
//!
//! # Read-Only vs Read-Write
//!
//! All providers must implement read operations. Write operations are optional
//! and default to returning [`VfsError::ReadOnly`]. Use
//! [`VfsProvider::is_read_only()`] to check capability.

mod error;
#[cfg(all(feature = "filesystem", not(target_arch = "wasm32")))]
mod filesystem;
mod memory;
pub mod path;
mod poll;
mod provider;
#[cfg(all(feature = "sftp", not(target_arch = "wasm32")))]
mod sftp;
mod vfs;

pub use error::VfsError;
#[cfg(all(feature = "filesystem", not(target_arch = "wasm32")))]
pub use filesystem::FileSystemProvider;
pub use memory::MemoryProvider;
pub use poll::poll_now;
pub use provider::{VfsFuture, VfsProvider};
#[cfg(all(feature = "sftp", not(target_arch = "wasm32")))]
pub use sftp::{SftpConfig, SftpProvider};
pub use vfs::Vfs;
