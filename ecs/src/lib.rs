//! # RedLilium ECS
//!
//! Custom Entity-Component-System with integrated async compute support.
//!
//! ## Core Types
//!
//! - [`Entity`] — Lightweight generational entity identifier
//! - [`World`] — Central ECS container owning entities, components, and resources
//! - [`Ref`] / [`RefMut`] — Borrow-checked access to component storages
//! - [`Read`] / [`Write`] — Type aliases for query access
//! - [`ContainsChecker`] — Filter for `With<T>` / `Without<T>` queries
//!
//! ## Systems & Scheduling
//!
//! - [`System`] — Async system trait with compile-time borrow safety
//! - [`QueryAccess`] — Token for scoped component access within systems
//! - [`Schedule`] — System registration, dependency resolution, and execution
//! - [`ThreadPool`] — Scoped thread pool for parallel system execution
//! - [`Access`] — Component/resource access descriptors for conflict detection
//!
//! ## Async Compute
//!
//! - [`ComputePool`] — Spawn and manage async background tasks
//! - [`TaskHandle`] — Retrieve results from completed tasks
//! - [`Priority`] — Task priority levels (Critical, High, Low)
//! - [`yield_now`] — Cooperative yielding for async tasks
//!
//! See `DESIGN.md` in this crate for architecture decisions and goals.

mod access;
mod commands;
pub mod component;
mod compute;
mod entity;
mod events;
mod priority;
mod query;
pub mod query_access;
mod resource;
mod schedule;
mod sparse_set;
pub mod string_table;
mod system;
pub mod system_future;
pub mod thread_pool;
mod world;
mod yield_now;

pub use access::Access;
pub use commands::CommandBuffer;
pub use component::{Component, FieldInfo, FieldKind};
pub use compute::{ComputePool, TaskHandle};
pub use ecs_macro::Component;
pub use ecs_macro::system;
pub use entity::Entity;
pub use events::{EventUpdateSystem, Events};
pub use priority::Priority;
pub use query::{AddedFilter, ChangedFilter, ContainsChecker, Read, With, Without, Write};
pub use query_access::QueryAccess;
pub use resource::{ResourceRef, ResourceRefMut};
pub use schedule::Schedule;
pub use sparse_set::{Ref, RefMut, SparseSetInner};
pub use string_table::{StringId, StringTable};
pub use system::{System, SystemRef, SystemResult, run_system_blocking};
pub use system_future::SystemFuture;
pub use thread_pool::ThreadPool;
pub use world::World;
pub use yield_now::yield_now;
