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
//! ## Scheduling
//!
//! - [`Schedule`] — System registration, dependency resolution, and execution
//! - [`ThreadPool`] — Scoped thread pool for parallel system execution
//! - [`Access`] — Component/resource access descriptors for conflict detection
//! - [`System`] — Trait for system functions
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
mod compute;
mod entity;
mod priority;
mod query;
mod resource;
mod schedule;
mod sparse_set;
mod system;
pub mod thread_pool;
mod world;
mod yield_now;

pub use access::Access;
pub use compute::{ComputePool, TaskHandle};
pub use entity::Entity;
pub use priority::Priority;
pub use query::{ContainsChecker, Read, With, Without, Write};
pub use resource::{ResourceRef, ResourceRefMut};
pub use schedule::Schedule;
pub use sparse_set::{Ref, RefMut, SparseSetInner};
pub use system::System;
pub use thread_pool::ThreadPool;
pub use world::World;
pub use yield_now::yield_now;
