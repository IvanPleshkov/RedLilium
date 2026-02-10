#![allow(refining_impl_trait)]

//! # RedLilium ECS
//!
//! Custom Entity-Component-System with integrated async compute support.
//!
//! ## Core Types
//!
//! - [`Entity`] — Lightweight generational entity identifier
//! - [`World`] — Central ECS container owning entities, components, and resources
//! - [`Ref`] / [`RefMut`] — Borrow-checked access to component storages
//! - [`ContainsChecker`] — Filter for `With<T>` / `Without<T>` queries
//!
//! ## Systems & Scheduling
//!
//! - [`System`] — Async system trait with compile-time borrow safety
//! - [`SystemContext`] — Context for component access, compute, and commands
//! - [`SystemsContainer`] — System registration with dependency tracking
//! - [`EcsRunner`] — Single-threaded or multi-threaded system executor
//!
//! ## Access Types
//!
//! - [`Read`] / [`Write`] — Component access markers for lock tuples
//! - [`OptionalRead`] / [`OptionalWrite`] — Non-panicking component access
//! - [`Res`] / [`ResMut`] — Resource access markers
//!
//! ## Async Compute
//!
//! - [`ComputePool`] — Spawn and manage async background tasks
//! - [`TaskHandle`] — Retrieve results from completed tasks
//! - [`Priority`] — Task priority levels (Critical, High, Low)
//! - [`yield_now`] — Cooperative yielding for async tasks
//!
//! See `DESIGN.md` in this crate for architecture decisions and goals.

mod access_set;
mod command_collector;
mod commands;
pub mod component;
mod compute;
mod entity;
mod events;
mod lock_request;
mod priority;
mod query;
mod resource;
mod runner;
mod runner_multi;
mod runner_single;
mod sparse_set;
pub mod string_table;
mod system;
pub mod system_context;
pub mod system_future;
mod systems_container;
mod world;
mod world_locks;
mod yield_now;

// Core types
pub use commands::CommandBuffer;
pub use component::{Component, FieldInfo, FieldKind};
pub use compute::{ComputePool, TaskHandle};
pub use ecs_macro::Component;
pub use ecs_macro::system;
pub use entity::Entity;
pub use events::{EventUpdateSystem, Events};
pub use priority::Priority;
pub use query::{AddedFilter, ChangedFilter, ContainsChecker, With, Without};
pub use resource::{ResourceRef, ResourceRefMut};
pub use sparse_set::{Ref, RefMut, SparseSetInner};
pub use string_table::{StringId, StringTable};
pub use world::World;
pub use yield_now::yield_now;

// System & scheduling (new API)
pub use access_set::{AccessSet, OptionalRead, OptionalWrite, Read, Res, ResMut, Write};
pub use command_collector::CommandCollector;
pub use lock_request::LockRequest;
pub use runner::EcsRunner;
pub use runner_single::{EcsRunnerSingleThread, ShutdownError};
pub use system::{System, SystemResult, run_system_blocking};
pub use system_context::SystemContext;
pub use systems_container::{CycleError, Edge, SystemsContainer};

#[cfg(not(target_arch = "wasm32"))]
pub use runner_multi::EcsRunnerMultiThread;
