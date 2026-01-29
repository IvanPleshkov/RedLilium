//! Render Graph System
//!
//! A declarative system for defining render passes as a directed acyclic graph (DAG).
//! The graph handles resource allocation, pass ordering, and execution.

pub mod executor;
pub mod graph;
pub mod pass;
pub mod resource;

pub use executor::*;
pub use graph::*;
pub use pass::*;
pub use resource::*;
