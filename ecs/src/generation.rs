//! Game generation tracking and reload lifecycle.
//!
//! # Overview
//!
//! A [`GameGenerationRegistry`] tracks each game cdylib reload as a distinct generation,
//! identified by a [`SourceId`](crate::SourceId). This module enforces the structural
//! invariant that a generation cannot be unloaded while any world still has active
//! registrations under that generation — the first line of defense against UB from
//! stale fn pointers and drop glue (see ADR-020 amendment, #72 Fable review R1).
//!
//! # Lifecycle
//!
//! 1. **Load**: Allocate a new generation ID via `next_generation()`
//! 2. **Register**: Game code registers types under that generation
//! 3. **Unload guard**: At unload time, check `registration_count(generation_id) == 0`
//!    - If count > 0: refuse to unload (error), preventing dlclose while types are live
//!    - If count == 0: proceed, dylib is safe to unmap
//! 4. **Snapshot restore**: Existing component/resource data survives, re-stamped under new gen

use std::any::TypeId;
use std::collections::HashMap;

use crate::SourceId;

/// Tracks active generations and their registration counts.
///
/// This is a *shared* resource (not per-world) because the invariant applies
/// globally: a generation cannot be unloaded if *any* world has registrations
/// under it.
#[derive(Debug)]
pub struct GameGenerationRegistry {
    /// Monotonic generation counter; starts at 1 (gen-0 is SourceId::HOST).
    next_generation: u32,

    /// Per-generation metadata.
    generations: HashMap<u32, GenerationInfo>,

    /// TypeId → generation owner (which generation registered this type).
    type_to_generation: HashMap<TypeId, u32>,

    /// How many registrations exist for each generation.
    /// Incremented when a type registers under a generation.
    /// Decremented when a world containing types from that generation is dropped.
    registration_counts: HashMap<u32, usize>,
}

/// Metadata for a single generation (one game cdylib load).
#[derive(Debug, Clone)]
pub struct GenerationInfo {
    /// The generation ID.
    pub generation: u32,
    /// Types registered by this generation.
    pub types: Vec<TypeId>,
}

impl GameGenerationRegistry {
    /// Create a new registry. Ready to allocate generations starting at 1.
    pub fn new() -> Self {
        Self {
            next_generation: 1,
            generations: HashMap::new(),
            type_to_generation: HashMap::new(),
            registration_counts: HashMap::new(),
        }
    }

    /// Allocate the next generation ID for a game cdylib load.
    pub fn allocate_generation(&mut self) -> SourceId {
        let generation_id = self.next_generation;
        self.next_generation = generation_id.saturating_add(1);

        self.generations.insert(
            generation_id,
            GenerationInfo {
                generation: generation_id,
                types: Vec::new(),
            },
        );
        self.registration_counts.insert(generation_id, 0);

        SourceId(generation_id)
    }

    /// Record that a type has been registered under a generation.
    ///
    /// # Panics
    ///
    /// If `type_id` is already registered under a different generation (type-ID collision).
    pub fn record_type_registration(&mut self, type_id: TypeId, generation_id: u32) {
        if let Some(&existing_gen) = self.type_to_generation.get(&type_id) {
            if existing_gen != generation_id {
                panic!(
                    "Type registration conflict: TypeId {type_id:?} already registered \
                     under generation {existing_gen}, refusing generation {generation_id}"
                );
            }
            // Same generation: re-registration is idempotent, no change.
            return;
        }

        // New registration.
        self.type_to_generation.insert(type_id, generation_id);

        if let Some(info) = self.generations.get_mut(&generation_id) {
            info.types.push(type_id);
        } else {
            panic!(
                "Attempt to register type under unknown generation {generation_id}; \
                 did you forget to call allocate_generation()?"
            );
        }

        // Increment the count for this generation.
        *self.registration_counts.entry(generation_id).or_insert(0) += 1;
    }

    /// Check whether a generation is currently active (has registrations).
    pub fn is_generation_active(&self, generation_id: u32) -> bool {
        self.registration_counts
            .get(&generation_id)
            .is_some_and(|&count| count > 0)
    }

    /// Get the registration count for a generation.
    pub fn registration_count(&self, generation_id: u32) -> usize {
        self.registration_counts
            .get(&generation_id)
            .copied()
            .unwrap_or(0)
    }

    /// Decrement registration count when a world drops a type under this generation.
    ///
    /// Called during world teardown to track that one fewer world holds registrations
    /// from this generation.
    #[allow(dead_code)]
    pub(crate) fn decrement_registration_count(&mut self, generation_id: u32) {
        if let Some(count) = self.registration_counts.get_mut(&generation_id) {
            *count = count.saturating_sub(1);
        }
    }

    /// Attempt to unload a generation.
    ///
    /// # Returns
    ///
    /// - `Ok(())` if the generation can be safely unloaded (no active registrations).
    /// - `Err(GenerationInUseError)` if types from this generation are still registered.
    pub fn unload_generation(&self, generation_id: u32) -> Result<(), GenerationInUseError> {
        let count = self.registration_count(generation_id);
        if count > 0 {
            return Err(GenerationInUseError {
                generation: generation_id,
                registration_count: count,
            });
        }
        Ok(())
    }

    /// For testing: get all active generation IDs.
    #[cfg(test)]
    pub fn active_generations(&self) -> Vec<u32> {
        self.generations
            .keys()
            .filter(|&&generation_id| self.is_generation_active(generation_id))
            .copied()
            .collect()
    }
}

impl Default for GameGenerationRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Error returned when attempting to unload a generation that still has active registrations.
#[derive(Debug, Clone)]
pub struct GenerationInUseError {
    /// The generation ID that was attempted to be unloaded.
    pub generation: u32,
    /// How many registrations are still active.
    pub registration_count: usize,
}

impl std::fmt::Display for GenerationInUseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Cannot unload generation {} while {} registrations are still active",
            self.generation, self.registration_count
        )
    }
}

impl std::error::Error for GenerationInUseError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_registry_starts_at_gen_1() {
        let mut registry = GameGenerationRegistry::new();
        let generation_id = registry.allocate_generation();
        assert_eq!(generation_id, SourceId(1));
    }

    #[test]
    fn allocate_generation_increments_counter() {
        let mut registry = GameGenerationRegistry::new();
        let gen1 = registry.allocate_generation();
        let gen2 = registry.allocate_generation();
        assert_eq!(gen1, SourceId(1));
        assert_eq!(gen2, SourceId(2));
    }

    #[test]
    fn record_type_registration_increments_count() {
        let mut registry = GameGenerationRegistry::new();
        let _generation = registry.allocate_generation();

        let type_id = TypeId::of::<i32>();
        registry.record_type_registration(type_id, 1);

        assert_eq!(registry.registration_count(1), 1);
        assert!(registry.is_generation_active(1));
    }

    #[test]
    fn multiple_types_same_generation() {
        let mut registry = GameGenerationRegistry::new();
        let _generation = registry.allocate_generation();

        registry.record_type_registration(TypeId::of::<i32>(), 1);
        registry.record_type_registration(TypeId::of::<String>(), 1);

        assert_eq!(registry.registration_count(1), 2);
    }

    #[test]
    #[should_panic(expected = "Type registration conflict")]
    fn conflict_different_generation_panics() {
        let mut registry = GameGenerationRegistry::new();
        let _gen1 = registry.allocate_generation();
        let _gen2 = registry.allocate_generation();

        let type_id = TypeId::of::<i32>();
        registry.record_type_registration(type_id, 1);
        // Same TypeId under different generation → panic
        registry.record_type_registration(type_id, 2);
    }

    #[test]
    fn same_generation_reregistration_is_idempotent() {
        let mut registry = GameGenerationRegistry::new();
        registry.allocate_generation();

        let type_id = TypeId::of::<i32>();
        registry.record_type_registration(type_id, 1);
        registry.record_type_registration(type_id, 1); // no panic

        assert_eq!(registry.registration_count(1), 1);
    }

    #[test]
    fn unload_succeeds_when_count_zero() {
        let mut registry = GameGenerationRegistry::new();
        registry.allocate_generation();

        let result = registry.unload_generation(1);
        assert!(result.is_ok());
    }

    #[test]
    fn unload_fails_when_count_nonzero() {
        let mut registry = GameGenerationRegistry::new();
        registry.allocate_generation();

        let type_id = TypeId::of::<i32>();
        registry.record_type_registration(type_id, 1);

        let result = registry.unload_generation(1);
        assert!(result.is_err());
        if let Err(e) = result {
            assert_eq!(e.generation, 1);
            assert_eq!(e.registration_count, 1);
        }
    }

    #[test]
    fn decrement_registration_count() {
        let mut registry = GameGenerationRegistry::new();
        registry.allocate_generation();

        registry.record_type_registration(TypeId::of::<i32>(), 1);
        registry.record_type_registration(TypeId::of::<String>(), 1);
        assert_eq!(registry.registration_count(1), 2);

        registry.decrement_registration_count(1);
        assert_eq!(registry.registration_count(1), 1);

        registry.decrement_registration_count(1);
        assert_eq!(registry.registration_count(1), 0);
        assert!(!registry.is_generation_active(1));
    }

    #[test]
    fn can_unload_after_decrement_to_zero() {
        let mut registry = GameGenerationRegistry::new();
        registry.allocate_generation();

        registry.record_type_registration(TypeId::of::<i32>(), 1);
        registry.unload_generation(1).unwrap_err(); // fails

        registry.decrement_registration_count(1);
        assert!(registry.unload_generation(1).is_ok()); // succeeds
    }
}
