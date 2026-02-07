//! Object pooling utilities for allocation reuse.
//!
//! This module provides [`Pooled<T>`], a container that preserves allocations
//! when a value is no longer actively used. Instead of dropping the value
//! (which deallocates), it transitions to a "pooled" state where the value
//! is cleared but its underlying memory (e.g., `Vec` capacity) is retained.
//!
//! # Motivation
//!
//! In frame-based rendering, structures like compiled graphs are rebuilt every
//! frame. Using `Option<T>` means setting `None` deallocates, and the next
//! frame must reallocate. `Pooled<T>` avoids this by keeping the allocation
//! alive across frames.
//!
//! # Example
//!
//! ```
//! use redlilium_core::pool::{Poolable, Pooled};
//!
//! #[derive(Debug, Default)]
//! struct Buffer {
//!     data: Vec<u8>,
//! }
//!
//! impl Poolable for Buffer {
//!     fn new_empty() -> Self {
//!         Self::default()
//!     }
//!     fn reset(&mut self) {
//!         self.data.clear();
//!     }
//! }
//!
//! let mut pooled = Pooled::<Buffer>::default(); // starts as Pooled
//! assert!(pooled.is_pooled());
//!
//! // Activate to fill in data
//! let buf = pooled.activate();
//! buf.data.extend_from_slice(&[1, 2, 3]);
//! assert!(pooled.is_active());
//!
//! // Release back to pool — clears data but keeps Vec capacity
//! pooled.release();
//! assert!(pooled.is_pooled());
//! assert!(pooled.inner().data.capacity() >= 3);
//! ```

/// Trait for types that can be pooled and reused.
///
/// Implementors must be able to create an empty instance and clear their
/// contents while preserving allocated capacity.
pub trait Poolable {
    /// Create a new empty instance for pool initialization.
    fn new_empty() -> Self;

    /// Reset the value to an empty state, preserving allocated capacity.
    ///
    /// For example, call `Vec::clear()` rather than replacing with a new `Vec`.
    fn reset(&mut self);
}

/// A container that preserves allocations across active/pooled transitions.
///
/// `Pooled<T>` is an enum with two states:
/// - [`Active`](Pooled::Active) — the value contains valid data and is in use
/// - [`Pooled`](Pooled::Pooled) — the value is cleared but its allocation is preserved
///
/// This avoids the deallocation that would occur with `Option<T>` when
/// setting the value to `None`.
#[derive(Debug)]
pub enum Pooled<T: Poolable> {
    /// The value is active and contains valid data.
    Active(T),
    /// The value is cleared but its allocation is preserved for reuse.
    Pooled(T),
}

impl<T: Poolable> Pooled<T> {
    /// Create a new `Pooled` in active state with the given value.
    pub fn new(value: T) -> Self {
        Self::Active(value)
    }

    /// Check if the value is active (contains valid data).
    pub fn is_active(&self) -> bool {
        matches!(self, Self::Active(_))
    }

    /// Check if the value is pooled (cleared, available for reuse).
    pub fn is_pooled(&self) -> bool {
        matches!(self, Self::Pooled(_))
    }

    /// Get a reference to the active value.
    ///
    /// Returns `None` if the value is pooled.
    pub fn get(&self) -> Option<&T> {
        match self {
            Self::Active(t) => Some(t),
            Self::Pooled(_) => None,
        }
    }

    /// Get a mutable reference to the active value.
    ///
    /// Returns `None` if the value is pooled.
    pub fn get_mut(&mut self) -> Option<&mut T> {
        match self {
            Self::Active(t) => Some(t),
            Self::Pooled(_) => None,
        }
    }

    /// Release the value back to the pool.
    ///
    /// Clears the value but preserves its allocation. If already pooled,
    /// this is a no-op.
    pub fn release(&mut self) {
        if matches!(self, Self::Active(_)) {
            // Use mem::replace to safely move between variants.
            // The temporary T::new_empty() is zero-cost for types like Vec.
            let taken = std::mem::replace(self, Self::Pooled(T::new_empty()));
            if let Self::Active(mut t) = taken {
                t.reset();
                *self = Self::Pooled(t);
            }
        }
    }

    /// Activate the pooled value for reuse.
    ///
    /// Transitions from pooled to active state and returns a mutable reference
    /// to the (cleared) value for the caller to fill in.
    ///
    /// If already active, returns a mutable reference to the existing value.
    pub fn activate(&mut self) -> &mut T {
        if matches!(self, Self::Pooled(_)) {
            let taken = std::mem::replace(self, Self::Active(T::new_empty()));
            if let Self::Pooled(t) = taken {
                *self = Self::Active(t);
            }
        }
        match self {
            Self::Active(t) => t,
            _ => unreachable!(),
        }
    }

    /// Get a reference to the inner value regardless of state.
    pub fn inner(&self) -> &T {
        match self {
            Self::Active(t) | Self::Pooled(t) => t,
        }
    }

    /// Get a mutable reference to the inner value regardless of state.
    pub fn inner_mut(&mut self) -> &mut T {
        match self {
            Self::Active(t) | Self::Pooled(t) => t,
        }
    }
}

impl<T: Poolable> Default for Pooled<T> {
    fn default() -> Self {
        Self::Pooled(T::new_empty())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Default, PartialEq)]
    struct TestBuffer {
        data: Vec<u32>,
    }

    impl Poolable for TestBuffer {
        fn new_empty() -> Self {
            Self::default()
        }
        fn reset(&mut self) {
            self.data.clear();
        }
    }

    #[test]
    fn test_default_is_pooled() {
        let pooled = Pooled::<TestBuffer>::default();
        assert!(pooled.is_pooled());
        assert!(!pooled.is_active());
        assert!(pooled.get().is_none());
    }

    #[test]
    fn test_new_is_active() {
        let pooled = Pooled::new(TestBuffer {
            data: vec![1, 2, 3],
        });
        assert!(pooled.is_active());
        assert!(!pooled.is_pooled());
        assert_eq!(pooled.get().unwrap().data, vec![1, 2, 3]);
    }

    #[test]
    fn test_release_clears_and_preserves_capacity() {
        let mut pooled = Pooled::new(TestBuffer {
            data: vec![1, 2, 3, 4, 5],
        });

        pooled.release();

        assert!(pooled.is_pooled());
        assert!(pooled.get().is_none());

        // Allocation preserved
        let inner = pooled.inner();
        assert!(inner.data.is_empty());
        assert!(inner.data.capacity() >= 5);
    }

    #[test]
    fn test_release_on_pooled_is_noop() {
        let mut pooled = Pooled::<TestBuffer>::default();
        pooled.release(); // should not panic
        assert!(pooled.is_pooled());
    }

    #[test]
    fn test_activate_reuses_allocation() {
        let mut pooled = Pooled::new(TestBuffer {
            data: vec![1, 2, 3, 4, 5],
        });
        pooled.release();
        let capacity_after_release = pooled.inner().data.capacity();

        let buf = pooled.activate();
        assert!(buf.data.is_empty());
        assert_eq!(buf.data.capacity(), capacity_after_release);

        buf.data.push(42);

        assert!(pooled.is_active());
        assert_eq!(pooled.get().unwrap().data, vec![42]);
    }

    #[test]
    fn test_activate_on_active_returns_existing() {
        let mut pooled = Pooled::new(TestBuffer {
            data: vec![1, 2, 3],
        });

        let buf = pooled.activate();
        assert_eq!(buf.data, vec![1, 2, 3]);
    }

    #[test]
    fn test_get_mut() {
        let mut pooled = Pooled::new(TestBuffer {
            data: vec![1, 2, 3],
        });

        pooled.get_mut().unwrap().data.push(4);
        assert_eq!(pooled.get().unwrap().data, vec![1, 2, 3, 4]);
    }

    #[test]
    fn test_get_mut_on_pooled_returns_none() {
        let mut pooled = Pooled::<TestBuffer>::default();
        assert!(pooled.get_mut().is_none());
    }

    #[test]
    fn test_inner_always_accessible() {
        let pooled = Pooled::new(TestBuffer {
            data: vec![1, 2, 3],
        });
        assert_eq!(pooled.inner().data, vec![1, 2, 3]);

        let pooled = Pooled::<TestBuffer>::default();
        assert!(pooled.inner().data.is_empty());
    }

    #[test]
    fn test_inner_mut_always_accessible() {
        let mut pooled = Pooled::<TestBuffer>::default();
        pooled.inner_mut().data.push(42);
        // Note: state is still Pooled even though we modified the inner value
        assert!(pooled.is_pooled());
    }

    #[test]
    fn test_round_trip() {
        let mut pooled = Pooled::<TestBuffer>::default();

        // Activate, fill, release — repeat
        for i in 0..3 {
            let buf = pooled.activate();
            for j in 0..10 {
                buf.data.push(i * 10 + j);
            }
            assert!(pooled.is_active());
            assert_eq!(pooled.get().unwrap().data.len(), 10);

            pooled.release();
            assert!(pooled.is_pooled());
            assert!(pooled.inner().data.capacity() >= 10);
        }
    }
}
