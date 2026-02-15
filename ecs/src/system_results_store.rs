use std::any::{Any, TypeId};
use std::collections::HashMap;
use std::sync::OnceLock;

/// Storage for system results during a single runner execution.
///
/// Each system has one slot (indexed by its position in the container).
/// Slots use [`OnceLock`] for thread-safe write-once/read-many semantics.
/// The dependency graph guarantees that a result is written before any
/// system that reads it starts executing.
pub(crate) struct SystemResultsStore {
    slots: Vec<OnceLock<Box<dyn Any + Send + Sync>>>,
    type_to_idx: HashMap<TypeId, usize>,
}

impl SystemResultsStore {
    /// Creates a new store with the given number of slots and type-to-index mapping.
    pub(crate) fn new(count: usize, type_to_idx: HashMap<TypeId, usize>) -> Self {
        Self {
            slots: (0..count).map(|_| OnceLock::new()).collect(),
            type_to_idx,
        }
    }

    /// Stores a type-erased result for the system at the given index.
    ///
    /// # Panics
    ///
    /// Panics if the slot has already been written (double execution).
    pub(crate) fn store(&self, idx: usize, result: Box<dyn Any + Send + Sync>) {
        if self.slots[idx].set(result).is_err() {
            panic!("system result slot already written");
        }
    }

    /// Returns the result for a system identified by its `TypeId`, downcast to `&T`.
    ///
    /// Returns `None` if the system has no stored result yet.
    ///
    /// # Panics
    ///
    /// Panics if the stored type does not match `T`.
    pub(crate) fn get<T: 'static>(&self, system_type_id: TypeId) -> Option<&T> {
        let &idx = self.type_to_idx.get(&system_type_id)?;
        let boxed = self.slots[idx].get()?;
        Some(
            boxed
                .downcast_ref::<T>()
                .expect("system result type mismatch"),
        )
    }

    /// Returns a type-erased reference to the result at the given index.
    ///
    /// Used by condition checking to evaluate results without knowing
    /// the concrete type (the stored function pointer handles downcasting).
    pub(crate) fn get_raw(&self, idx: usize) -> Option<&(dyn Any + Send + Sync)> {
        self.slots[idx].get().map(|boxed| &**boxed)
    }

    /// Consumes the store and returns the results as a flat vec of optional
    /// boxed values. Used by runners to extract previous-tick results for
    /// [`System::reuse_result`](crate::System::reuse_result).
    pub(crate) fn into_prev_results(self) -> Vec<Option<Box<dyn Any + Send + Sync>>> {
        self.slots.into_iter().map(OnceLock::into_inner).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct FakeSystemA;
    struct FakeSystemB;

    #[test]
    fn store_and_retrieve() {
        let mut map = HashMap::new();
        map.insert(TypeId::of::<FakeSystemA>(), 0);
        map.insert(TypeId::of::<FakeSystemB>(), 1);

        let store = SystemResultsStore::new(2, map);
        store.store(0, Box::new(42u32));
        store.store(1, Box::new("hello".to_string()));

        let a: &u32 = store.get(TypeId::of::<FakeSystemA>()).unwrap();
        assert_eq!(*a, 42);

        let b: &String = store.get(TypeId::of::<FakeSystemB>()).unwrap();
        assert_eq!(b, "hello");
    }

    #[test]
    fn get_before_store_returns_none() {
        let mut map = HashMap::new();
        map.insert(TypeId::of::<FakeSystemA>(), 0);

        let store = SystemResultsStore::new(1, map);
        let result: Option<&u32> = store.get(TypeId::of::<FakeSystemA>());
        assert!(result.is_none());
    }

    #[test]
    fn get_unknown_type_returns_none() {
        let store = SystemResultsStore::new(0, HashMap::new());
        let result: Option<&u32> = store.get(TypeId::of::<FakeSystemA>());
        assert!(result.is_none());
    }
}
