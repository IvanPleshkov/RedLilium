use std::any::Any;
use std::marker::PhantomData;
use std::ops::{Deref, DerefMut};

use fixedbitset::FixedBitSet;

use crate::entity::Entity;
use crate::world::World;

// ---------------------------------------------------------------------------
// Mut<T> — change-detecting mutable reference
// ---------------------------------------------------------------------------

/// A mutable reference to a component that automatically marks it as changed
/// when [`DerefMut`] is invoked.
///
/// Reading through [`Deref`] does **not** mark the component as changed.
/// This is the ECS equivalent of Bevy's `Mut<T>`.
pub struct Mut<'a, T: 'static> {
    value: &'a mut T,
    ticks_changed: &'a mut u64,
    tick: u64,
}

impl<'a, T: 'static> Mut<'a, T> {
    /// Creates a new change-detecting mutable reference.
    pub(crate) fn new(value: &'a mut T, ticks_changed: &'a mut u64, tick: u64) -> Self {
        Self {
            value,
            ticks_changed,
            tick,
        }
    }

    /// Creates a `Mut<T>` from raw pointers.
    ///
    /// # Safety
    ///
    /// - Both pointers must be valid, aligned, and dereferenceable.
    /// - The caller must have exclusive access to both pointees.
    /// - The pointers must not alias any other live references.
    pub(crate) unsafe fn from_raw(value_ptr: *mut T, tick_ptr: *mut u64, tick: u64) -> Mut<'a, T> {
        unsafe {
            Mut {
                value: &mut *value_ptr,
                ticks_changed: &mut *tick_ptr,
                tick,
            }
        }
    }

    /// Returns a mutable reference **without** marking the component as changed.
    ///
    /// Use this when you need to write to a component but intentionally do not
    /// want to trigger change detection (e.g., resetting a dirty flag).
    pub fn bypass_change_detection(&mut self) -> &mut T {
        self.value
    }

    /// Manually marks this component as changed at the current tick.
    pub fn set_changed(&mut self) {
        *self.ticks_changed = self.tick;
    }
}

impl<T: 'static> Deref for Mut<'_, T> {
    type Target = T;

    fn deref(&self) -> &T {
        self.value
    }
}

impl<T: 'static> DerefMut for Mut<'_, T> {
    fn deref_mut(&mut self) -> &mut T {
        *self.ticks_changed = self.tick;
        self.value
    }
}

// ---------------------------------------------------------------------------
// Type-erased function pointer aliases for ComponentMeta
// ---------------------------------------------------------------------------

/// Type-erased extract: reads a component from the world and returns a boxed bag.
pub(crate) type ExtractFn = fn(&World, Entity) -> Option<Box<dyn crate::prefab::ComponentBag>>;

/// Type-erased serialize: reads a component from the world and serializes it.
pub(crate) type SerializeComponentFn =
    fn(
        &World,
        Entity,
        &mut crate::serialize::SerializeContext<'_>,
    )
        -> Result<Option<crate::serialize::SerializedComponent>, crate::serialize::SerializeError>;

/// Type-erased deserialize: deserializes a component and inserts it on an entity.
pub(crate) type DeserializeComponentFn = fn(
    Entity,
    &crate::serialize::Value,
    &mut crate::serialize::DeserializeContext<'_>,
) -> Result<(), crate::serialize::DeserializeError>;

/// The return type of an inspector's `inspect_fn`.
///
/// `None` means the entity didn't have the component or nothing was edited.
/// `Some(actions)` contains one or more undoable [`EditAction`]s.
pub type InspectResult = Option<Vec<Box<dyn redlilium_core::abstract_editor::EditAction<World>>>>;

// ---------------------------------------------------------------------------
// ComponentMeta — type-erased operations for registered component types
// ---------------------------------------------------------------------------

/// Type-erased metadata and operations for a registered component type.
///
/// Stored inside [`ComponentStorage`] for components registered via
/// [`World::register_inspector`] or [`World::register_inspector_default`].
/// Provides capabilities beyond raw storage: inspection, serialization,
/// cloning, entity reference traversal, and more.
pub(crate) struct ComponentMeta {
    /// The short component name (from `Component::NAME`).
    pub name: &'static str,
    /// Check if an entity has this component.
    pub has_fn: fn(&World, Entity) -> bool,
    /// Render the component's inspector UI with an immutable world reference.
    pub inspect_fn: fn(&World, Entity, &mut egui::Ui) -> InspectResult,
    /// Remove this component from an entity. Returns true if removed.
    pub remove_fn: fn(&mut World, Entity) -> bool,
    /// Insert a default instance on an entity (None if T doesn't impl Default).
    pub insert_default_fn: Option<fn(&mut World, Entity)>,
    /// Collect all entity references from this component on an entity.
    pub collect_entities_fn: fn(&World, Entity, &mut Vec<Entity>),
    /// Remap all entity references in this component on an entity.
    pub remap_entities_fn: fn(&mut World, Entity, &mut dyn FnMut(Entity) -> Entity),
    /// Clone this component from src entity to dst entity. None if T is not Clone.
    pub clone_fn: Option<fn(&mut World, Entity, Entity) -> bool>,
    /// Extract this component into a type-erased bag. None if T is not Clone.
    pub extract_fn: Option<ExtractFn>,
    /// Serialize this component on an entity.
    pub serialize_fn: SerializeComponentFn,
    /// Deserialize and insert this component on an entity.
    pub deserialize_fn: DeserializeComponentFn,
    /// Get the axis-aligned bounding box contributed by this component on an entity.
    pub aabb_fn: fn(&World, Entity) -> Option<redlilium_core::math::Aabb>,
    /// Display order in the inspector panel. Lower values appear first.
    pub display_order: u32,
}

/// Function signature for component lifecycle hooks.
///
/// Hooks receive exclusive world access and the entity being modified.
/// They fire synchronously during structural changes (insert/remove/despawn).
///
/// # Hook types
///
/// | Hook | When | Typical use |
/// |------|------|-------------|
/// | `on_add` | First insertion only | Initialize derived state, required components |
/// | `on_insert` | Every insertion (add + replace) | Sync external systems |
/// | `on_replace` | Before existing value is overwritten | Cleanup old value |
/// | `on_remove` | Before component is removed | Cleanup resources |
pub type ComponentHookFn = fn(&mut crate::world::World, crate::entity::Entity);

/// Function that inserts a required component's default value on an entity
/// if not already present.
///
/// Called after a component with requirements is first added to an entity.
/// Transitivity is handled naturally: inserting a required component triggers
/// its own requirements via the same mechanism in [`World::insert`](crate::World::insert).
pub type RequiredComponentFn = fn(&mut crate::world::World, crate::entity::Entity);

/// Typed sparse set storing components of type T.
///
/// Uses a sparse array (entity index → dense index) and a dense array
/// (contiguous component data + entity mapping) for O(1) insert/remove/get
/// and cache-friendly iteration.
///
/// ## Change Detection
///
/// Each component tracks the tick when it was added and last changed.
/// Use [`insert_with_tick`](SparseSetInner::insert_with_tick) and
/// [`get_mut_tracked`](SparseSetInner::get_mut_tracked) to enable tracking.
/// Query with [`changed_since`](SparseSetInner::changed_since) and
/// [`added_since`](SparseSetInner::added_since).
pub struct SparseSetInner<T: 'static> {
    /// Sparse array: `entity_index -> dense_index`. `None` means the entity
    /// does not have this component.
    sparse: Vec<Option<u32>>,
    /// Dense array of component values (contiguous for iteration).
    dense: Vec<T>,
    /// Entity indices corresponding to each dense element.
    entities: Vec<u32>,
    /// Tick when each component was added (parallel to dense).
    ticks_added: Vec<u64>,
    /// Tick when each component was last changed (parallel to dense).
    ticks_changed: Vec<u64>,
    /// Bitset tracking which entity indices have this component.
    /// Bit N is set iff entity index N has this component stored.
    membership: FixedBitSet,
}

impl<T: 'static> SparseSetInner<T> {
    /// Creates a new empty sparse set.
    pub fn new() -> Self {
        Self {
            sparse: Vec::new(),
            dense: Vec::new(),
            entities: Vec::new(),
            ticks_added: Vec::new(),
            ticks_changed: Vec::new(),
            membership: FixedBitSet::new(),
        }
    }

    /// Inserts a component for the given entity index.
    /// If the entity already has this component, the value is replaced.
    /// Uses tick 0 for change tracking (untracked).
    pub fn insert(&mut self, entity_index: u32, value: T) {
        self.insert_with_tick(entity_index, value, 0);
    }

    /// Inserts a component with change tracking at the given tick.
    ///
    /// If the entity already has this component, the value is replaced
    /// and `ticks_changed` is updated. If it's a new insertion,
    /// both `ticks_added` and `ticks_changed` are set to `tick`.
    pub fn insert_with_tick(&mut self, entity_index: u32, value: T, tick: u64) {
        let idx = entity_index as usize;

        // Grow sparse array if needed
        if idx >= self.sparse.len() {
            self.sparse.resize(idx + 1, None);
        }

        if let Some(dense_idx) = self.sparse[idx] {
            // Replace existing value
            let di = dense_idx as usize;
            self.dense[di] = value;
            self.ticks_changed[di] = tick;
        } else {
            // Insert new value
            let dense_idx = self.dense.len() as u32;
            self.sparse[idx] = Some(dense_idx);
            self.dense.push(value);
            self.entities.push(entity_index);
            self.ticks_added.push(tick);
            self.ticks_changed.push(tick);
            // Track membership
            if idx >= self.membership.len() {
                self.membership.grow(idx + 1);
            }
            self.membership.insert(idx);
        }
    }

    /// Removes a component for the given entity index.
    /// Returns the removed value, or `None` if the entity did not have this component.
    pub fn remove(&mut self, entity_index: u32) -> Option<T> {
        let idx = entity_index as usize;
        if idx >= self.sparse.len() {
            return None;
        }

        let dense_idx = self.sparse[idx]?;
        self.sparse[idx] = None;
        self.membership.set(idx, false);

        let last_dense = self.dense.len() - 1;
        let dense_idx = dense_idx as usize;

        if dense_idx != last_dense {
            // Swap-remove: move last element into the removed slot
            let swapped_entity = self.entities[last_dense];
            self.sparse[swapped_entity as usize] = Some(dense_idx as u32);
            self.entities[dense_idx] = swapped_entity;
            self.ticks_added[dense_idx] = self.ticks_added[last_dense];
            self.ticks_changed[dense_idx] = self.ticks_changed[last_dense];
        }

        self.entities.pop();
        self.ticks_added.pop();
        self.ticks_changed.pop();
        Some(self.dense.swap_remove(dense_idx))
    }

    /// Returns a reference to the component for the given entity index.
    pub fn get(&self, entity_index: u32) -> Option<&T> {
        let idx = entity_index as usize;
        let dense_idx = *self.sparse.get(idx)?.as_ref()? as usize;
        Some(&self.dense[dense_idx])
    }

    /// Returns a mutable reference to the component for the given entity index.
    pub fn get_mut(&mut self, entity_index: u32) -> Option<&mut T> {
        let idx = entity_index as usize;
        let dense_idx = *self.sparse.get(idx)?.as_ref()? as usize;
        Some(&mut self.dense[dense_idx])
    }

    /// Returns whether the entity has this component.
    pub fn contains(&self, entity_index: u32) -> bool {
        let idx = entity_index as usize;
        idx < self.sparse.len() && self.sparse[idx].is_some()
    }

    /// Returns the number of components stored.
    pub fn len(&self) -> usize {
        self.dense.len()
    }

    /// Returns whether this sparse set is empty.
    pub fn is_empty(&self) -> bool {
        self.dense.is_empty()
    }

    /// Reserves capacity for at least `additional` more components.
    ///
    /// Pre-grows dense arrays to avoid repeated allocations during
    /// batch insertions.
    pub fn reserve(&mut self, additional: usize) {
        self.dense.reserve(additional);
        self.entities.reserve(additional);
        self.ticks_added.reserve(additional);
        self.ticks_changed.reserve(additional);
    }

    /// Iterates over `(entity_index, &component)` pairs in dense order.
    pub fn iter(&self) -> impl Iterator<Item = (u32, &T)> {
        self.entities.iter().copied().zip(self.dense.iter())
    }

    /// Iterates over `(entity_index, &mut component)` pairs in dense order.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (u32, &mut T)> {
        self.entities.iter().copied().zip(self.dense.iter_mut())
    }

    /// Returns a slice of entity indices in dense order.
    pub fn entities(&self) -> &[u32] {
        &self.entities
    }

    /// Returns the membership bitset tracking which entity indices have this component.
    pub fn membership(&self) -> &FixedBitSet {
        &self.membership
    }

    /// Returns mutable pointers to both the component value and its
    /// `ticks_changed` slot for the given entity index.
    ///
    /// # Safety
    ///
    /// Same requirements as [`get_ptr_mut`](SparseSetInner::get_ptr_mut).
    pub(crate) unsafe fn get_ptr_mut_with_tick(
        this: *mut Self,
        entity_index: u32,
    ) -> Option<(*mut T, *mut u64)> {
        unsafe {
            let set = &mut *this;
            let idx = entity_index as usize;
            let dense_idx = *set.sparse.get(idx)?.as_ref()? as usize;
            Some((
                set.dense.as_mut_ptr().add(dense_idx),
                set.ticks_changed.as_mut_ptr().add(dense_idx),
            ))
        }
    }

    /// Returns a [`Mut`] wrapper for the component at `entity_index`.
    ///
    /// The component is marked as changed only when [`DerefMut`] is invoked
    /// on the returned `Mut<T>`.
    pub fn get_mut_tracked(&mut self, entity_index: u32, tick: u64) -> Option<Mut<'_, T>> {
        let idx = entity_index as usize;
        let dense_idx = *self.sparse.get(idx)?.as_ref()? as usize;
        Some(Mut::new(
            &mut self.dense[dense_idx],
            &mut self.ticks_changed[dense_idx],
            tick,
        ))
    }

    /// Iterates yielding [`Mut`] wrappers that mark components as changed
    /// only when [`DerefMut`] is invoked.
    pub fn iter_mut_tracked(&mut self, tick: u64) -> impl Iterator<Item = (u32, Mut<'_, T>)> + '_ {
        self.entities.iter().copied().zip(
            self.dense
                .iter_mut()
                .zip(self.ticks_changed.iter_mut())
                .map(move |(val, tc)| Mut::new(val, tc, tick)),
        )
    }

    /// Returns true if the component was changed since (strictly after) `since_tick`.
    pub fn changed_since(&self, entity_index: u32, since_tick: u64) -> bool {
        let idx = entity_index as usize;
        if let Some(Some(dense_idx)) = self.sparse.get(idx) {
            self.ticks_changed[*dense_idx as usize] > since_tick
        } else {
            false
        }
    }

    /// Returns true if the component was added since (strictly after) `since_tick`.
    pub fn added_since(&self, entity_index: u32, since_tick: u64) -> bool {
        let idx = entity_index as usize;
        if let Some(Some(dense_idx)) = self.sparse.get(idx) {
            self.ticks_added[*dense_idx as usize] > since_tick
        } else {
            false
        }
    }
}

impl<T: 'static> Default for SparseSetInner<T> {
    fn default() -> Self {
        Self::new()
    }
}

// ---------------------------------------------------------------------------
// ErasedSparseSet — trait for type-erased sparse set operations
// ---------------------------------------------------------------------------

/// Type-erased interface for sparse set operations.
///
/// Implemented by [`SparseSetInner<T>`] and used as the inner storage of
/// [`ComponentStorage`] via `Box<dyn ErasedSparseSet>`.
pub(crate) trait ErasedSparseSet: Send + Sync {
    /// Returns the human-readable type name of the stored component.
    fn type_name(&self) -> &'static str;

    /// Removes a component by entity index. Returns true if removed.
    fn remove(&mut self, entity_index: u32) -> bool;

    /// Checks if the entity has this component.
    fn contains(&self, entity_index: u32) -> bool;

    /// Checks if the component was changed since (strictly after) `since_tick`.
    fn changed_since(&self, entity_index: u32, since_tick: u64) -> bool;

    /// Checks if the component was added since (strictly after) `since_tick`.
    fn added_since(&self, entity_index: u32, since_tick: u64) -> bool;

    /// Downcast to `&dyn Any` for typed access.
    fn as_any(&self) -> &dyn Any;

    /// Downcast to `&mut dyn Any` for typed access.
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

impl<T: Send + Sync + 'static> ErasedSparseSet for SparseSetInner<T> {
    fn type_name(&self) -> &'static str {
        std::any::type_name::<T>()
    }

    fn remove(&mut self, entity_index: u32) -> bool {
        SparseSetInner::remove(self, entity_index).is_some()
    }

    fn contains(&self, entity_index: u32) -> bool {
        SparseSetInner::contains(self, entity_index)
    }

    fn changed_since(&self, entity_index: u32, since_tick: u64) -> bool {
        SparseSetInner::changed_since(self, entity_index, since_tick)
    }

    fn added_since(&self, entity_index: u32, since_tick: u64) -> bool {
        SparseSetInner::added_since(self, entity_index, since_tick)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

/// A lock guard for either a read or write lock on a component storage.
///
/// The guard is held purely for its RAII drop behavior (releasing the lock).
#[allow(dead_code)]
pub(crate) enum LockGuard<'a> {
    Read(parking_lot::RwLockReadGuard<'a, ComponentStorage>),
    Write(parking_lot::RwLockWriteGuard<'a, ComponentStorage>),
}

/// A type-erased sparse set that stores components of a single type.
///
/// Provides per-storage RwLock synchronization for thread-safe access.
/// Used internally by [`World`](crate::World).
pub(crate) struct ComponentStorage {
    inner: Box<dyn ErasedSparseSet>,
    /// Records of (entity_index, tick) for recently removed components.
    /// Cleared by [`World::clear_removed_tracking`](crate::World::clear_removed_tracking).
    removed_ticks: Vec<(u32, u64)>,
    /// Hook called when a component is added to an entity for the first time.
    pub(crate) on_add: Option<ComponentHookFn>,
    /// Hook called on every insertion (both new addition and replacement).
    pub(crate) on_insert: Option<ComponentHookFn>,
    /// Hook called just before an existing component value is replaced.
    pub(crate) on_replace: Option<ComponentHookFn>,
    /// Hook called just before a component is removed from an entity.
    pub(crate) on_remove: Option<ComponentHookFn>,
    /// Functions that insert required components when this component is first
    /// added to an entity. Each function checks for presence and inserts a
    /// default if absent.
    pub(crate) required_components: Vec<RequiredComponentFn>,
    /// Type-erased component metadata (inspection, serialization, cloning, etc.).
    /// Present only for components registered via `register_inspector` / `register_inspector_default`.
    pub(crate) meta: Option<ComponentMeta>,
}

impl ComponentStorage {
    /// Creates a new component storage for type `T`.
    pub fn new<T: Send + Sync + 'static>() -> Self {
        Self {
            inner: Box::new(SparseSetInner::<T>::new()),
            removed_ticks: Vec::new(),
            on_add: None,
            on_insert: None,
            on_replace: None,
            on_remove: None,
            required_components: Vec::new(),
            meta: None,
        }
    }

    /// Returns the component meta if registered.
    pub(crate) fn meta(&self) -> Option<&ComponentMeta> {
        self.meta.as_ref()
    }

    /// Returns a mutable reference to the component meta if registered.
    pub(crate) fn meta_mut(&mut self) -> Option<&mut ComponentMeta> {
        self.meta.as_mut()
    }

    /// Downcasts to the typed sparse set.
    pub fn typed<T: 'static>(&self) -> &SparseSetInner<T> {
        self.inner
            .as_any()
            .downcast_ref::<SparseSetInner<T>>()
            .unwrap()
    }

    /// Downcasts to the typed sparse set (mutable).
    pub fn typed_mut<T: 'static>(&mut self) -> &mut SparseSetInner<T> {
        self.inner
            .as_any_mut()
            .downcast_mut::<SparseSetInner<T>>()
            .unwrap()
    }

    /// Returns the human-readable type name of the stored component.
    pub(crate) fn type_name(&self) -> &'static str {
        self.inner.type_name()
    }

    /// Removes a component by entity index (type-erased). Returns true if removed.
    pub fn remove_untyped(&mut self, entity_index: u32) -> bool {
        self.inner.remove(entity_index)
    }

    /// Returns true if this component has any required components registered.
    pub fn has_required_components(&self) -> bool {
        !self.required_components.is_empty()
    }

    /// Checks if the entity has this component (type-erased).
    pub fn contains_untyped(&self, entity_index: u32) -> bool {
        self.inner.contains(entity_index)
    }

    /// Checks if the component was changed since `since_tick` (type-erased).
    pub fn changed_since_untyped(&self, entity_index: u32, since_tick: u64) -> bool {
        self.inner.changed_since(entity_index, since_tick)
    }

    /// Checks if the component was added since `since_tick` (type-erased).
    pub fn added_since_untyped(&self, entity_index: u32, since_tick: u64) -> bool {
        self.inner.added_since(entity_index, since_tick)
    }

    /// Records that a component was removed from the given entity at the given tick.
    pub fn record_removal(&mut self, entity_index: u32, tick: u64) {
        self.removed_ticks.push((entity_index, tick));
    }

    /// Checks if a component was removed from the entity since (strictly after) `since_tick`.
    pub fn removed_since_untyped(&self, entity_index: u32, since_tick: u64) -> bool {
        self.removed_ticks
            .iter()
            .any(|&(e, t)| e == entity_index && t > since_tick)
    }

    /// Returns an iterator over `(entity_index, tick)` pairs for all removals
    /// that happened since (strictly after) `since_tick`.
    pub fn removed_entities_since(&self, since_tick: u64) -> impl Iterator<Item = u32> + '_ {
        self.removed_ticks
            .iter()
            .filter(move |&&(_, t)| t > since_tick)
            .map(|&(e, _)| e)
    }

    /// Clears all removal tracking records.
    pub fn clear_removed(&mut self) {
        self.removed_ticks.clear();
    }
}

/// Shared read access to a component storage.
///
/// Automatically releases the lock when dropped.
/// Dereferences to [`SparseSetInner<T>`] for accessing component data.
///
/// Inherent methods (`get`, `iter`, `contains`, etc.) automatically filter
/// out excluded entities (disabled and/or static, depending on the
/// `exclude_mask`). Use the `_unfiltered` variants (e.g. `get_unfiltered`,
/// `iter_unfiltered`) to include all entities.
pub struct Ref<'a, T: 'static> {
    inner: *const SparseSetInner<T>,
    entity_flags: &'a [u32],
    /// Bitmask of entity flag bits that cause an entity to be excluded
    /// from filtered methods. Default: `DISABLED | STATIC`.
    exclude_mask: u32,
    _guard: Option<parking_lot::RwLockReadGuard<'a, ComponentStorage>>,
}

impl<'a, T: 'static> Ref<'a, T> {
    /// Default exclude mask: skip disabled, static, and editor entities.
    const DEFAULT_MASK: u32 = Entity::DISABLED | Entity::STATIC | Entity::EDITOR;

    /// Creates a new shared borrow guard, acquiring the storage's read lock.
    ///
    /// Uses the default exclude mask (`DISABLED | STATIC | EDITOR`).
    pub(crate) fn new(
        lock: &'a parking_lot::RwLock<ComponentStorage>,
        entity_flags: &'a [u32],
    ) -> Self {
        let guard = lock.try_read().unwrap_or_else(|| {
            panic!("Cannot borrow component immutably: already borrowed mutably")
        });
        let inner = guard.typed::<T>() as *const SparseSetInner<T>;
        Self {
            inner,
            entity_flags,
            exclude_mask: Self::DEFAULT_MASK,
            _guard: Some(guard),
        }
    }

    /// Creates a new shared borrow guard with a custom exclude mask.
    ///
    /// Used by `ReadAll<T>` to create a `Ref` that only excludes disabled
    /// entities (mask = `DISABLED`), including static entities in iteration.
    pub(crate) fn new_with_mask(
        lock: &'a parking_lot::RwLock<ComponentStorage>,
        entity_flags: &'a [u32],
        exclude_mask: u32,
    ) -> Self {
        let guard = lock.try_read().unwrap_or_else(|| {
            panic!("Cannot borrow component immutably: already borrowed mutably")
        });
        let inner = guard.typed::<T>() as *const SparseSetInner<T>;
        Self {
            inner,
            entity_flags,
            exclude_mask,
            _guard: Some(guard),
        }
    }

    /// Creates a shared borrow without acquiring a lock.
    ///
    /// The caller must ensure the lock is already held externally
    /// (e.g. via `acquire_sorted`). Uses the default exclude mask.
    pub(crate) fn new_unlocked(storage: &'a ComponentStorage, entity_flags: &'a [u32]) -> Self {
        let inner = storage.typed::<T>() as *const SparseSetInner<T>;
        Self {
            inner,
            entity_flags,
            exclude_mask: Self::DEFAULT_MASK,
            _guard: None,
        }
    }

    /// Creates a shared borrow without acquiring a lock, with a custom exclude mask.
    pub(crate) fn new_unlocked_with_mask(
        storage: &'a ComponentStorage,
        entity_flags: &'a [u32],
        exclude_mask: u32,
    ) -> Self {
        let inner = storage.typed::<T>() as *const SparseSetInner<T>;
        Self {
            inner,
            entity_flags,
            exclude_mask,
            _guard: None,
        }
    }

    /// Returns a reference to the underlying storage with the storage lifetime.
    ///
    /// Unlike `Deref` (which ties the result to the borrow of `Ref`), this
    /// returns a reference with the original `'a` lifetime of the storage.
    ///
    /// # Safety
    ///
    /// The raw pointer is valid for `'a` because either:
    /// - The guard holds the lock (locked path), or
    /// - The caller holds the lock externally (unlocked path).
    pub(crate) fn storage(&self) -> &'a SparseSetInner<T> {
        unsafe { &*self.inner }
    }

    // ---- Filtered methods (shadow Deref'd SparseSetInner methods) ----

    /// Returns whether the entity at the given index is excluded by this
    /// reference's exclude mask (disabled, static, or both).
    pub fn is_entity_excluded(&self, entity_index: u32) -> bool {
        let idx = entity_index as usize;
        idx < self.entity_flags.len() && self.entity_flags[idx] & self.exclude_mask != 0
    }

    /// Returns whether the entity at the given index has the DISABLED flag.
    pub fn is_entity_disabled(&self, entity_index: u32) -> bool {
        let idx = entity_index as usize;
        idx < self.entity_flags.len() && self.entity_flags[idx] & Entity::DISABLED != 0
    }

    /// Returns the entity flags slice reference.
    pub fn entity_flags(&self) -> &'a [u32] {
        self.entity_flags
    }

    /// Returns the exclude mask used by filtered methods.
    pub fn exclude_mask(&self) -> u32 {
        self.exclude_mask
    }

    /// Returns a reference to the inner sparse set.
    ///
    /// # Safety
    ///
    /// The raw pointer is valid because either the guard holds the lock
    /// or the caller holds it externally.
    fn inner(&self) -> &SparseSetInner<T> {
        unsafe { &*self.inner }
    }

    /// Returns a reference to the component for the given entity index.
    /// Returns `None` if the entity is excluded or does not have this component.
    pub fn get(&self, entity_index: u32) -> Option<&T> {
        if self.is_entity_excluded(entity_index) {
            return None;
        }
        self.inner().get(entity_index)
    }

    /// Iterates over `(entity_index, &component)` pairs, skipping excluded entities.
    pub fn iter(&self) -> impl Iterator<Item = (u32, &T)> + '_ {
        self.inner()
            .iter()
            .filter(|(idx, _)| !self.is_entity_excluded(*idx))
    }

    /// Returns whether the entity has this component and is not excluded.
    pub fn contains(&self, entity_index: u32) -> bool {
        !self.is_entity_excluded(entity_index) && self.inner().contains(entity_index)
    }

    /// Returns true if the component was changed since `since_tick` and the entity is not excluded.
    pub fn changed_since(&self, entity_index: u32, since_tick: u64) -> bool {
        !self.is_entity_excluded(entity_index)
            && self.inner().changed_since(entity_index, since_tick)
    }

    /// Returns true if the component was added since `since_tick` and the entity is not excluded.
    pub fn added_since(&self, entity_index: u32, since_tick: u64) -> bool {
        !self.is_entity_excluded(entity_index) && self.inner().added_since(entity_index, since_tick)
    }

    // ---- Unfiltered escape hatches ----

    /// Returns a reference to the component, ignoring disabled status.
    pub fn get_unfiltered(&self, entity_index: u32) -> Option<&T> {
        self.inner().get(entity_index)
    }

    /// Iterates over all `(entity_index, &component)` pairs, including disabled entities.
    pub fn iter_unfiltered(&self) -> impl Iterator<Item = (u32, &T)> + '_ {
        self.inner().iter()
    }

    /// Returns whether the entity has this component, ignoring disabled status.
    pub fn contains_unfiltered(&self, entity_index: u32) -> bool {
        self.inner().contains(entity_index)
    }
}

impl<T: 'static> Deref for Ref<'_, T> {
    type Target = SparseSetInner<T>;

    fn deref(&self) -> &Self::Target {
        unsafe { &*self.inner }
    }
}

// Ref holds either an RwLockReadGuard (auto-released on drop) or nothing.
// No manual Drop needed — the Option<RwLockReadGuard> handles it.

// SAFETY: Ref only provides shared (&) access to the inner data.
// The RwLock guarantees no exclusive access exists when the guard is held.
// When unlocked, the caller guarantees external lock management.
unsafe impl<T: Send + Sync + 'static> Send for Ref<'_, T> {}
unsafe impl<T: Send + Sync + 'static> Sync for Ref<'_, T> {}

/// Exclusive write access to a component storage.
///
/// Automatically releases the lock when dropped.
/// Dereferences to [`SparseSetInner<T>`] for accessing and modifying component data.
///
/// Inherent methods (`get`, `get_mut`, `iter`, `iter_mut`, etc.) automatically
/// filter out excluded entities (disabled and static, via entity flag bits).
/// Use the `_unfiltered` variants to include all entities.
pub struct RefMut<'a, T: 'static> {
    inner: *mut SparseSetInner<T>,
    entity_flags: &'a [u32],
    /// Bitmask of entity flag bits that cause an entity to be excluded.
    /// Default: `DISABLED | STATIC`.
    exclude_mask: u32,
    /// Current world tick, stamped into `ticks_changed` via [`Mut`].
    tick: u64,
    _guard: Option<parking_lot::RwLockWriteGuard<'a, ComponentStorage>>,
    _marker: PhantomData<&'a mut SparseSetInner<T>>,
}

impl<'a, T: 'static> RefMut<'a, T> {
    /// Default exclude mask: skip disabled, static, and editor entities.
    const DEFAULT_MASK: u32 = Entity::DISABLED | Entity::STATIC | Entity::EDITOR;

    /// Creates a new exclusive borrow guard, acquiring the storage's write lock.
    ///
    /// Uses the default exclude mask (`DISABLED | STATIC | EDITOR`).
    pub(crate) fn new(
        lock: &'a parking_lot::RwLock<ComponentStorage>,
        entity_flags: &'a [u32],
        tick: u64,
    ) -> Self {
        let guard = lock
            .try_write()
            .unwrap_or_else(|| panic!("Cannot borrow component mutably: already borrowed"));
        // SAFETY: write lock guarantees exclusive access. data_ptr() provides
        // interior mutability through the lock — no dubious &T→*mut T cast.
        let inner = unsafe { (*lock.data_ptr()).typed_mut::<T>() as *mut SparseSetInner<T> };
        Self {
            inner,
            entity_flags,
            exclude_mask: Self::DEFAULT_MASK,
            tick,
            _guard: Some(guard),
            _marker: PhantomData,
        }
    }

    /// Creates a new exclusive borrow guard with a custom exclude mask.
    ///
    /// Used by `WriteAll<T>` to create a `RefMut` that only excludes disabled
    /// entities (mask = `DISABLED`), including static and editor entities.
    pub(crate) fn new_with_mask(
        lock: &'a parking_lot::RwLock<ComponentStorage>,
        entity_flags: &'a [u32],
        exclude_mask: u32,
        tick: u64,
    ) -> Self {
        let guard = lock
            .try_write()
            .unwrap_or_else(|| panic!("Cannot borrow component mutably: already borrowed"));
        let inner = unsafe { (*lock.data_ptr()).typed_mut::<T>() as *mut SparseSetInner<T> };
        Self {
            inner,
            entity_flags,
            exclude_mask,
            tick,
            _guard: Some(guard),
            _marker: PhantomData,
        }
    }

    /// Creates an exclusive borrow without acquiring a lock.
    ///
    /// # Safety
    ///
    /// The caller must ensure the write lock is already held externally
    /// (e.g. via `acquire_sorted`).
    pub(crate) fn new_unlocked(
        storage_ptr: *mut ComponentStorage,
        entity_flags: &'a [u32],
        tick: u64,
    ) -> Self {
        let inner = unsafe { (*storage_ptr).typed_mut::<T>() as *mut SparseSetInner<T> };
        Self {
            inner,
            entity_flags,
            exclude_mask: Self::DEFAULT_MASK,
            tick,
            _guard: None,
            _marker: PhantomData,
        }
    }

    /// Creates an exclusive borrow without acquiring a lock, with a custom exclude mask.
    pub(crate) fn new_unlocked_with_mask(
        storage_ptr: *mut ComponentStorage,
        entity_flags: &'a [u32],
        exclude_mask: u32,
        tick: u64,
    ) -> Self {
        let inner = unsafe { (*storage_ptr).typed_mut::<T>() as *mut SparseSetInner<T> };
        Self {
            inner,
            entity_flags,
            exclude_mask,
            tick,
            _guard: None,
            _marker: PhantomData,
        }
    }

    /// Returns the raw pointer to the underlying storage.
    ///
    /// Used by [`QueryItem`](crate::QueryItem) to access components without
    /// requiring `&mut self`, enabling per-entity mutable access in iterators.
    pub(crate) fn storage_ptr(&self) -> *mut SparseSetInner<T> {
        self.inner
    }

    /// Returns the tick stored in this guard (for [`QueryItem`](crate::QueryItem)).
    pub(crate) fn query_tick(&self) -> u64 {
        self.tick
    }

    // ---- Filtered methods (shadow Deref'd SparseSetInner methods) ----

    /// Returns whether the entity at the given index is excluded by this
    /// reference's exclude mask (disabled, static, or both).
    pub fn is_entity_excluded(&self, entity_index: u32) -> bool {
        let idx = entity_index as usize;
        idx < self.entity_flags.len() && self.entity_flags[idx] & self.exclude_mask != 0
    }

    /// Returns whether the entity at the given index has the DISABLED flag.
    pub fn is_entity_disabled(&self, entity_index: u32) -> bool {
        let idx = entity_index as usize;
        idx < self.entity_flags.len() && self.entity_flags[idx] & Entity::DISABLED != 0
    }

    /// Returns the entity flags slice reference.
    pub fn entity_flags(&self) -> &'a [u32] {
        self.entity_flags
    }

    /// Returns the exclude mask used by filtered methods.
    pub fn exclude_mask(&self) -> u32 {
        self.exclude_mask
    }

    /// Returns a reference to the component for the given entity index.
    /// Returns `None` if the entity is excluded or does not have this component.
    pub fn get(&self, entity_index: u32) -> Option<&T> {
        if self.is_entity_excluded(entity_index) {
            return None;
        }
        // SAFETY: write lock guarantees exclusive access.
        unsafe { &*self.inner }.get(entity_index)
    }

    /// Returns a [`Mut`] wrapper for the component at the given entity index.
    ///
    /// The component is marked as changed only when [`DerefMut`] is invoked.
    /// Returns `None` if the entity is excluded or does not have this component.
    pub fn get_mut(&mut self, entity_index: u32) -> Option<Mut<'_, T>> {
        if self.is_entity_excluded(entity_index) {
            return None;
        }
        // SAFETY: write lock guarantees exclusive access.
        unsafe { &mut *self.inner }.get_mut_tracked(entity_index, self.tick)
    }

    /// Iterates over `(entity_index, &component)` pairs, skipping excluded entities.
    pub fn iter(&self) -> impl Iterator<Item = (u32, &T)> + '_ {
        // SAFETY: write lock guarantees exclusive access.
        unsafe { &*self.inner }
            .iter()
            .filter(|(idx, _)| !self.is_entity_excluded(*idx))
    }

    /// Iterates over `(entity_index, Mut<T>)` pairs, skipping excluded entities.
    ///
    /// Each yielded [`Mut`] marks the component as changed only when
    /// [`DerefMut`] is invoked.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = (u32, Mut<'_, T>)> + '_ {
        let flags = self.entity_flags;
        let mask = self.exclude_mask;
        let tick = self.tick;
        // SAFETY: write lock guarantees exclusive access.
        unsafe { &mut *self.inner }
            .iter_mut_tracked(tick)
            .filter(move |(idx, _)| {
                let i = *idx as usize;
                i >= flags.len() || flags[i] & mask == 0
            })
    }

    /// Returns whether the entity has this component and is not excluded.
    pub fn contains(&self, entity_index: u32) -> bool {
        !self.is_entity_excluded(entity_index) && unsafe { &*self.inner }.contains(entity_index)
    }

    /// Returns true if the component was changed since `since_tick` and the entity is not excluded.
    pub fn changed_since(&self, entity_index: u32, since_tick: u64) -> bool {
        !self.is_entity_excluded(entity_index)
            && unsafe { &*self.inner }.changed_since(entity_index, since_tick)
    }

    /// Returns true if the component was added since `since_tick` and the entity is not excluded.
    pub fn added_since(&self, entity_index: u32, since_tick: u64) -> bool {
        !self.is_entity_excluded(entity_index)
            && unsafe { &*self.inner }.added_since(entity_index, since_tick)
    }

    // ---- Unfiltered escape hatches ----

    /// Returns a reference to the component, ignoring disabled status.
    pub fn get_unfiltered(&self, entity_index: u32) -> Option<&T> {
        unsafe { &*self.inner }.get(entity_index)
    }

    /// Returns a [`Mut`] wrapper for the component, ignoring disabled status.
    pub fn get_mut_unfiltered(&mut self, entity_index: u32) -> Option<Mut<'_, T>> {
        unsafe { &mut *self.inner }.get_mut_tracked(entity_index, self.tick)
    }

    /// Iterates over all `(entity_index, &component)` pairs, including disabled entities.
    pub fn iter_unfiltered(&self) -> impl Iterator<Item = (u32, &T)> + '_ {
        unsafe { &*self.inner }.iter()
    }

    /// Iterates over all `(entity_index, Mut<T>)` pairs, including disabled entities.
    pub fn iter_mut_unfiltered(&mut self) -> impl Iterator<Item = (u32, Mut<'_, T>)> + '_ {
        let tick = self.tick;
        unsafe { &mut *self.inner }.iter_mut_tracked(tick)
    }

    /// Returns whether the entity has this component, ignoring disabled status.
    pub fn contains_unfiltered(&self, entity_index: u32) -> bool {
        unsafe { &*self.inner }.contains(entity_index)
    }
}

impl<T: 'static> Deref for RefMut<'_, T> {
    type Target = SparseSetInner<T>;

    fn deref(&self) -> &Self::Target {
        // SAFETY: We have exclusive access guaranteed by the write lock.
        unsafe { &*self.inner }
    }
}

impl<T: 'static> DerefMut for RefMut<'_, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        // SAFETY: We have exclusive access guaranteed by the write lock.
        unsafe { &mut *self.inner }
    }
}

// RefMut holds either an RwLockWriteGuard (auto-released on drop) or nothing.
// No manual Drop needed — the Option<RwLockWriteGuard> handles it.

// SAFETY: RefMut has exclusive access to the inner data.
// The RwLock ensures no other access exists when the guard is held.
unsafe impl<T: Send + Sync + 'static> Send for RefMut<'_, T> {}
unsafe impl<T: Send + Sync + 'static> Sync for RefMut<'_, T> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_and_get() {
        let mut set = SparseSetInner::<u32>::new();
        set.insert(5, 42);
        assert_eq!(set.get(5), Some(&42));
    }

    #[test]
    fn insert_replace() {
        let mut set = SparseSetInner::<u32>::new();
        set.insert(5, 42);
        set.insert(5, 99);
        assert_eq!(set.get(5), Some(&99));
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn remove_returns_value() {
        let mut set = SparseSetInner::<u32>::new();
        set.insert(5, 42);
        assert_eq!(set.remove(5), Some(42));
        assert_eq!(set.get(5), None);
    }

    #[test]
    fn remove_nonexistent() {
        let mut set = SparseSetInner::<u32>::new();
        assert_eq!(set.remove(5), None);
    }

    #[test]
    fn contains() {
        let mut set = SparseSetInner::<u32>::new();
        assert!(!set.contains(5));
        set.insert(5, 42);
        assert!(set.contains(5));
        set.remove(5);
        assert!(!set.contains(5));
    }

    #[test]
    fn iteration() {
        let mut set = SparseSetInner::<&str>::new();
        set.insert(1, "a");
        set.insert(5, "b");
        set.insert(3, "c");

        let mut items: Vec<_> = set.iter().collect();
        items.sort_by_key(|(idx, _)| *idx);
        assert_eq!(items, vec![(1, &"a"), (3, &"c"), (5, &"b")]);
    }

    #[test]
    fn swap_remove_correctness() {
        let mut set = SparseSetInner::<u32>::new();
        set.insert(0, 10);
        set.insert(1, 20);
        set.insert(2, 30);

        // Remove middle element (dense index 1), last element (entity 2) swaps in
        set.remove(1);

        assert_eq!(set.get(0), Some(&10));
        assert_eq!(set.get(1), None);
        assert_eq!(set.get(2), Some(&30));
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn lock_shared_multiple() {
        let lock = parking_lot::RwLock::new(ComponentStorage::new::<u32>());
        let _a = lock.read();
        let _b = lock.read();
        // Both locks succeed
    }

    #[test]
    fn lock_exclusive_alone() {
        let lock = parking_lot::RwLock::new(ComponentStorage::new::<u32>());
        let _guard = lock.write();
    }

    #[test]
    fn lock_exclusive_conflicts_shared() {
        let lock = parking_lot::RwLock::new(ComponentStorage::new::<u32>());
        let _r = lock.read();
        // parking_lot RwLock would deadlock here, not panic.
        // With the new design, conflicts are handled by the scheduler,
        // not by the lock itself. This test is no longer applicable.
        // Just verify that a try_write returns None when read-locked.
        assert!(lock.try_write().is_none());
    }

    #[test]
    fn lock_shared_conflicts_exclusive() {
        let lock = parking_lot::RwLock::new(ComponentStorage::new::<u32>());
        let _w = lock.write();
        // parking_lot RwLock would deadlock here, not panic.
        // Just verify that a try_read returns None when write-locked.
        assert!(lock.try_read().is_none());
    }

    #[test]
    fn lock_released_on_drop() {
        let lock = parking_lot::RwLock::new(ComponentStorage::new::<u32>());
        let flags: &[u32] = &[];
        {
            let _guard = Ref::<u32>::new(&lock, flags);
        }
        // After Ref is dropped, exclusive lock should succeed
        let _guard = RefMut::<u32>::new(&lock, flags, 0);
    }

    #[test]
    fn ref_mut_allows_mutation() {
        let lock = parking_lot::RwLock::new(ComponentStorage::new::<u32>());
        lock.write().typed_mut::<u32>().insert(0, 42);
        let flags: &[u32] = &[];
        {
            let mut guard = RefMut::<u32>::new(&lock, flags, 0);
            guard.insert(0, 99);
        }
        assert_eq!(lock.read().typed::<u32>().get(0), Some(&99));
    }

    #[test]
    fn remove_untyped_works() {
        let mut storage = ComponentStorage::new::<u32>();
        storage.typed_mut::<u32>().insert(5, 42);
        assert!(storage.contains_untyped(5));
        assert!(storage.remove_untyped(5));
        assert!(!storage.contains_untyped(5));
    }

    // ---- Change detection tests ----

    #[test]
    fn insert_with_tick_tracks_added() {
        let mut set = SparseSetInner::<u32>::new();
        set.insert_with_tick(5, 42, 10);
        assert!(set.added_since(5, 0));
        assert!(set.added_since(5, 9));
        assert!(!set.added_since(5, 10)); // not strictly after
        assert!(!set.added_since(5, 11));
    }

    #[test]
    fn insert_with_tick_tracks_changed() {
        let mut set = SparseSetInner::<u32>::new();
        set.insert_with_tick(5, 42, 10);
        assert!(set.changed_since(5, 0));
        assert!(set.changed_since(5, 9));
        assert!(!set.changed_since(5, 10));
    }

    #[test]
    fn replace_updates_changed_tick() {
        let mut set = SparseSetInner::<u32>::new();
        set.insert_with_tick(5, 42, 10);
        set.insert_with_tick(5, 99, 20); // replace

        // Added tick stays at 10
        assert!(set.added_since(5, 9));
        assert!(!set.added_since(5, 10));

        // Changed tick is now 20
        assert!(set.changed_since(5, 19));
        assert!(!set.changed_since(5, 20));
    }

    #[test]
    fn get_mut_tracked_marks_changed() {
        let mut set = SparseSetInner::<u32>::new();
        set.insert_with_tick(5, 42, 10);

        *set.get_mut_tracked(5, 25).unwrap() = 99;
        assert_eq!(set.get(5), Some(&99));
        assert!(set.changed_since(5, 24));
        assert!(!set.changed_since(5, 25));
    }

    #[test]
    fn iter_mut_tracked_marks_all_changed() {
        let mut set = SparseSetInner::<u32>::new();
        set.insert_with_tick(0, 10, 1);
        set.insert_with_tick(1, 20, 2);
        set.insert_with_tick(2, 30, 3);

        for (_, mut val) in set.iter_mut_tracked(50) {
            *val += 1;
        }

        assert!(set.changed_since(0, 49));
        assert!(set.changed_since(1, 49));
        assert!(set.changed_since(2, 49));
        assert!(!set.changed_since(0, 50));
    }

    #[test]
    fn untracked_insert_uses_tick_zero() {
        let mut set = SparseSetInner::<u32>::new();
        set.insert(5, 42); // tick = 0
        assert!(!set.added_since(5, 0));
        assert!(!set.changed_since(5, 0));
    }

    #[test]
    fn remove_maintains_tick_arrays() {
        let mut set = SparseSetInner::<u32>::new();
        set.insert_with_tick(0, 10, 1);
        set.insert_with_tick(1, 20, 5);
        set.insert_with_tick(2, 30, 10);

        // Remove entity 0 — entity 2 swaps into its slot
        set.remove(0);

        // Entity 2 should retain its ticks
        assert!(set.added_since(2, 9));
        assert!(!set.added_since(2, 10));

        // Entity 1 should retain its ticks
        assert!(set.added_since(1, 4));
        assert!(!set.added_since(1, 5));
    }

    #[test]
    fn changed_since_nonexistent_entity() {
        let set = SparseSetInner::<u32>::new();
        assert!(!set.changed_since(999, 0));
        assert!(!set.added_since(999, 0));
    }

    #[test]
    fn storage_changed_since_untyped() {
        let mut storage = ComponentStorage::new::<u32>();
        storage.typed_mut::<u32>().insert_with_tick(5, 42, 10);

        assert!(storage.changed_since_untyped(5, 9));
        assert!(!storage.changed_since_untyped(5, 10));
        assert!(storage.added_since_untyped(5, 9));
        assert!(!storage.added_since_untyped(5, 10));
    }

    // ---- Membership bitset tests ----

    #[test]
    fn membership_tracks_insert() {
        let mut set = SparseSetInner::<u32>::new();
        assert!(!set.membership().contains(5));
        set.insert(5, 42);
        assert!(set.membership().contains(5));
    }

    #[test]
    fn membership_tracks_remove() {
        let mut set = SparseSetInner::<u32>::new();
        set.insert(5, 42);
        assert!(set.membership().contains(5));
        set.remove(5);
        assert!(!set.membership().contains(5));
    }

    #[test]
    fn membership_replace_keeps_bit() {
        let mut set = SparseSetInner::<u32>::new();
        set.insert(5, 42);
        set.insert(5, 99); // replace
        assert!(set.membership().contains(5));
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn membership_multiple_entities() {
        let mut set = SparseSetInner::<u32>::new();
        set.insert(0, 10);
        set.insert(5, 20);
        set.insert(100, 30);
        assert!(set.membership().contains(0));
        assert!(set.membership().contains(5));
        assert!(set.membership().contains(100));
        assert!(!set.membership().contains(1));
        assert!(!set.membership().contains(50));
    }

    #[test]
    fn membership_remove_middle_entity() {
        let mut set = SparseSetInner::<u32>::new();
        set.insert(0, 10);
        set.insert(1, 20);
        set.insert(2, 30);
        set.remove(1);
        assert!(set.membership().contains(0));
        assert!(!set.membership().contains(1));
        assert!(set.membership().contains(2));
    }
}
