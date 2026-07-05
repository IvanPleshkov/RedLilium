use std::any::TypeId;
use std::marker::PhantomData;

use smallvec::SmallVec;

use crate::query::{
    AddedFilter, AnyFilter, ChangedFilter, ContainsChecker, Filter, OrFilter, RemovedFilter, With,
    Without,
};
use crate::resource::{ResourceRef, ResourceRefMut};
use crate::sparse_set::{Ref, RefMut};
use crate::world::World;

/// The storage class an [`AccessInfo`] refers to.
///
/// Two accesses are only the *same underlying storage* when both their
/// [`AccessInfo::type_id`] and `kind` match. This distinction matters because
/// components, `Arc<RwLock<T>>` resources, and main-thread resources of the
/// same type `T` all report `TypeId::of::<T>()` yet live in independent
/// storages, so they must not be merged (or rejected) against one another.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum AccessKind {
    /// Real component storage (`Read`/`Write`/`ReadAll`/`WriteAll`/`Optional*`).
    Component,
    /// A filter that reads component-storage metadata (change/add/remove ticks
    /// or membership): `Changed`/`Added`/`Removed`/`With`/`Without` and their
    /// `Maybe*` variants. Carries the real component `TypeId` so it acquires a
    /// **read** lock on that component's storage — this serializes the filter
    /// against any concurrent `Write<T>` system, preventing a data race on the
    /// change/add ticks. Shares a storage with [`AccessKind::Component`].
    ComponentFilter,
    /// `Arc<RwLock<T>>` resource (`Res`/`ResMut`).
    Resource,
    /// Main-thread-only resource (`MainThreadRes`/`MainThreadResMut`).
    MainThreadResource,
    /// Pure filter marker that borrows no storage (`Or`/`Any` combinators).
    /// Carries a unique marker `TypeId`.
    Filter,
}

/// Identifies the storage an access refers to, for deduplication and conflict
/// checks. [`AccessKind::Component`] and [`AccessKind::ComponentFilter`] of the
/// same type map to the same class because they lock the same underlying
/// component storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum StorageClass {
    Component,
    Resource,
    MainThreadResource,
    Marker,
}

impl AccessKind {
    fn storage_class(self) -> StorageClass {
        match self {
            AccessKind::Component | AccessKind::ComponentFilter => StorageClass::Component,
            AccessKind::Resource => StorageClass::Resource,
            AccessKind::MainThreadResource => StorageClass::MainThreadResource,
            AccessKind::Filter => StorageClass::Marker,
        }
    }
}

/// Metadata about a single component/resource access request.
#[derive(Debug, Clone, Copy)]
pub struct AccessInfo {
    pub type_id: TypeId,
    pub is_write: bool,
    pub kind: AccessKind,
}

impl AccessInfo {
    /// Access to a component storage.
    pub(crate) fn component(type_id: TypeId, is_write: bool) -> Self {
        Self {
            type_id,
            is_write,
            kind: AccessKind::Component,
        }
    }

    /// Access to an `Arc<RwLock<T>>` resource.
    pub(crate) fn resource(type_id: TypeId, is_write: bool) -> Self {
        Self {
            type_id,
            is_write,
            kind: AccessKind::Resource,
        }
    }

    /// Access to a main-thread-only resource.
    pub(crate) fn main_thread(type_id: TypeId, is_write: bool) -> Self {
        Self {
            type_id,
            is_write,
            kind: AccessKind::MainThreadResource,
        }
    }

    /// A filter that reads a component's storage metadata (locks the component
    /// for read). `type_id` is the real component `TypeId`.
    pub(crate) fn component_filter(type_id: TypeId) -> Self {
        Self {
            type_id,
            is_write: false,
            kind: AccessKind::ComponentFilter,
        }
    }

    /// A pure filter marker access that borrows no storage.
    pub(crate) fn filter(type_id: TypeId) -> Self {
        Self {
            type_id,
            is_write: false,
            kind: AccessKind::Filter,
        }
    }

    /// Sort/dedup key identifying the underlying storage.
    fn storage_key(&self) -> (TypeId, StorageClass) {
        (self.type_id, self.kind.storage_class())
    }

    /// Whether two infos refer to the same underlying storage.
    fn same_storage(&self, other: &AccessInfo) -> bool {
        self.storage_key() == other.storage_key()
    }
}

/// Validates that no aliasing-sensitive storage (a component, resource, or
/// main-thread resource) appears more than once in a single access set when
/// any of those accesses is mutable.
///
/// Such a set — e.g. `(Write<T>, Write<T>)`, `(Read<T>, Write<T>)`,
/// `(ResMut<T>, ResMut<T>)`, or `(MainThreadRes<T>, MainThreadResMut<T>)` —
/// would otherwise hand out two references to the same storage (`&mut`/`&mut`
/// or `&`/`&mut`) because locks are deduplicated by storage while data is
/// fetched per element, producing undefined behavior. Panics with a clear
/// message instead.
///
/// Resources do self-lock via their `Arc<RwLock<T>>`, but only on the locking
/// `fetch` path; the production `fetch_unlocked` path relies on the
/// deduplicated lock plan and hands out guard-less references, so duplicate
/// resources must be rejected here just like components. Filters carry unique
/// marker `TypeId`s and borrow no storage exclusively.
pub(crate) fn validate_no_aliasing_conflict(infos: &[AccessInfo]) {
    for (i, a) in infos.iter().enumerate() {
        let class = a.kind.storage_class();
        // Pure markers borrow no storage; everything else can alias.
        if class == StorageClass::Marker {
            continue;
        }
        for b in &infos[i + 1..] {
            // A storage accessed mutably must be the sole accessor in the set:
            // any second access (read, write, or storage-reading filter) to the
            // same storage would alias it (`&mut`/`&mut` or `&mut`/`&`) = UB.
            if a.same_storage(b) && (a.is_write || b.is_write) {
                let what = match class {
                    StorageClass::MainThreadResource => "main-thread resource",
                    StorageClass::Resource => "resource",
                    _ => "component",
                };
                panic!(
                    "the same {what} appears more than once in a single access set \
                     with at least one mutable access; this would alias the same \
                     storage (undefined behavior). A {what} accessed mutably must be \
                     the sole access to it in the set (e.g. `(Write<T>, Write<T>)`, \
                     `(Read<T>, Write<T>)`, and `(Write<T>, Changed<T>)` are not \
                     allowed)."
                );
            }
        }
    }
}

/// Normalizes access infos: sorts by underlying storage and deduplicates,
/// upgrading to write if any duplicate requests write access.
pub(crate) fn normalize_access_infos(infos: &[AccessInfo]) -> SmallVec<[AccessInfo; 8]> {
    let mut sorted: SmallVec<[AccessInfo; 8]> = infos.into();
    sorted.sort_by_key(|info| info.storage_key());
    sorted.dedup_by(|a, b| {
        if a.same_storage(b) {
            b.is_write = b.is_write || a.is_write;
            true
        } else {
            false
        }
    });
    sorted
}

/// The tick window a fetch operates in.
///
/// Runners assign each system run a fresh `this_run` tick and remember the
/// previous one as `last_run` (per system), so change filters see exactly
/// the mutations recorded since that system's previous run — regardless of
/// system order within a frame or how many frames ago it last ran.
#[derive(Debug, Clone, Copy)]
pub struct FetchTicks {
    /// Change/add/remove filters match events recorded strictly after this.
    pub last_run: u64,
    /// Mutable accessors stamp component writes with this tick.
    pub this_run: u64,
}

impl FetchTicks {
    /// Legacy frame-window ticks: "changed since the previous frame".
    ///
    /// Used outside of runner-managed systems (tests, [`World::query`],
    /// `run_system_once`), where there is no per-system `last_run`.
    pub fn frame(world: &World) -> Self {
        let now = world.current_tick();
        Self {
            last_run: now.saturating_sub(1),
            this_run: now,
        }
    }
}

/// Trait for a single access element (Read, Write, Res, etc.).
///
/// Each element knows its TypeId, whether it's a write, and how to
/// fetch its data from a World.
pub trait AccessElement {
    /// The type received by the execute closure for this element.
    type Item<'w>;

    /// Returns the access metadata for this element.
    fn access_info() -> AccessInfo;

    /// Appends this element's access metadata to `out`.
    ///
    /// Most elements contribute exactly one entry
    /// ([`access_info`](Self::access_info)); combinator filters ([`Or`],
    /// [`Any`]) recurse into their nested elements instead, so the storages
    /// those filters read are visible to the aliasing validator and included
    /// in the lock plan. A combinator that hid them would let
    /// `(Write<T>, Or<With<T>, …>)` fetch the same storage mutably and
    /// shared at once, and would let the filter read storage metadata
    /// without any lock against a parallel writer.
    fn collect_access_infos(out: &mut SmallVec<[AccessInfo; 8]>) {
        out.push(Self::access_info());
    }

    /// Fetches this element's data from the world, acquiring per-storage locks.
    fn fetch(world: &World, ticks: FetchTicks) -> Self::Item<'_>;

    /// Fetches this element's data without acquiring locks.
    ///
    /// The caller must ensure that the appropriate locks are already held
    /// externally (e.g. via `World::acquire_sorted`).
    fn fetch_unlocked(world: &World, ticks: FetchTicks) -> Self::Item<'_>;

    /// Whether this element requires main-thread access.
    ///
    /// Returns `true` for [`MainThreadRes`] and [`MainThreadResMut`].
    /// When any element in an access set returns `true`, the entire
    /// `execute()` closure is dispatched to the main thread.
    fn needs_main_thread() -> bool {
        false
    }
}

/// Trait for a set of access elements (tuples of Read/Write/Res/etc.).
///
/// Implemented for tuples up to 8 elements via macro.
/// Provides sorted access metadata and batch fetching.
pub trait AccessSet {
    /// The tuple of items received by the execute closure.
    type Item<'w>;

    /// Returns access metadata for all elements.
    fn access_infos() -> SmallVec<[AccessInfo; 8]>;

    /// Fetches all elements from the world, acquiring per-storage locks.
    fn fetch(world: &World, ticks: FetchTicks) -> Self::Item<'_>;

    /// Fetches all elements without acquiring locks.
    ///
    /// The caller must ensure locks are already held externally.
    fn fetch_unlocked(world: &World, ticks: FetchTicks) -> Self::Item<'_>;

    /// Returns `true` if any element in the set requires main-thread access.
    fn needs_main_thread() -> bool {
        false
    }
}

// ---- Marker types ----

/// Shared read access to component type `T`.
///
/// In the execute closure, yields `Ref<'_, T>` (deref to `SparseSetInner<T>`).
///
/// # Panics
///
/// Panics if `T` has never been registered.
pub struct Read<T: 'static>(PhantomData<T>);

/// Exclusive write access to component type `T`.
///
/// In the execute closure, yields `RefMut<'_, T>` (deref to `SparseSetInner<T>`).
///
/// # Panics
///
/// Panics if `T` has never been registered.
pub struct Write<T: 'static>(PhantomData<T>);

/// Optional shared read access to component type `T`.
///
/// In the execute closure, yields `Option<Ref<'_, T>>`.
/// Returns `None` if the type has never been registered (no panic).
pub struct OptionalRead<T: 'static>(PhantomData<T>);

/// Optional exclusive write access to component type `T`.
///
/// In the execute closure, yields `Option<RefMut<'_, T>>`.
/// Returns `None` if the type has never been registered (no panic).
pub struct OptionalWrite<T: 'static>(PhantomData<T>);

/// Shared read access to a resource of type `T`.
///
/// In the execute closure, yields `ResourceRef<'_, T>`.
///
/// # Panics
///
/// Panics if the resource does not exist.
pub struct Res<T: 'static>(PhantomData<T>);

/// Exclusive write access to a resource of type `T`.
///
/// In the execute closure, yields `ResourceRefMut<'_, T>`.
///
/// # Panics
///
/// Panics if the resource does not exist.
pub struct ResMut<T: 'static>(PhantomData<T>);

/// Shared read access to component type `T`, including static entities.
///
/// Like [`Read<T>`], but the resulting [`Ref`] only excludes disabled
/// entities — static entities are included in iteration. Use this in
/// systems that need to observe all active entities (e.g., rendering,
/// physics broadphase).
///
/// In the execute closure, yields `Ref<'_, T>`.
///
/// # Panics
///
/// Panics if `T` has never been registered.
pub struct ReadAll<T: 'static>(PhantomData<T>);

/// Exclusive write access including static and editor entities.
///
/// Like [`Write`], but only excludes disabled entities — static and editor
/// entities are included. Use this in editor systems or rendering passes
/// that need to mutate all active entities.
///
/// In the execute closure, yields `RefMut<'_, T>`.
///
/// # Panics
///
/// Panics if `T` has never been registered.
pub struct WriteAll<T: 'static>(PhantomData<T>);

/// Shared read access to a main-thread resource of type `T`.
///
/// `T` does **not** need to be `Send + Sync`. The scheduler transparently
/// dispatches the `execute()` closure to the main thread when this type
/// is in the access set.
///
/// In the execute closure, yields `&T`.
///
/// # Panics
///
/// Panics if the main-thread resource does not exist.
pub struct MainThreadRes<T: 'static>(PhantomData<T>);

/// Exclusive write access to a main-thread resource of type `T`.
///
/// `T` does **not** need to be `Send + Sync`. The scheduler transparently
/// dispatches the `execute()` closure to the main thread when this type
/// is in the access set.
///
/// In the execute closure, yields `&mut T`.
///
/// # Panics
///
/// Panics if the main-thread resource does not exist.
pub struct MainThreadResMut<T: 'static>(PhantomData<T>);

/// Filter for entities whose component `T` was added this tick.
///
/// In the execute closure, yields [`AddedFilter`](crate::AddedFilter).
/// Use `filter.matches(entity_index)` to check individual entities.
///
/// # Panics
///
/// Panics if `T` has never been registered. Use [`MaybeAdded`] for
/// a non-panicking variant.
pub struct Added<T: 'static>(PhantomData<T>);

/// Filter for entities whose component `T` was removed this tick.
///
/// In the execute closure, yields [`RemovedFilter`](crate::RemovedFilter).
/// Use `filter.matches(entity_index)` or `filter.iter()` to query.
///
/// # Panics
///
/// Panics if `T` has never been registered. Use [`MaybeRemoved`] for
/// a non-panicking variant.
pub struct Removed<T: 'static>(PhantomData<T>);

/// Optional filter for entities whose component `T` was added this tick.
///
/// In the execute closure, yields [`AddedFilter`](crate::AddedFilter).
/// If `T` has never been registered, the filter matches nothing (no panic).
pub struct MaybeAdded<T: 'static>(PhantomData<T>);

/// Optional filter for entities whose component `T` was removed this tick.
///
/// In the execute closure, yields [`RemovedFilter`](crate::RemovedFilter).
/// If `T` has never been registered, the filter matches nothing (no panic).
pub struct MaybeRemoved<T: 'static>(PhantomData<T>);

/// Filter for entities whose component `T` was changed this tick.
///
/// In the execute closure, yields [`ChangedFilter`](crate::ChangedFilter).
/// Use `filter.matches(entity_index)` to check individual entities.
///
/// A component is marked as changed when it is mutated through [`Mut<T>`](crate::Mut).
///
/// # Panics
///
/// Panics if `T` has never been registered. Use [`MaybeChanged`] for
/// a non-panicking variant.
pub struct Changed<T: 'static>(PhantomData<T>);

/// Optional filter for entities whose component `T` was changed this tick.
///
/// In the execute closure, yields [`ChangedFilter`](crate::ChangedFilter).
/// If `T` has never been registered, the filter matches nothing (no panic).
pub struct MaybeChanged<T: 'static>(PhantomData<T>);

/// Logical OR of two filter access elements.
///
/// In the execute closure, yields [`OrFilter`](crate::OrFilter).
/// Matches entities where **either** filter A or filter B matches.
///
/// Both `A` and `B` must be filter access elements (e.g. [`With`], [`Without`],
/// [`Added`], [`Removed`], or nested `Or`).
///
/// # Example
///
/// ```ignore
/// ctx.lock::<(Read<Position>, Or<With<Flying>, With<Swimming>>)>()
///     .execute(|(positions, can_move)| {
///         for (idx, pos) in positions.iter() {
///             if can_move.matches(idx) {
///                 // entity has Flying OR Swimming
///             }
///         }
///     });
/// ```
pub struct Or<A, B>(PhantomData<(A, B)>);

/// Logical OR of any number of filter access elements (tuple-based).
///
/// In the execute closure, yields [`AnyFilter`](crate::AnyFilter).
/// Matches entities where **any** of the sub-filters match.
/// Supports tuples of 2-8 filter elements.
///
/// # Example
///
/// ```ignore
/// ctx.lock::<(Read<Position>, Any<(With<Flying>, With<Swimming>, With<Walking>)>)>()
///     .execute(|(positions, movable)| {
///         for (idx, pos) in positions.iter() {
///             if movable.matches(idx) {
///                 // entity has Flying OR Swimming OR Walking
///             }
///         }
///     });
/// ```
pub struct Any<T>(PhantomData<T>);

// ---- AccessElement implementations ----

impl<T: 'static> AccessElement for Read<T> {
    type Item<'w> = Ref<'w, T>;

    fn access_info() -> AccessInfo {
        AccessInfo::component(TypeId::of::<T>(), false)
    }

    fn fetch(world: &World, _ticks: FetchTicks) -> Self::Item<'_> {
        world
            .read::<T>()
            .expect("Component not registered for Read<T> access")
    }

    fn fetch_unlocked(world: &World, _ticks: FetchTicks) -> Self::Item<'_> {
        world
            .read_unlocked::<T>()
            .expect("Component not registered for Read<T> access")
    }
}

impl<T: 'static> AccessElement for Write<T> {
    type Item<'w> = RefMut<'w, T>;

    fn access_info() -> AccessInfo {
        AccessInfo::component(TypeId::of::<T>(), true)
    }

    fn fetch(world: &World, ticks: FetchTicks) -> Self::Item<'_> {
        world
            .write_storage_at::<T>(ticks.this_run)
            .expect("Component not registered for Write<T> access")
    }

    fn fetch_unlocked(world: &World, ticks: FetchTicks) -> Self::Item<'_> {
        world
            .write_unlocked::<T>(ticks.this_run)
            .expect("Component not registered for Write<T> access")
    }
}

impl<T: 'static> AccessElement for ReadAll<T> {
    type Item<'w> = Ref<'w, T>;

    fn access_info() -> AccessInfo {
        AccessInfo::component(TypeId::of::<T>(), false)
    }

    fn fetch(world: &World, _ticks: FetchTicks) -> Self::Item<'_> {
        world
            .read_all::<T>()
            .expect("Component not registered for ReadAll<T> access")
    }

    fn fetch_unlocked(world: &World, _ticks: FetchTicks) -> Self::Item<'_> {
        world
            .read_all_unlocked::<T>()
            .expect("Component not registered for ReadAll<T> access")
    }
}

impl<T: 'static> AccessElement for WriteAll<T> {
    type Item<'w> = RefMut<'w, T>;

    fn access_info() -> AccessInfo {
        AccessInfo::component(TypeId::of::<T>(), true)
    }

    fn fetch(world: &World, ticks: FetchTicks) -> Self::Item<'_> {
        world
            .write_all_storage_at::<T>(ticks.this_run)
            .expect("Component not registered for WriteAll<T> access")
    }

    fn fetch_unlocked(world: &World, ticks: FetchTicks) -> Self::Item<'_> {
        world
            .write_all_unlocked::<T>(ticks.this_run)
            .expect("Component not registered for WriteAll<T> access")
    }
}

impl<T: 'static> AccessElement for OptionalRead<T> {
    type Item<'w> = Option<Ref<'w, T>>;

    fn access_info() -> AccessInfo {
        AccessInfo::component(TypeId::of::<T>(), false)
    }

    fn fetch(world: &World, _ticks: FetchTicks) -> Self::Item<'_> {
        world.try_read::<T>()
    }

    fn fetch_unlocked(world: &World, _ticks: FetchTicks) -> Self::Item<'_> {
        world.try_read_unlocked::<T>()
    }
}

impl<T: 'static> AccessElement for OptionalWrite<T> {
    type Item<'w> = Option<RefMut<'w, T>>;

    fn access_info() -> AccessInfo {
        AccessInfo::component(TypeId::of::<T>(), true)
    }

    fn fetch(world: &World, ticks: FetchTicks) -> Self::Item<'_> {
        world.try_write_storage_at::<T>(ticks.this_run)
    }

    fn fetch_unlocked(world: &World, ticks: FetchTicks) -> Self::Item<'_> {
        world.try_write_unlocked::<T>(ticks.this_run)
    }
}

impl<T: 'static> AccessElement for Res<T> {
    type Item<'w> = ResourceRef<'w, T>;

    fn access_info() -> AccessInfo {
        AccessInfo::resource(TypeId::of::<T>(), false)
    }

    fn fetch(world: &World, _ticks: FetchTicks) -> Self::Item<'_> {
        world.resource::<T>()
    }

    fn fetch_unlocked(world: &World, _ticks: FetchTicks) -> Self::Item<'_> {
        // The read lock was already acquired (in TypeId-sorted order) by
        // `acquire_sorted`; build a guardless view to avoid re-locking.
        // SAFETY: the lock is held for the duration of this access set.
        unsafe { world.resource_unlocked::<T>() }
    }
}

impl<T: 'static> AccessElement for ResMut<T> {
    type Item<'w> = ResourceRefMut<'w, T>;

    fn access_info() -> AccessInfo {
        AccessInfo::resource(TypeId::of::<T>(), true)
    }

    fn fetch(world: &World, _ticks: FetchTicks) -> Self::Item<'_> {
        world.resource_mut::<T>()
    }

    fn fetch_unlocked(world: &World, _ticks: FetchTicks) -> Self::Item<'_> {
        // The write lock was already acquired by `acquire_sorted`.
        // SAFETY: the lock is held for the duration of this access set.
        unsafe { world.resource_mut_unlocked::<T>() }
    }
}

impl<T: 'static> AccessElement for MainThreadRes<T> {
    type Item<'w> = &'w T;

    fn access_info() -> AccessInfo {
        AccessInfo::main_thread(TypeId::of::<T>(), false)
    }

    fn fetch(world: &World, _ticks: FetchTicks) -> Self::Item<'_> {
        // SAFETY: only called from main thread via dispatcher
        unsafe { world.main_thread_resource::<T>() }
    }

    fn fetch_unlocked(world: &World, _ticks: FetchTicks) -> Self::Item<'_> {
        // Same as fetch — main-thread resources have no locks
        unsafe { world.main_thread_resource::<T>() }
    }

    fn needs_main_thread() -> bool {
        true
    }
}

impl<T: 'static> AccessElement for MainThreadResMut<T> {
    type Item<'w> = &'w mut T;

    fn access_info() -> AccessInfo {
        AccessInfo::main_thread(TypeId::of::<T>(), true)
    }

    fn fetch(world: &World, _ticks: FetchTicks) -> Self::Item<'_> {
        // SAFETY: only called from main thread via dispatcher
        unsafe { world.main_thread_resource_mut::<T>() }
    }

    fn fetch_unlocked(world: &World, _ticks: FetchTicks) -> Self::Item<'_> {
        // Same as fetch — main-thread resources have no locks
        unsafe { world.main_thread_resource_mut::<T>() }
    }

    fn needs_main_thread() -> bool {
        true
    }
}

impl<T: 'static> AccessElement for Added<T> {
    type Item<'w> = AddedFilter<'w>;

    fn access_info() -> AccessInfo {
        // Read the real component's storage metadata under a read lock so this
        // filter serializes against any concurrent `Write<T>` system.
        AccessInfo::component_filter(TypeId::of::<T>())
    }

    fn fetch(world: &World, ticks: FetchTicks) -> Self::Item<'_> {
        assert!(
            world.is_component_registered::<T>(),
            "Component `{}` not registered for Added<T> filter",
            std::any::type_name::<T>()
        );
        let since_tick = ticks.last_run;
        world.added::<T>(since_tick)
    }

    fn fetch_unlocked(world: &World, ticks: FetchTicks) -> Self::Item<'_> {
        // Filters don't hold locks — same as fetch
        Self::fetch(world, ticks)
    }
}

impl<T: 'static> AccessElement for Removed<T> {
    type Item<'w> = RemovedFilter<'w>;

    fn access_info() -> AccessInfo {
        AccessInfo::component_filter(TypeId::of::<T>())
    }

    fn fetch(world: &World, ticks: FetchTicks) -> Self::Item<'_> {
        assert!(
            world.is_component_registered::<T>(),
            "Component `{}` not registered for Removed<T> filter",
            std::any::type_name::<T>()
        );
        let since_tick = ticks.last_run;
        world.removed::<T>(since_tick)
    }

    fn fetch_unlocked(world: &World, ticks: FetchTicks) -> Self::Item<'_> {
        Self::fetch(world, ticks)
    }
}

impl<T: 'static> AccessElement for MaybeAdded<T> {
    type Item<'w> = AddedFilter<'w>;

    fn access_info() -> AccessInfo {
        AccessInfo::component_filter(TypeId::of::<T>())
    }

    fn fetch(world: &World, ticks: FetchTicks) -> Self::Item<'_> {
        let since_tick = ticks.last_run;
        world.added::<T>(since_tick)
    }

    fn fetch_unlocked(world: &World, ticks: FetchTicks) -> Self::Item<'_> {
        Self::fetch(world, ticks)
    }
}

impl<T: 'static> AccessElement for Changed<T> {
    type Item<'w> = ChangedFilter<'w>;

    fn access_info() -> AccessInfo {
        AccessInfo::component_filter(TypeId::of::<T>())
    }

    fn fetch(world: &World, ticks: FetchTicks) -> Self::Item<'_> {
        assert!(
            world.is_component_registered::<T>(),
            "Component `{}` not registered for Changed<T> filter",
            std::any::type_name::<T>()
        );
        let since_tick = ticks.last_run;
        world.changed::<T>(since_tick)
    }

    fn fetch_unlocked(world: &World, ticks: FetchTicks) -> Self::Item<'_> {
        Self::fetch(world, ticks)
    }
}

impl<T: 'static> AccessElement for MaybeChanged<T> {
    type Item<'w> = ChangedFilter<'w>;

    fn access_info() -> AccessInfo {
        AccessInfo::component_filter(TypeId::of::<T>())
    }

    fn fetch(world: &World, ticks: FetchTicks) -> Self::Item<'_> {
        let since_tick = ticks.last_run;
        world.changed::<T>(since_tick)
    }

    fn fetch_unlocked(world: &World, ticks: FetchTicks) -> Self::Item<'_> {
        Self::fetch(world, ticks)
    }
}

impl<T: 'static> AccessElement for MaybeRemoved<T> {
    type Item<'w> = RemovedFilter<'w>;

    fn access_info() -> AccessInfo {
        AccessInfo::component_filter(TypeId::of::<T>())
    }

    fn fetch(world: &World, ticks: FetchTicks) -> Self::Item<'_> {
        let since_tick = ticks.last_run;
        world.removed::<T>(since_tick)
    }

    fn fetch_unlocked(world: &World, ticks: FetchTicks) -> Self::Item<'_> {
        Self::fetch(world, ticks)
    }
}

impl<T: 'static> AccessElement for With<T> {
    type Item<'w> = ContainsChecker<'w>;

    fn access_info() -> AccessInfo {
        AccessInfo::component_filter(TypeId::of::<T>())
    }

    fn fetch(world: &World, _ticks: FetchTicks) -> Self::Item<'_> {
        world.with::<T>()
    }

    fn fetch_unlocked(world: &World, _ticks: FetchTicks) -> Self::Item<'_> {
        // Filters don't hold locks — same as fetch
        world.with::<T>()
    }
}

impl<T: 'static> AccessElement for Without<T> {
    type Item<'w> = ContainsChecker<'w>;

    fn access_info() -> AccessInfo {
        AccessInfo::component_filter(TypeId::of::<T>())
    }

    fn fetch(world: &World, _ticks: FetchTicks) -> Self::Item<'_> {
        world.without::<T>()
    }

    fn fetch_unlocked(world: &World, _ticks: FetchTicks) -> Self::Item<'_> {
        // Filters don't hold locks — same as fetch
        world.without::<T>()
    }
}

// ---- Or<A, B> AccessElement ----

impl<A, B> AccessElement for Or<A, B>
where
    A: AccessElement + 'static,
    B: AccessElement + 'static,
    for<'w> A::Item<'w>: Filter,
    for<'w> B::Item<'w>: Filter,
{
    type Item<'w> = OrFilter<A::Item<'w>, B::Item<'w>>;

    fn access_info() -> AccessInfo {
        // Inert marker; the real metadata comes from `collect_access_infos`,
        // which surfaces the nested filters' storage reads.
        AccessInfo::filter(TypeId::of::<Or<A, B>>())
    }

    fn collect_access_infos(out: &mut SmallVec<[AccessInfo; 8]>) {
        A::collect_access_infos(out);
        B::collect_access_infos(out);
    }

    fn fetch(world: &World, ticks: FetchTicks) -> Self::Item<'_> {
        OrFilter::new(A::fetch(world, ticks), B::fetch(world, ticks))
    }

    fn fetch_unlocked(world: &World, ticks: FetchTicks) -> Self::Item<'_> {
        OrFilter::new(
            A::fetch_unlocked(world, ticks),
            B::fetch_unlocked(world, ticks),
        )
    }
}

// ---- Any<(A, B, ...)> AccessElement ----

macro_rules! impl_any_access_element {
    ($($idx:tt $T:ident),+) => {
        impl<$($T),+> AccessElement for Any<($($T,)+)>
        where
            $($T: AccessElement + 'static,)+
            $(for<'w> $T::Item<'w>: Filter,)+
        {
            type Item<'w> = AnyFilter<($($T::Item<'w>,)+)>;

            fn access_info() -> AccessInfo {
                // Inert marker; see `Or::access_info`.
                AccessInfo::filter(TypeId::of::<Any<($($T,)+)>>())
            }

            fn collect_access_infos(out: &mut SmallVec<[AccessInfo; 8]>) {
                $($T::collect_access_infos(out);)+
            }

            fn fetch(world: &World, ticks: FetchTicks) -> Self::Item<'_> {
                AnyFilter::new(($($T::fetch(world, ticks),)+))
            }

            fn fetch_unlocked(world: &World, ticks: FetchTicks) -> Self::Item<'_> {
                AnyFilter::new(($($T::fetch_unlocked(world, ticks),)+))
            }
        }
    };
}

impl_any_access_element!(0 A, 1 B);
impl_any_access_element!(0 A, 1 B, 2 C);
impl_any_access_element!(0 A, 1 B, 2 C, 3 D);
impl_any_access_element!(0 A, 1 B, 2 C, 3 D, 4 E);
impl_any_access_element!(0 A, 1 B, 2 C, 3 D, 4 E, 5 F);
impl_any_access_element!(0 A, 1 B, 2 C, 3 D, 4 E, 5 F, 6 G);
impl_any_access_element!(0 A, 1 B, 2 C, 3 D, 4 E, 5 F, 6 G, 7 H);

// ---- Tuple AccessSet implementations ----

// Empty tuple (no access)
impl AccessSet for () {
    type Item<'w> = ();

    fn access_infos() -> SmallVec<[AccessInfo; 8]> {
        SmallVec::new()
    }

    fn fetch(_world: &World, _ticks: FetchTicks) -> Self::Item<'_> {}

    fn fetch_unlocked(_world: &World, _ticks: FetchTicks) -> Self::Item<'_> {}

    fn needs_main_thread() -> bool {
        false
    }
}

macro_rules! impl_access_set {
    ($($idx:tt $T:ident),+) => {
        impl<$($T: AccessElement),+> AccessSet for ($($T,)+) {
            type Item<'w> = ($($T::Item<'w>,)+);

            fn access_infos() -> SmallVec<[AccessInfo; 8]> {
                let mut infos = SmallVec::new();
                $($T::collect_access_infos(&mut infos);)+
                infos
            }

            fn fetch(world: &World, ticks: FetchTicks) -> Self::Item<'_> {
                ($($T::fetch(world, ticks),)+)
            }

            fn fetch_unlocked(world: &World, ticks: FetchTicks) -> Self::Item<'_> {
                ($($T::fetch_unlocked(world, ticks),)+)
            }

            fn needs_main_thread() -> bool {
                $($T::needs_main_thread())||+
            }
        }
    };
}

impl_access_set!(0 A);
impl_access_set!(0 A, 1 B);
impl_access_set!(0 A, 1 B, 2 C);
impl_access_set!(0 A, 1 B, 2 C, 3 D);
impl_access_set!(0 A, 1 B, 2 C, 3 D, 4 E);
impl_access_set!(0 A, 1 B, 2 C, 3 D, 4 E, 5 F);
impl_access_set!(0 A, 1 B, 2 C, 3 D, 4 E, 5 F, 6 G);
impl_access_set!(0 A, 1 B, 2 C, 3 D, 4 E, 5 F, 6 G, 7 H);

#[cfg(test)]
mod tests {
    use super::*;

    struct Position {
        x: f32,
    }
    struct Velocity {
        _x: f32,
    }

    #[test]
    fn read_access_info() {
        let info = <Read<Position>>::access_info();
        assert_eq!(info.type_id, TypeId::of::<Position>());
        assert!(!info.is_write);
    }

    #[test]
    fn write_access_info() {
        let info = <Write<Position>>::access_info();
        assert_eq!(info.type_id, TypeId::of::<Position>());
        assert!(info.is_write);
    }

    #[test]
    fn tuple_access_infos() {
        let infos = <(Read<Position>, Write<Velocity>)>::access_infos();
        assert_eq!(infos.len(), 2);
        assert_eq!(infos[0].type_id, TypeId::of::<Position>());
        assert!(!infos[0].is_write);
        assert_eq!(infos[1].type_id, TypeId::of::<Velocity>());
        assert!(infos[1].is_write);
    }

    #[test]
    fn empty_tuple() {
        let infos = <()>::access_infos();
        assert!(infos.is_empty());
    }

    #[test]
    #[should_panic(expected = "more than once")]
    fn duplicate_write_write_rejected() {
        let infos = <(Write<Position>, Write<Position>)>::access_infos();
        validate_no_aliasing_conflict(&infos);
    }

    #[test]
    #[should_panic(expected = "more than once")]
    fn duplicate_read_write_rejected() {
        let infos = <(Read<Position>, Write<Position>)>::access_infos();
        validate_no_aliasing_conflict(&infos);
    }

    #[test]
    #[should_panic(expected = "main-thread resource")]
    fn duplicate_main_thread_res_resmut_rejected() {
        let infos = <(MainThreadRes<Position>, MainThreadResMut<Position>)>::access_infos();
        validate_no_aliasing_conflict(&infos);
    }

    #[test]
    fn or_surfaces_nested_storages() {
        // Or/Any must report the storages their nested filters read; a bare
        // marker would hide them from the validator and the lock plan
        // (issue #10).
        let infos = <(Or<With<Position>, Changed<Velocity>>,)>::access_infos();
        assert_eq!(infos.len(), 2);
        assert_eq!(infos[0].type_id, TypeId::of::<Position>());
        assert_eq!(infos[1].type_id, TypeId::of::<Velocity>());
        assert!(infos.iter().all(|i| !i.is_write));

        let infos = <(Any<(With<Position>, With<Velocity>, Changed<Position>)>,)>::access_infos();
        assert_eq!(infos.len(), 3);
    }

    #[test]
    #[should_panic(expected = "more than once")]
    fn write_and_or_nested_same_component_rejected() {
        // A filter on T nested in Or reads T's storage; combined with
        // Write<T> that aliases mutably (issue #10).
        let infos = <(Write<Position>, Or<With<Position>, With<Velocity>>)>::access_infos();
        validate_no_aliasing_conflict(&infos);
    }

    #[test]
    #[should_panic(expected = "resource")]
    fn duplicate_resmut_resmut_rejected() {
        // Resources self-lock only on the locking `fetch` path; the unlocked
        // path dedups the pair into one write lock and would hand out two
        // `&mut` to the same resource (issue #11).
        let infos = <(ResMut<Position>, ResMut<Position>)>::access_infos();
        validate_no_aliasing_conflict(&infos);
    }

    #[test]
    #[should_panic(expected = "resource")]
    fn duplicate_res_resmut_rejected() {
        let infos = <(Res<Position>, ResMut<Position>)>::access_infos();
        validate_no_aliasing_conflict(&infos);
    }

    #[test]
    fn duplicate_res_res_allowed() {
        let infos = <(Res<Position>, Res<Position>)>::access_infos();
        validate_no_aliasing_conflict(&infos);
    }

    #[test]
    fn duplicate_read_read_allowed() {
        // Two shared reads of the same component do not alias mutably.
        let infos = <(Read<Position>, Read<Position>)>::access_infos();
        validate_no_aliasing_conflict(&infos);
    }

    #[test]
    fn distinct_components_allowed() {
        let infos = <(Write<Position>, Write<Velocity>)>::access_infos();
        validate_no_aliasing_conflict(&infos);
    }

    #[test]
    fn component_and_resource_same_type_not_conflated() {
        // A type used as both a component and a resource lives in independent
        // storages and must not be rejected even when both are mutable.
        let infos = <(Write<Position>, ResMut<Position>)>::access_infos();
        validate_no_aliasing_conflict(&infos);
    }

    #[test]
    #[should_panic(expected = "more than once")]
    fn write_and_changed_same_component_rejected() {
        // A storage-reading filter borrows the component storage shared while
        // Write borrows it mutably — aliasing, so the combination is rejected.
        let infos = <(Write<Position>, Changed<Position>)>::access_infos();
        validate_no_aliasing_conflict(&infos);
    }

    #[test]
    fn read_and_changed_same_component_allowed() {
        // Read + filter are both shared borrows of the storage — no aliasing.
        let infos = <(Read<Position>, Changed<Position>)>::access_infos();
        validate_no_aliasing_conflict(&infos);
    }

    #[test]
    fn changed_filter_dedups_with_read_into_single_lock() {
        // Read<T> and a storage-reading filter on the same T must collapse to a
        // single (read) lock entry, never two locks on the same storage.
        let infos = <(Read<Position>, With<Position>, Changed<Position>)>::access_infos();
        let normalized = normalize_access_infos(&infos);
        assert_eq!(normalized.len(), 1);
        assert_eq!(normalized[0].type_id, TypeId::of::<Position>());
        assert!(!normalized[0].is_write);
    }

    #[test]
    fn fetch_reads_from_world() {
        let mut world = World::new();
        world.register_component::<Position>();
        world.register_component::<Velocity>();
        let e = world.spawn();
        world.insert(e, Position { x: 42.0 }).unwrap();
        world.insert(e, Velocity { _x: 5.0 }).unwrap();

        let (positions, velocities) =
            <(Read<Position>, Read<Velocity>)>::fetch(&world, FetchTicks::frame(&world));
        assert_eq!(positions.len(), 1);
        assert_eq!(velocities.len(), 1);
    }

    #[test]
    fn fetch_write_from_world() {
        let mut world = World::new();
        world.register_component::<Position>();
        let e = world.spawn();
        world.insert(e, Position { x: 0.0 }).unwrap();

        let (mut positions,) = <(Write<Position>,)>::fetch(&world, FetchTicks::frame(&world));
        for (_, mut pos) in positions.iter_mut() {
            pos.x = 99.0;
        }
        drop(positions);

        assert_eq!(world.get::<Position>(e).unwrap().x, 99.0);
    }

    #[test]
    fn optional_read_returns_none_for_unregistered() {
        let world = World::new();
        let (opt,) = <(OptionalRead<Position>,)>::fetch(&world, FetchTicks::frame(&world));
        assert!(opt.is_none());
    }

    #[test]
    fn optional_read_returns_some_for_registered() {
        let mut world = World::new();
        world.register_component::<Position>();
        let e = world.spawn();
        world.insert(e, Position { x: 1.0 }).unwrap();

        let (opt,) = <(OptionalRead<Position>,)>::fetch(&world, FetchTicks::frame(&world));
        assert!(opt.is_some());
        assert_eq!(opt.unwrap().len(), 1);
    }

    #[test]
    fn res_fetch_from_world() {
        let mut world = World::new();
        world.insert_resource(1.5f64);

        let (dt,) = <(Res<f64>,)>::fetch(&world, FetchTicks::frame(&world));
        assert_eq!(*dt, 1.5);
    }

    // ---- Added/Removed filter tests ----

    #[derive(Debug, PartialEq)]
    struct Health(u32);

    #[test]
    fn added_filter_access_info_locks_component() {
        let info = <Added<Position>>::access_info();
        // Storage-reading filters lock the real component (read), so they
        // report the component's TypeId and a ComponentFilter kind.
        assert_eq!(info.type_id, TypeId::of::<Position>());
        assert_eq!(info.kind, AccessKind::ComponentFilter);
        assert!(!info.is_write);
    }

    #[test]
    fn removed_filter_access_info_locks_component() {
        let info = <Removed<Position>>::access_info();
        assert_eq!(info.type_id, TypeId::of::<Position>());
        assert_eq!(info.kind, AccessKind::ComponentFilter);
        assert!(!info.is_write);
    }

    #[test]
    fn added_filter_detects_addition() {
        let mut world = World::new();
        world.register_component::<Health>();

        world.advance_tick(); // tick = 1
        let e = world.spawn();
        world.insert(e, Health(100)).unwrap();

        let (filter,) = <(Added<Health>,)>::fetch(&world, FetchTicks::frame(&world));
        assert!(filter.matches(e.index()));
    }

    #[test]
    fn added_filter_does_not_match_old() {
        let mut world = World::new();
        world.register_component::<Health>();

        let e = world.spawn();
        world.insert(e, Health(100)).unwrap(); // tick 0

        world.advance_tick(); // tick = 1
        world.advance_tick(); // tick = 2

        // since_tick = 2 - 1 = 1, component was added at tick 0, so 0 > 1 is false
        let (filter,) = <(Added<Health>,)>::fetch(&world, FetchTicks::frame(&world));
        assert!(!filter.matches(e.index()));
    }

    #[test]
    fn removed_filter_detects_removal() {
        let mut world = World::new();
        world.register_component::<Health>();

        let e = world.spawn();
        world.insert(e, Health(100)).unwrap();

        world.advance_tick(); // tick = 1
        let _ = world.remove::<Health>(e); // removed at tick 1

        let (filter,) = <(Removed<Health>,)>::fetch(&world, FetchTicks::frame(&world));
        assert!(filter.matches(e.index()));
    }

    #[test]
    fn removed_filter_iter_in_tuple() {
        let mut world = World::new();
        world.register_component::<Position>();
        world.register_component::<Health>();

        let e1 = world.spawn();
        let e2 = world.spawn();
        world.insert(e1, Position { x: 1.0 }).unwrap();
        world.insert(e1, Health(100)).unwrap();
        world.insert(e2, Position { x: 2.0 }).unwrap();
        world.insert(e2, Health(200)).unwrap();

        world.advance_tick(); // tick = 1
        let _ = world.remove::<Health>(e1); // removed at tick 1

        let (positions, removed) =
            <(Read<Position>, Removed<Health>)>::fetch(&world, FetchTicks::frame(&world));
        let affected: Vec<f32> = positions
            .iter()
            .filter(|(idx, _)| removed.matches(*idx))
            .map(|(_, p)| p.x)
            .collect();
        assert_eq!(affected, vec![1.0]);
    }

    #[test]
    #[should_panic(expected = "not registered for Added")]
    fn added_panics_for_unregistered() {
        let world = World::new();
        let _ = <(Added<Health>,)>::fetch(&world, FetchTicks::frame(&world));
    }

    #[test]
    #[should_panic(expected = "not registered for Removed")]
    fn removed_panics_for_unregistered() {
        let world = World::new();
        let _ = <(Removed<Health>,)>::fetch(&world, FetchTicks::frame(&world));
    }

    #[test]
    fn maybe_added_no_panic_for_unregistered() {
        let world = World::new();
        let (filter,) = <(MaybeAdded<Health>,)>::fetch(&world, FetchTicks::frame(&world));
        assert!(!filter.matches(0));
    }

    #[test]
    fn maybe_removed_no_panic_for_unregistered() {
        let world = World::new();
        let (filter,) = <(MaybeRemoved<Health>,)>::fetch(&world, FetchTicks::frame(&world));
        assert!(!filter.matches(0));
    }

    #[test]
    fn maybe_added_works_when_registered() {
        let mut world = World::new();
        world.register_component::<Health>();

        world.advance_tick(); // tick = 1
        let e = world.spawn();
        world.insert(e, Health(50)).unwrap();

        let (filter,) = <(MaybeAdded<Health>,)>::fetch(&world, FetchTicks::frame(&world));
        assert!(filter.matches(e.index()));
    }

    #[test]
    fn maybe_removed_works_when_registered() {
        let mut world = World::new();
        world.register_component::<Health>();

        let e = world.spawn();
        world.insert(e, Health(50)).unwrap();

        world.advance_tick(); // tick = 1
        let _ = world.remove::<Health>(e);

        let (filter,) = <(MaybeRemoved<Health>,)>::fetch(&world, FetchTicks::frame(&world));
        assert!(filter.matches(e.index()));
    }

    // ---- With/Without filter tests ----

    #[derive(Debug, PartialEq)]
    struct Frozen;

    #[test]
    fn with_access_info_locks_component() {
        let info = <With<Position>>::access_info();
        assert_eq!(info.type_id, TypeId::of::<Position>());
        assert_eq!(info.kind, AccessKind::ComponentFilter);
        assert!(!info.is_write);
    }

    #[test]
    fn without_access_info_locks_component() {
        let info = <Without<Position>>::access_info();
        assert_eq!(info.type_id, TypeId::of::<Position>());
        assert_eq!(info.kind, AccessKind::ComponentFilter);
        assert!(!info.is_write);
    }

    #[test]
    fn with_filter_in_tuple() {
        let mut world = World::new();
        world.register_component::<Position>();
        world.register_component::<Frozen>();

        let e1 = world.spawn();
        let e2 = world.spawn();
        world.insert(e1, Position { x: 1.0 }).unwrap();
        world.insert(e1, Frozen).unwrap();
        world.insert(e2, Position { x: 2.0 }).unwrap();

        let (positions, has_frozen) =
            <(Read<Position>, With<Frozen>)>::fetch(&world, FetchTicks::frame(&world));
        let matched: Vec<f32> = positions
            .iter()
            .filter(|(idx, _)| has_frozen.matches(*idx))
            .map(|(_, p)| p.x)
            .collect();
        assert_eq!(matched, vec![1.0]);
    }

    #[test]
    fn without_filter_in_tuple() {
        let mut world = World::new();
        world.register_component::<Position>();
        world.register_component::<Frozen>();

        let e1 = world.spawn();
        let e2 = world.spawn();
        world.insert(e1, Position { x: 1.0 }).unwrap();
        world.insert(e1, Frozen).unwrap();
        world.insert(e2, Position { x: 2.0 }).unwrap();

        let (positions, not_frozen) =
            <(Read<Position>, Without<Frozen>)>::fetch(&world, FetchTicks::frame(&world));
        let matched: Vec<f32> = positions
            .iter()
            .filter(|(idx, _)| not_frozen.matches(*idx))
            .map(|(_, p)| p.x)
            .collect();
        assert_eq!(matched, vec![2.0]);
    }

    #[test]
    fn without_unregistered_matches_everything() {
        let mut world = World::new();
        world.register_component::<Position>();

        let e = world.spawn();
        world.insert(e, Position { x: 1.0 }).unwrap();

        // Frozen never registered — Without<Frozen> matches all entities
        let (positions, not_frozen) =
            <(Read<Position>, Without<Frozen>)>::fetch(&world, FetchTicks::frame(&world));
        let count = positions
            .iter()
            .filter(|(idx, _)| not_frozen.matches(*idx))
            .count();
        assert_eq!(count, 1);
    }

    #[test]
    fn with_unregistered_matches_nothing() {
        let mut world = World::new();
        world.register_component::<Position>();

        let e = world.spawn();
        world.insert(e, Position { x: 1.0 }).unwrap();

        // Frozen never registered — With<Frozen> matches no entities
        let (positions, has_frozen) =
            <(Read<Position>, With<Frozen>)>::fetch(&world, FetchTicks::frame(&world));
        let count = positions
            .iter()
            .filter(|(idx, _)| has_frozen.matches(*idx))
            .count();
        assert_eq!(count, 0);
    }

    // ---- Or<A, B> AccessElement tests ----

    #[derive(Debug, PartialEq)]
    struct Flying;
    #[derive(Debug, PartialEq)]
    struct Swimming;
    #[derive(Debug, PartialEq)]
    struct Walking;

    #[test]
    fn or_access_info_uses_marker_type() {
        let info = <Or<With<Flying>, With<Swimming>>>::access_info();
        assert_eq!(
            info.type_id,
            TypeId::of::<Or<With<Flying>, With<Swimming>>>()
        );
        assert!(!info.is_write);
    }

    #[test]
    fn or_filter_in_tuple_matches_first() {
        let mut world = World::new();
        world.register_component::<Position>();
        world.register_component::<Flying>();
        world.register_component::<Swimming>();

        let e1 = world.spawn();
        let e2 = world.spawn();
        let e3 = world.spawn();
        world.insert(e1, Position { x: 1.0 }).unwrap();
        world.insert(e1, Flying).unwrap();
        world.insert(e2, Position { x: 2.0 }).unwrap();
        world.insert(e2, Swimming).unwrap();
        world.insert(e3, Position { x: 3.0 }).unwrap();
        // e3 has neither Flying nor Swimming

        let (positions, can_move) = <(Read<Position>, Or<With<Flying>, With<Swimming>>)>::fetch(
            &world,
            FetchTicks::frame(&world),
        );
        let mut matched: Vec<f32> = positions
            .iter()
            .filter(|(idx, _)| can_move.matches(*idx))
            .map(|(_, p)| p.x)
            .collect();
        matched.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(matched, vec![1.0, 2.0]);
    }

    #[test]
    fn or_filter_rejects_neither() {
        let mut world = World::new();
        world.register_component::<Position>();
        world.register_component::<Flying>();
        world.register_component::<Swimming>();

        let e = world.spawn();
        world.insert(e, Position { x: 1.0 }).unwrap();

        let (positions, can_move) = <(Read<Position>, Or<With<Flying>, With<Swimming>>)>::fetch(
            &world,
            FetchTicks::frame(&world),
        );
        let count = positions
            .iter()
            .filter(|(idx, _)| can_move.matches(*idx))
            .count();
        assert_eq!(count, 0);
    }

    #[test]
    fn or_with_without_combination() {
        let mut world = World::new();
        world.register_component::<Position>();
        world.register_component::<Flying>();
        world.register_component::<Frozen>();

        let e1 = world.spawn();
        let e2 = world.spawn();
        let e3 = world.spawn();
        world.insert(e1, Position { x: 1.0 }).unwrap();
        world.insert(e1, Flying).unwrap(); // has Flying → matches With<Flying>
        world.insert(e2, Position { x: 2.0 }).unwrap();
        // e2: no Flying, no Frozen → matches Without<Frozen>
        world.insert(e3, Position { x: 3.0 }).unwrap();
        world.insert(e3, Frozen).unwrap();
        // e3: no Flying, has Frozen → matches neither

        let (positions, filter) = <(Read<Position>, Or<With<Flying>, Without<Frozen>>)>::fetch(
            &world,
            FetchTicks::frame(&world),
        );
        let mut matched: Vec<f32> = positions
            .iter()
            .filter(|(idx, _)| filter.matches(*idx))
            .map(|(_, p)| p.x)
            .collect();
        matched.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(matched, vec![1.0, 2.0]);
    }

    #[test]
    fn nested_or() {
        let mut world = World::new();
        world.register_component::<Position>();
        world.register_component::<Flying>();
        world.register_component::<Swimming>();
        world.register_component::<Walking>();

        let e1 = world.spawn();
        let e2 = world.spawn();
        let e3 = world.spawn();
        let e4 = world.spawn();
        world.insert(e1, Position { x: 1.0 }).unwrap();
        world.insert(e1, Flying).unwrap();
        world.insert(e2, Position { x: 2.0 }).unwrap();
        world.insert(e2, Swimming).unwrap();
        world.insert(e3, Position { x: 3.0 }).unwrap();
        world.insert(e3, Walking).unwrap();
        world.insert(e4, Position { x: 4.0 }).unwrap();
        // e4 has none

        let (positions, filter) = <(
            Read<Position>,
            Or<With<Flying>, Or<With<Swimming>, With<Walking>>>,
        )>::fetch(&world, FetchTicks::frame(&world));
        let mut matched: Vec<f32> = positions
            .iter()
            .filter(|(idx, _)| filter.matches(*idx))
            .map(|(_, p)| p.x)
            .collect();
        matched.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(matched, vec![1.0, 2.0, 3.0]);
    }

    // ---- Any<(...)> AccessElement tests ----

    #[test]
    fn any_access_info_uses_marker_type() {
        let info = <Any<(With<Flying>, With<Swimming>, With<Walking>)>>::access_info();
        assert_eq!(
            info.type_id,
            TypeId::of::<Any<(With<Flying>, With<Swimming>, With<Walking>)>>()
        );
        assert!(!info.is_write);
    }

    #[test]
    fn any_filter_in_tuple() {
        let mut world = World::new();
        world.register_component::<Position>();
        world.register_component::<Flying>();
        world.register_component::<Swimming>();
        world.register_component::<Walking>();

        let e1 = world.spawn();
        let e2 = world.spawn();
        let e3 = world.spawn();
        let e4 = world.spawn();
        world.insert(e1, Position { x: 1.0 }).unwrap();
        world.insert(e1, Flying).unwrap();
        world.insert(e2, Position { x: 2.0 }).unwrap();
        world.insert(e2, Swimming).unwrap();
        world.insert(e3, Position { x: 3.0 }).unwrap();
        world.insert(e3, Walking).unwrap();
        world.insert(e4, Position { x: 4.0 }).unwrap();
        // e4 has none of the movement components

        let (positions, movable) = <(
            Read<Position>,
            Any<(With<Flying>, With<Swimming>, With<Walking>)>,
        )>::fetch(&world, FetchTicks::frame(&world));
        let mut matched: Vec<f32> = positions
            .iter()
            .filter(|(idx, _)| movable.matches(*idx))
            .map(|(_, p)| p.x)
            .collect();
        matched.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(matched, vec![1.0, 2.0, 3.0]);
    }

    #[test]
    fn any_filter_rejects_all() {
        let mut world = World::new();
        world.register_component::<Position>();
        world.register_component::<Flying>();
        world.register_component::<Swimming>();

        let e = world.spawn();
        world.insert(e, Position { x: 1.0 }).unwrap();

        let (positions, movable) = <(Read<Position>, Any<(With<Flying>, With<Swimming>)>)>::fetch(
            &world,
            FetchTicks::frame(&world),
        );
        let count = positions
            .iter()
            .filter(|(idx, _)| movable.matches(*idx))
            .count();
        assert_eq!(count, 0);
    }
}
