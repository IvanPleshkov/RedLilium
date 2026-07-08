//! Type-identity primitive: [`QualifiedTypeId`] / [`SourceId`].
//!
//! Per the ADR-020 amendment (2026-07-06, `docs/DECISIONS.md` → *Type identity
//! is `QualifiedTypeId`, not bare `TypeId`*), the engine's conceptual identity
//! for a type is not bare [`TypeId`] but the pair of `TypeId` and the
//! **origin of the type's definition**, named by a [`SourceId`].
//!
//! This module is a **behavior-preserving plinth**: in a single-process build
//! everything registers under [`SourceId::HOST`], so `QualifiedTypeId` collapses
//! to bare `TypeId` and nothing observable changes. It exists so #45 (game
//! cdylib loading) can stamp non-`HOST` generations onto game-defined types
//! without also introducing the identity primitive in the same change.
//!
//! ## Not the storage key
//!
//! `QualifiedTypeId` is the conceptual identity and the boundary **guard**, not
//! the physical storage key. `World.components` and `Resources.entries` stay
//! keyed by bare `TypeId`: within a live process the load-time fail-fast
//! guarantees no two live sources share a `TypeId`, so bare `TypeId` is a
//! lossless proxy on the hot path. `source` is carried as registration metadata
//! and checked only at the boundaries (downcast guard, load-time conflict).

use std::any::TypeId;

/// Names the **origin of a type's definition** for [`QualifiedTypeId`].
///
/// - `SourceId::HOST` (= `SourceId(0)`) is the host / shared-engine-dylib. Every
///   type defined in the engine (and everything registered in a single-process
///   build) lives here.
/// - A non-zero value is a monotonic *generation index*, incremented once per
///   game-cdylib reload, naming one game generation. Assigning non-zero values
///   is #45's job; this crate only ever produces `HOST` in production code —
///   tests synthesize other values via
///   [`World::with_registration_source`](crate::World::with_registration_source).
#[derive(Copy, Clone, Eq, PartialEq, Ord, PartialOrd, Hash, Debug)]
pub struct SourceId(pub u32);

impl SourceId {
    /// The host / shared-engine-dylib source. All engine-defined types and all
    /// registrations in a single-process build resolve to this.
    pub const HOST: SourceId = SourceId(0);
}

impl Default for SourceId {
    fn default() -> Self {
        SourceId::HOST
    }
}

/// The engine's type-identity primitive: a [`TypeId`] qualified by the
/// [`SourceId`] of the artifact that *defined* the type.
///
/// Two distinct sources sharing one `TypeId` (a game type whose `TypeId`
/// collides with an engine type's) are **different** `QualifiedTypeId`s — the
/// distinction the downcast guard and the load-time fail-fast rely on.
///
/// There is deliberately **no** generic `of::<T>()` constructor: a generic call
/// site cannot know where `T` was defined (ADR variant B), so the source is
/// supplied by the registration path, never inferred. Only [`host`](Self::host)
/// is offered for the one case where the origin is statically known: the host.
#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct QualifiedTypeId {
    /// The bare runtime type identity.
    pub type_id: TypeId,
    /// The origin that defined the type.
    pub source: SourceId,
}

impl QualifiedTypeId {
    /// Qualifies `T` under [`SourceId::HOST`] — the identity of any
    /// engine-defined type, and of every type in a single-process build.
    pub fn host<T: 'static>() -> Self {
        Self {
            type_id: TypeId::of::<T>(),
            source: SourceId::HOST,
        }
    }

    /// Qualifies a bare `TypeId` under an explicit source. Used by the
    /// registration/resolution paths that already know the origin; not exposed
    /// to generic call sites (see the type-level note).
    pub(crate) fn new(type_id: TypeId, source: SourceId) -> Self {
        Self { type_id, source }
    }
}

/// Downcast source-guard: refuse to reinterpret data whose recorded origin
/// (`entry`) differs from the source the type currently resolves to
/// (`expected`).
///
/// This is the seam the ADR calls "the real prize": downcasting generation-*N*
/// data with generation-*N+1*'s vtable is UB even when the `TypeId`
/// coincidentally matches. Today, every registration is `HOST`, so `entry ==
/// expected` always holds and this is a no-op; #45 makes it load-bearing across
/// a real reload. It is a `debug_assert` — a HOST-only build pays nothing in
/// release.
#[inline]
pub(crate) fn assert_source_match(entry: SourceId, expected: SourceId, type_name: &str) {
    debug_assert_eq!(
        entry, expected,
        "type-identity guard: `{type_name}` data was produced by source {entry:?} but the \
         live registration is {expected:?} — refusing to downcast across a stale generation"
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- §1: QualifiedTypeId / SourceId equality + hash ----

    struct A;
    struct B;

    #[test]
    fn host_is_zero() {
        assert_eq!(SourceId::HOST, SourceId(0));
        assert_eq!(SourceId::default(), SourceId::HOST);
    }

    #[test]
    fn same_type_distinct_sources_are_distinct_keys() {
        let host = QualifiedTypeId::host::<A>();
        let gen1 = QualifiedTypeId::new(TypeId::of::<A>(), SourceId(1));

        // Same bare TypeId, different source → different qualified identity.
        assert_eq!(host.type_id, gen1.type_id);
        assert_ne!(host, gen1);

        // And they hash to distinct buckets in a real map.
        use std::collections::HashMap;
        let mut map = HashMap::new();
        map.insert(host, "host");
        map.insert(gen1, "gen1");
        assert_eq!(map.len(), 2);
        assert_eq!(map[&host], "host");
        assert_eq!(map[&gen1], "gen1");
    }

    #[test]
    fn distinct_types_under_same_source_are_distinct() {
        assert_ne!(QualifiedTypeId::host::<A>(), QualifiedTypeId::host::<B>());
    }

    #[test]
    fn host_constructor_matches_manual() {
        assert_eq!(
            QualifiedTypeId::host::<A>(),
            QualifiedTypeId::new(TypeId::of::<A>(), SourceId::HOST)
        );
    }

    // ---- §3 (function level): the downcast guard fires on a synthetic mismatch ----

    #[test]
    fn guard_passes_for_matching_host() {
        // No panic when the recorded and expected sources agree.
        assert_source_match(SourceId::HOST, SourceId::HOST, "A");
    }

    // The guard is a `debug_assert`, compiled out in release — so this panic
    // only exists in debug builds. Gate the test to match, or it fails under
    // `cargo test --release`.
    #[test]
    #[cfg(debug_assertions)]
    #[should_panic(expected = "refusing to downcast across a stale generation")]
    fn guard_fires_on_source_mismatch() {
        // Synthetic generation-1 data reinterpreted under HOST.
        assert_source_match(SourceId(1), SourceId::HOST, "A");
    }

    // ---- World-integration tests (synthetic SourceId via with_registration_source) ----

    use crate::World;

    // A plain resource used to exercise the resource-side seam.
    struct Score(u32);
    // A component-like type; registered as a bare component (no inspector meta
    // needed for the identity paths).
    struct Marker;

    // §2: registering the same TypeId under a different source fails fast.
    #[test]
    #[should_panic(expected = "type-identity conflict")]
    fn conflicting_source_registration_panics() {
        let mut world = World::new();
        world.register_component::<Marker>(); // HOST
        // A second source claiming the same TypeId must be rejected loudly.
        world.with_registration_source(SourceId(1), |w| {
            w.register_component::<Marker>();
        });
    }

    // §2: re-registering under the SAME source stays idempotent.
    #[test]
    fn same_source_reregistration_is_idempotent() {
        let mut world = World::new();
        world.register_component::<Marker>();
        world.register_component::<Marker>(); // no panic
        assert_eq!(
            world.qualified_type_id_of::<Marker>(),
            Some(QualifiedTypeId::host::<Marker>())
        );
    }

    // §4: accessors report HOST for normally-registered types.
    #[test]
    fn qualified_accessors_report_host() {
        let mut world = World::new();
        world.register_component::<Marker>();
        world.insert_resource(Score(1));

        assert_eq!(
            world.qualified_type_id_of::<Marker>(),
            Some(QualifiedTypeId::host::<Marker>())
        );
        // host::<T>() equals the qualified id of a HOST-registered type.
        assert_eq!(
            world.qualified_type_id_of::<Score>(),
            Some(QualifiedTypeId::host::<Score>())
        );
        // Never-registered types resolve to None.
        assert_eq!(world.qualified_type_id_of::<A>(), None);

        let ids: Vec<_> = world.component_qualified_type_ids().collect();
        assert!(ids.contains(&QualifiedTypeId::host::<Marker>()));
        assert!(ids.iter().all(|q| q.source == SourceId::HOST));
    }

    // §4: a type registered under a synthetic source reports that source.
    #[test]
    fn synthetic_source_is_reported() {
        let mut world = World::new();
        world.with_registration_source(SourceId(7), |w| {
            w.register_component::<Marker>();
        });
        assert_eq!(
            world.qualified_type_id_of::<Marker>(),
            Some(QualifiedTypeId::new(TypeId::of::<Marker>(), SourceId(7)))
        );
        // The scope restored the default source.
        assert_eq!(world.current_source(), SourceId::HOST);
    }

    // A value insert of an already-declared resource resolves to the declared
    // origin and never conflicts — the warm-reload restore path (a snapshot
    // re-inserts a resource under a different scope than its declaration).
    #[test]
    fn value_insert_resolves_to_declared_origin_without_conflict() {
        let mut world = World::new();
        // Declare the resource's origin under a synthetic generation.
        world.with_registration_source(SourceId(3), |w| {
            w.insert_resource(Score(1));
        });
        // A later insert under HOST (as a snapshot restore would do) must not
        // panic and must respect the original origin, not re-stamp it HOST.
        world.insert_resource(Score(2));
        assert_eq!(
            world.qualified_type_id_of::<Score>(),
            Some(QualifiedTypeId::new(TypeId::of::<Score>(), SourceId(3)))
        );
        assert_eq!(world.resource::<Score>().0, 2);
    }

    // §3 (live path): the resource downcast guard passes under matching HOST...
    #[test]
    fn resource_guard_passes_under_host() {
        let mut world = World::new();
        world.insert_resource(Score(42));
        assert_eq!(world.resource::<Score>().0, 42); // no panic
    }

    // ...and fires when the entry's recorded source no longer matches the
    // source the type resolves to (a stale generation, forced via the test
    // seam since the fail-fast makes this unreachable in production).
    #[test]
    #[cfg(debug_assertions)] // guard is a debug_assert; see guard_fires_on_source_mismatch
    #[should_panic(expected = "refusing to downcast across a stale generation")]
    fn resource_guard_fires_on_stale_generation() {
        let mut world = World::new();
        world.insert_resource(Score(42)); // entry.source = HOST
        // Simulate the type being re-registered by a newer generation while the
        // old entry lingers.
        world.force_type_source_for_test::<Score>(SourceId(1));
        let _ = world.resource::<Score>(); // guard: entry HOST != expected gen-1
    }

    // ---- §5: Supersession on re-registration (Phase 2, R4) ----

    // Registering a type under a dead (inactive) generation should succeed,
    // overwriting the old source — this is the supersession case that allows
    // re-stamping types during a hot-reload cycle.
    #[test]
    fn supersession_overwrites_dead_generation() {
        use crate::sync::RwLock;
        use std::sync::Arc;

        let shared_registry = Arc::new(RwLock::new(crate::GameGenerationRegistry::new()));
        let mut world = World::new();
        world.insert_resource_shared(shared_registry.clone());

        // Allocate gen-1 and gen-2 properly.
        let gen1 = shared_registry.write().allocate_generation();
        let gen2 = shared_registry.write().allocate_generation();

        // Register Marker under generation 1.
        world.with_registration_source(gen1, |w| {
            w.register_component::<Marker>();
        });
        assert_eq!(
            world.qualified_type_id_of::<Marker>(),
            Some(QualifiedTypeId::new(TypeId::of::<Marker>(), gen1))
        );

        // Manually mark generation 1 as inactive by decrementing the count.
        shared_registry.write().decrement_registration_count(gen1.0);

        // Now register under generation 2. Supersession should succeed.
        world.with_registration_source(gen2, |w| {
            w.register_component::<Marker>();
        });

        // Supersession should have occurred: Marker is now under gen-2, not gen-1.
        assert_eq!(
            world.qualified_type_id_of::<Marker>(),
            Some(QualifiedTypeId::new(TypeId::of::<Marker>(), gen2))
        );
    }

    // If a generation is still active (has registrations), a conflicting
    // re-registration should panic — this is the fail-fast that prevents UB.
    #[test]
    #[should_panic(expected = "type-identity conflict")]
    fn active_generation_conflicts_on_reregistration() {
        use crate::sync::RwLock;
        use std::sync::Arc;

        let shared_registry = Arc::new(RwLock::new(crate::GameGenerationRegistry::new()));
        let mut world = World::new();
        world.insert_resource_shared(shared_registry.clone());

        // Register Marker under generation 1, leaving it active.
        world.with_registration_source(SourceId(1), |w| {
            w.register_component::<Marker>();
        });

        // Try to register under generation 2 — should panic because gen-1 is active.
        world.with_registration_source(SourceId(2), |w| {
            w.register_component::<Marker>(); // panic: gen-1 still active
        });
    }

    // Same-generation re-registration remains idempotent (no count increment).
    #[test]
    fn same_generation_registration_is_idempotent() {
        use crate::sync::RwLock;
        use std::sync::Arc;

        let shared_registry = Arc::new(RwLock::new(crate::GameGenerationRegistry::new()));
        let mut world = World::new();
        world.insert_resource_shared(shared_registry.clone());

        world.register_component::<Marker>();
        let initial_count = shared_registry.read().registration_count(0); // HOST = gen-0
        world.register_component::<Marker>(); // re-register under HOST
        let after_count = shared_registry.read().registration_count(0);
        assert_eq!(initial_count, after_count); // count should not increase
    }
}
