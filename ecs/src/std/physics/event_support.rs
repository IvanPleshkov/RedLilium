// One implementation instantiated with each Rapier dimension's concrete types.
macro_rules! collision_events {
    ($participant:ident, $event:ident) => {
        /// Identity and sensor role snapshot. Collision transitions retain the
        /// snapshot from the start of tracking; force events capture it each step.
        /// Handles and the full entity identity may already be dead when this is read.
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub struct $participant {
            pub collider: ColliderHandle,
            pub body: Option<RigidBodyHandle>,
            pub entity: Option<crate::Entity>,
            pub is_sensor: bool,
        }

        /// One observed pair transition, published by StepPhysics through Events<T>.
        /// Participants have a stable canonical order, with no initiator/receiver role.
        /// The step counter starts at 1 and belongs to this PhysicsWorld instance.
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub struct $event {
            pub step: u64,
            pub a: $participant,
            pub b: $participant,
            pub phase: super::CollisionPhase,
        }

        type PairKey = (ColliderHandle, ColliderHandle);
        fn key(a: ColliderHandle, b: ColliderHandle) -> PairKey {
            if a.into_raw_parts() <= b.into_raw_parts() {
                (a, b)
            } else {
                (b, a)
            }
        }
        fn enabled(c: &Collider) -> bool {
            c.active_events().contains(ActiveEvents::COLLISION_EVENTS)
        }

        fn allowed(a: &Collider, b: &Collider, bodies: &RigidBodySet) -> bool {
            let kind = |c: &Collider| {
                c.parent()
                    .and_then(|h| bodies.get(h))
                    .map_or(RigidBodyType::Fixed, |body| body.body_type())
            };
            let (ak, bk) = (kind(a), kind(b));
            a.collision_groups().test(b.collision_groups())
                && (a.active_collision_types().test(ak, bk)
                    || b.active_collision_types().test(ak, bk))
        }

        // Rapier's handler is Send + Sync and receives &self, including in parallel builds.
        #[derive(Default)]
        struct Collector(std::sync::Mutex<Vec<RapierEvent>>, ForceCollector);
        impl EventHandler for Collector {
            // PhysicsWorld keeps its required soft-body set empty.
            fn handle_soft_body_tear_event(&self, _: &SoftBodySet, _: &SoftBodyTearEvent) {}

            fn handle_collision_event(
                &self,
                _: &RigidBodySet,
                _: &ColliderSet,
                event: RapierEvent,
                _: Option<&ContactPair>,
            ) {
                self.0.lock().unwrap().push(event);
            }
            fn handle_contact_force_event(
                &self,
                _: Real,
                bodies: &RigidBodySet,
                _: &ColliderSet,
                pair: &ContactPair,
                _: Real,
            ) {
                self.1.collect(bodies, pair);
            }
        }

        #[derive(Default)]
        pub(super) struct EventState {
            step: u64,
            pub(super) forces: ForceState,
            // Regular sync allocates Rapier bodies before deferred handle publication.
            // Keep their identities available to events until publication or rollback.
            pub(super) pending_entities: std::collections::HashMap<RigidBodyHandle, crate::Entity>,
            enabled: std::collections::HashSet<ColliderHandle>,
            dirty: std::collections::HashSet<ColliderHandle>,
            active: std::collections::HashMap<PairKey, [$participant; 2]>,
            // Only observed pairs are indexed. Configuration/removal work is local
            // to the changed collider instead of scanning all contacts each step.
            adjacent: std::collections::HashMap<ColliderHandle, std::collections::HashSet<PairKey>>,
            collector: Collector,
            pub(super) pending: Vec<$event>,
        }

        impl EventState {
            pub(super) fn requires_queue(&self) -> bool {
                !self.enabled.is_empty() || !self.active.is_empty() || !self.pending.is_empty()
            }

            pub(super) fn collider_changed(&mut self, handle: ColliderHandle, collider: &Collider) {
                let was_enabled = self.enabled.contains(&handle);
                if enabled(collider) {
                    self.enabled.insert(handle);
                } else {
                    self.enabled.remove(&handle);
                }
                if was_enabled || enabled(collider) || self.adjacent.contains_key(&handle) {
                    self.dirty.insert(handle);
                }
            }

            pub(super) fn collider_removed(&mut self, handle: ColliderHandle) {
                self.forces.remove(handle);
                if self.enabled.remove(&handle) || self.adjacent.contains_key(&handle) {
                    self.dirty.insert(handle);
                }
            }

            fn stop(&mut self, pair: PairKey, reason: super::CollisionStopReason) {
                let Some([a, b]) = self.active.remove(&pair) else {
                    return;
                };
                for handle in [pair.0, pair.1] {
                    if let Some(pairs) = self.adjacent.get_mut(&handle) {
                        pairs.remove(&pair);
                        if pairs.is_empty() {
                            self.adjacent.remove(&handle);
                        }
                    }
                }
                self.pending.push($event {
                    step: self.step,
                    a,
                    b,
                    phase: super::CollisionPhase::Stopped(reason),
                });
            }

            fn start(
                &mut self,
                pair: PairKey,
                colliders: &ColliderSet,
                bodies: &RigidBodySet,
                entities: &std::collections::HashMap<RigidBodyHandle, crate::Entity>,
            ) {
                if self.active.contains_key(&pair) {
                    return;
                }
                let (Some(a), Some(b)) = (colliders.get(pair.0), colliders.get(pair.1)) else {
                    return;
                };
                if (!enabled(a) && !enabled(b)) || !allowed(a, b, bodies) {
                    return;
                }
                let capture = |handle, c: &Collider| $participant {
                    collider: handle,
                    body: c.parent(),
                    entity: c.parent().and_then(|h| {
                        entities
                            .get(&h)
                            .or_else(|| self.pending_entities.get(&h))
                            .copied()
                    }),
                    is_sensor: c.is_sensor(),
                };
                let a = capture(pair.0, a);
                let b = capture(pair.1, b);
                self.active.insert(pair, [a, b]);
                self.adjacent.entry(pair.0).or_default().insert(pair);
                self.adjacent.entry(pair.1).or_default().insert(pair);
                self.pending.push($event {
                    step: self.step,
                    a,
                    b,
                    phase: super::CollisionPhase::Started,
                });
            }

            pub(super) fn prepare(&mut self, colliders: &ColliderSet, bodies: &RigidBodySet) {
                self.step = self
                    .step
                    .checked_add(1)
                    .expect("physics step counter exhausted");
                let affected: std::collections::HashSet<_> = self
                    .dirty
                    .iter()
                    .filter_map(|h| self.adjacent.get(h))
                    .flatten()
                    .copied()
                    .collect();
                for pair in affected {
                    let old = self.active[&pair];
                    let reason = match (colliders.get(pair.0), colliders.get(pair.1)) {
                        (Some(a), Some(b)) if !enabled(a) && !enabled(b) => {
                            Some(super::CollisionStopReason::TrackingDisabled)
                        }
                        (Some(a), Some(b))
                            if a.is_sensor() != old[0].is_sensor
                                || b.is_sensor() != old[1].is_sensor =>
                        {
                            Some(super::CollisionStopReason::Reconfigured)
                        }
                        (Some(a), Some(b)) if !allowed(a, b, bodies) => {
                            Some(super::CollisionStopReason::FilteredOut)
                        }
                        (Some(_), Some(_)) => None,
                        _ => Some(super::CollisionStopReason::Removed),
                    };
                    if let Some(reason) = reason {
                        self.stop(pair, reason);
                    }
                }
            }

            pub(super) fn handler(&self) -> &dyn EventHandler {
                &self.collector
            }

            pub(super) fn finish(
                &mut self,
                dt: Real,
                colliders: &ColliderSet,
                bodies: &RigidBodySet,
                narrow: &NarrowPhase,
                entities: &std::collections::HashMap<RigidBodyHandle, crate::Entity>,
            ) {
                self.forces.finish(
                    &mut self.collector.1,
                    self.step,
                    dt,
                    colliders,
                    entities,
                    &self.pending_entities,
                );
                // Reuse the callback buffer; no world access, user callbacks or queue
                // mutation is performed inside Rapier's callback.
                let mut raw = std::mem::take(self.collector.0.get_mut().unwrap());
                for event in raw.drain(..) {
                    let pair = key(event.collider1(), event.collider2());
                    match event {
                        RapierEvent::Started(_, _, flags) => {
                            if let (Some(a), Some(b)) =
                                (colliders.get(pair.0), colliders.get(pair.1))
                                && (a.is_sensor() || b.is_sensor())
                                    == flags.contains(CollisionEventFlags::SENSOR)
                            {
                                self.start(pair, colliders, bodies, entities);
                            }
                        }
                        RapierEvent::Stopped(_, _, flags) => {
                            // A sensor-role edit may tear down an old graph edge after
                            // creating its replacement. Do not stop the replacement pair.
                            if let Some([a, b]) = self.active.get(&pair)
                                && (a.is_sensor || b.is_sensor)
                                    == flags.contains(CollisionEventFlags::SENSOR)
                            {
                                let reason = if flags.contains(CollisionEventFlags::REMOVED) {
                                    super::CollisionStopReason::Removed
                                } else {
                                    super::CollisionStopReason::Separated
                                };
                                self.stop(pair, reason);
                            }
                        }
                    }
                }
                *self.collector.0.get_mut().unwrap() = raw;

                // Enabling events inside an existing overlap need not produce a Rapier
                // callback. Query only the changed colliders' neighborhoods to seed it.
                let mut touching = std::collections::HashSet::new();
                for handle in self.dirty.drain() {
                    for pair in narrow.contact_pairs_with(handle) {
                        if pair.has_any_active_contact() {
                            touching.insert(key(pair.collider1, pair.collider2));
                        }
                    }
                    for (a, b, intersecting) in narrow.intersection_pairs_with(handle) {
                        if intersecting {
                            touching.insert(key(a, b));
                        }
                    }
                }
                for pair in touching {
                    self.start(pair, colliders, bodies, entities);
                }
            }
        }
    };
}
pub(super) use collision_events;
