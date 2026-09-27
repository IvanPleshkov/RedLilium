// Instantiated inside each dimension's event module alongside collision_events!.
macro_rules! contact_force_events {
    ($participant:ident, $event:ident, $sample:ident) => {
        /// Strongest individual normal-impulse contribution across the step's
        /// contacts and CCD substeps. This is a representative point, not a manifold.
        #[derive(Debug, Clone, Copy, PartialEq)]
        pub struct $sample {
            /// Midpoint of the solver's two world-space surface points at collection.
            pub point: EventVector,
            /// World-space unit normal from event participant `a` toward `b`.
            pub normal: EventVector,
            pub normal_impulse: f64,
        }

        /// One pair's normal contact load over one complete physics step.
        /// Independent of collision transition tracking. Sleeping pairs and sensors
        /// produce no events. Participants are snapshots in canonical handle order.
        /// Scalars use f64 for both physics precisions; points/normals use engine math.
        #[derive(Debug, Clone, Copy, PartialEq)]
        pub struct $event {
            /// Same PhysicsWorld counter as collision events, starting at 1.
            pub step: u64,
            pub a: $participant,
            pub b: $participant,
            /// Complete physics step duration in seconds, including CCD substeps.
            pub dt: f64,
            /// Sum of normal impulse magnitudes; excludes friction/tangent impulses.
            pub normal_impulse: f64,
            pub strongest_contact: $sample,
        }

        impl $event {
            /// Mean normal force magnitude over the full step (not peak force).
            pub fn average_force(&self) -> f64 {
                self.normal_impulse / self.dt
            }
        }

        #[derive(Clone, Copy)]
        struct ForceSum {
            normal_impulse: f64,
            strongest_contact: $sample,
        }
        impl ForceSum {
            fn merge(&mut self, other: Self) {
                self.normal_impulse += other.normal_impulse;
                if other.strongest_contact.normal_impulse > self.strongest_contact.normal_impulse {
                    self.strongest_contact = other.strongest_contact;
                }
            }
        }

        #[derive(Default)]
        struct ForceCollector(std::sync::Mutex<std::collections::HashMap<PairKey, ForceSum>>);
        impl ForceCollector {
            fn collect(&self, bodies: &RigidBodySet, pair: &ContactPair) {
                let pair_key = key(pair.collider1, pair.collider2);
                let mut sum: Option<ForceSum> = None;
                for manifold in pair.solver_manifolds() {
                    for contact in &manifold.data.solver_contacts {
                        let id = (contact.contact_id[0] & !NEW_CONTACT_BIT) as usize;
                        let impulse = manifold.contacts()[id].data.impulse as f64;
                        if impulse <= 0.0 {
                            continue;
                        }
                        let (p1, p2) = manifold.data.solver_contact_world_points(contact, bodies);
                        let normal = if pair_key.0 == pair.collider1 {
                            manifold.data.normal
                        } else {
                            -manifold.data.normal
                        };
                        let contribution = ForceSum {
                            normal_impulse: impulse,
                            strongest_contact: $sample {
                                point: event_vector((p1 + p2) * 0.5),
                                normal: event_vector(normal),
                                normal_impulse: impulse,
                            },
                        };
                        if let Some(sum) = &mut sum {
                            sum.merge(contribution);
                        } else {
                            sum = Some(contribution);
                        }
                    }
                }
                if let Some(sum) = sum {
                    self.0
                        .lock()
                        .unwrap()
                        .entry(pair_key)
                        .and_modify(|total| total.merge(sum))
                        .or_insert(sum);
                }
            }
        }

        #[derive(Default)]
        pub(super) struct ForceState {
            thresholds: std::collections::HashMap<ColliderHandle, f64>,
            pub(super) pending: Vec<$event>,
        }
        impl ForceState {
            /// Capture the requested threshold before replacing it with zero for
            /// Rapier. All substep contributions are needed for the full-step mean.
            /// Call only with the authored threshold, never with the live zero.
            pub(super) fn configure(&mut self, handle: ColliderHandle, collider: &mut Collider) {
                if collider
                    .active_events()
                    .contains(ActiveEvents::CONTACT_FORCE_EVENTS)
                {
                    self.thresholds
                        .insert(handle, collider.contact_force_event_threshold() as f64);
                } else {
                    self.thresholds.remove(&handle);
                }
                collider.set_contact_force_event_threshold(0.0);
            }
            pub(super) fn remove(&mut self, handle: ColliderHandle) {
                self.thresholds.remove(&handle);
            }
            pub(super) fn requires_queue(&self) -> bool {
                !self.thresholds.is_empty() || !self.pending.is_empty()
            }
            fn finish(
                &mut self,
                collector: &mut ForceCollector,
                step: u64,
                dt: Real,
                colliders: &ColliderSet,
                entities: &std::collections::HashMap<RigidBodyHandle, crate::Entity>,
                pending_entities: &std::collections::HashMap<RigidBodyHandle, crate::Entity>,
            ) {
                let dt = dt as f64;
                for (pair, sum) in collector.0.get_mut().unwrap().drain() {
                    if !(dt > 0.0 && dt.is_finite())
                        || ![pair.0, pair.1].iter().any(|h| {
                            self.thresholds
                                .get(h)
                                .is_some_and(|min| sum.normal_impulse / dt > *min)
                        })
                    {
                        continue;
                    }
                    let capture = |handle| {
                        let c = colliders.get(handle)?;
                        Some($participant {
                            collider: handle,
                            body: c.parent(),
                            entity: c.parent().and_then(|h| {
                                entities
                                    .get(&h)
                                    .or_else(|| pending_entities.get(&h))
                                    .copied()
                            }),
                            is_sensor: c.is_sensor(),
                        })
                    };
                    if let (Some(a), Some(b)) = (capture(pair.0), capture(pair.1)) {
                        self.pending.push($event {
                            step,
                            a,
                            b,
                            dt,
                            normal_impulse: sum.normal_impulse,
                            strongest_contact: sum.strongest_contact,
                        });
                    }
                }
            }
        }

        #[cfg(test)]
        mod force_tests {
            use super::*;

            // Exercise actual Rapier CCD callbacks, including a supporting contact
            // that contributes an impulse during more than one CCD substep.
            #[test]
            fn ccd_aggregates_all_contributions_before_applying_the_full_step_threshold() {
                struct Handler {
                    pair: PairKey,
                    forces: ForceCollector,
                    samples: std::sync::Mutex<Vec<(Real, Real)>>,
                }
                impl EventHandler for Handler {
                    fn handle_soft_body_tear_event(&self, _: &SoftBodySet, _: &SoftBodyTearEvent) {}
                    fn handle_collision_event(
                        &self,
                        _: &RigidBodySet,
                        _: &ColliderSet,
                        _: RapierEvent,
                        _: Option<&ContactPair>,
                    ) {
                    }
                    fn handle_contact_force_event(
                        &self,
                        dt: Real,
                        bodies: &RigidBodySet,
                        _: &ColliderSet,
                        pair: &ContactPair,
                        _: Real,
                    ) {
                        if key(pair.collider1, pair.collider2) == self.pair {
                            self.samples
                                .lock()
                                .unwrap()
                                .push((dt, pair.total_impulse_magnitude()));
                            self.forces.collect(bodies, pair);
                        }
                    }
                }
                let mut bodies = RigidBodySet::new();
                let mut colliders = ColliderSet::new();
                let fixed = colliders.insert(
                    ColliderBuilder::ball(0.5)
                        .active_events(ActiveEvents::CONTACT_FORCE_EVENTS)
                        .contact_force_event_threshold(Real::MAX)
                        .build(),
                );
                let mut pos = Vector::ZERO;
                pos.x = 0.99;
                let body = bodies.insert(RigidBodyBuilder::dynamic().translation(pos).build());
                let moving = colliders.insert_with_parent(
                    ColliderBuilder::ball(0.5).build(),
                    body,
                    &mut bodies,
                );
                let mass = bodies[body].mass() as f64;
                // A distant fast ball forces CCD subdivision without touching the
                // supporting pair whose gravity impulse we measure.
                pos.x = 0.0;
                pos.y = 5.0;
                colliders.insert(ColliderBuilder::ball(0.2).translation(pos).build());
                pos.x = -1.7;
                let mut vel = Vector::ZERO;
                vel.x = 240.0;
                let bullet = bodies.insert(
                    RigidBodyBuilder::dynamic()
                        .translation(pos)
                        .linvel(vel)
                        .ccd_enabled(true)
                        .build(),
                );
                colliders.insert_with_parent(
                    ColliderBuilder::ball(0.05).build(),
                    bullet,
                    &mut bodies,
                );
                let mut state = ForceState::default();
                state.configure(fixed, &mut colliders[fixed]);
                let mut handler = Handler {
                    pair: key(fixed, moving),
                    forces: ForceCollector::default(),
                    samples: Default::default(),
                };
                let params = IntegrationParameters {
                    dt: 1.0 / 60.0,
                    max_ccd_substeps: 4,
                    ..Default::default()
                };
                let mut gravity = Vector::ZERO;
                gravity.x = -10.0;
                PhysicsPipeline::new().step(
                    gravity,
                    &params,
                    &mut IslandManager::new(),
                    &mut BroadPhaseBvh::new(),
                    &mut NarrowPhase::new(),
                    &mut bodies,
                    &mut colliders,
                    &mut ImpulseJointSet::new(),
                    &mut MultibodyJointSet::new(),
                    &mut SoftBodySet::new(),
                    &mut CCDSolver::new(),
                    &(),
                    &handler,
                );
                let samples = handler.samples.get_mut().unwrap();
                assert!(samples.len() >= 2, "fixture must subdivide: {samples:?}");
                let impulse: f64 = samples.iter().map(|(_, impulse)| *impulse as f64).sum();
                let expected = mass * (bodies[body].linvel().x as f64 + 10.0 * params.dt as f64);
                assert!((impulse - expected).abs() < 1e-4, "{impulse} != {expected}");
                let strongest = samples
                    .iter()
                    .map(|(_, impulse)| *impulse as f64)
                    .fold(0.0, f64::max);
                let sum = handler.forces.0.get_mut().unwrap()[&handler.pair];
                let entities = Default::default();
                // Equality is not sufficient. Threshold is per full step, not per
                // callback; no disabled collider's native zero can override it.
                state.thresholds.insert(fixed, impulse / params.dt as f64);
                state.finish(
                    &mut handler.forces,
                    1,
                    params.dt,
                    &colliders,
                    &entities,
                    &entities,
                );
                assert!(state.pending.is_empty());
                handler
                    .forces
                    .0
                    .get_mut()
                    .unwrap()
                    .insert(handler.pair, sum);
                state
                    .thresholds
                    .insert(fixed, impulse / params.dt as f64 * 0.99);
                state.finish(
                    &mut handler.forces,
                    1,
                    params.dt,
                    &colliders,
                    &entities,
                    &entities,
                );
                assert_eq!(state.pending.len(), 1);
                let event = state.pending[0];
                assert!((event.normal_impulse - impulse).abs() < 1e-8);
                assert!((event.strongest_contact.normal_impulse - strongest).abs() < 1e-8);
                assert_eq!(event.dt, params.dt as f64);
            }
        }
    };
}
pub(super) use contact_force_events;
