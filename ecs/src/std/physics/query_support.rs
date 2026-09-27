// Shared implementation instantiated with each dimension's concrete math types.
macro_rules! spatial_queries {
    ($target:ident, $ray_hit:ident, $shape_hit:ident, $geometry:ident) => {
        /// Collider identity captured by a query. The collider/body/entity may
        /// have been removed by the time the result is used. Free colliders have
        /// neither a body nor an entity; unmapped bodies can have no entity.
        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub struct $target {
            pub collider: ColliderHandle,
            pub body: Option<RigidBodyHandle>,
            pub entity: Option<crate::Entity>,
        }

        #[derive(Debug, Clone, Copy, PartialEq)]
        pub struct $ray_hit {
            pub target: $target,
            /// Fraction of the requested displacement, in [0, 1].
            pub fraction: f32,
            /// World-space origin + displacement * fraction. For an initial
            /// solid hit this is the origin, not necessarily a surface point.
            pub point: GameVector,
            /// Unit world-space normal, outward for closed shapes. Open surfaces
            /// retain the backend's hit-face orientation. None for an initial
            /// solid hit inside/on the collider or a degenerate surface normal.
            pub normal: Option<GameVector>,
        }

        #[derive(Debug, Clone, Copy, PartialEq)]
        pub struct $geometry {
            /// World-space point on the obstacle.
            pub point_on_collider: GameVector,
            /// World-space point on the moving query shape at the hit fraction.
            pub point_on_shape: GameVector,
            /// Unit world-space normal pointing out of the obstacle.
            pub normal: GameVector,
        }

        #[derive(Debug, Clone, Copy, PartialEq)]
        pub struct $shape_hit {
            pub target: $target,
            /// Fraction of the requested displacement, in [0, 1].
            pub fraction: f32,
            /// None if the backend cannot supply usable contact geometry.
            /// Approximate statuses retain their hit even without geometry.
            pub geometry: Option<$geometry>,
            pub status: ShapeCastStatus,
        }

        fn checked_point(v: Vector) -> Result<GameVector, PhysicsQueryError> {
            let p = to_game(v);
            if p.iter().all(|x| x.is_finite()) {
                Ok(p)
            } else {
                Err(PhysicsQueryError::InvalidResult)
            }
        }
        fn checked_fraction(t: Real) -> Result<f32, PhysicsQueryError> {
            if t.is_finite() && (0.0..=1.0).contains(&t) {
                Ok(t as f32)
            } else {
                Err(PhysicsQueryError::InvalidResult)
            }
        }
        fn normal(v: Vector) -> Option<GameVector> {
            v.try_normalize().and_then(|v| checked_point(v).ok())
        }
        fn displacement(
            origin: GameVector,
            delta: GameVector,
        ) -> Result<Vector, PhysicsQueryError> {
            if !delta.iter().all(|x| x.is_finite())
                || delta.iter().all(|x| *x == 0.0)
                || !(origin + delta).iter().all(|x| x.is_finite())
            {
                return Err(PhysicsQueryError::InvalidDisplacement);
            }
            Ok(to_vector(delta))
        }

        impl Physics {
            fn query_target(&self, handle: ColliderHandle, c: &Collider) -> $target {
                let body = c.parent();
                $target {
                    collider: handle,
                    body,
                    entity: body.and_then(|h| self.entity_for_body(h)),
                }
            }

            /// Nearest ray hit on the segment origin..origin+displacement.
            /// Filtering precedes nearest-hit selection. Zero/nonfinite displacement
            /// is an error. `solid = false` seeks the exit when starting inside.
            /// Hollow compounds report a boundary of a part, not the union's exit.
            /// Queries require StepPhysics after sync/teleports to see updated geometry.
            pub fn cast_ray(
                &self,
                origin: GameVector,
                delta: GameVector,
                options: RayCastOptions,
                filter: QueryFilter<'_>,
            ) -> Result<Option<$ray_hit>, PhysicsQueryError> {
                if !origin.iter().all(|x| x.is_finite()) {
                    return Err(PhysicsQueryError::InvalidOrigin);
                }
                let delta = displacement(origin, delta)?;
                let ray = Ray::new(to_vector(origin), delta);
                let pipeline = self.broad_phase.as_query_pipeline(
                    self.narrow_phase.query_dispatcher(),
                    &self.bodies,
                    &self.colliders,
                    filter,
                );
                // Backend best-first traversal uses a strict upper bound. Expand
                // by one representable value to include the segment endpoint,
                // then enforce the public closed interval on the actual hit.
                let Some((handle, hit)) =
                    pipeline.cast_ray_and_get_normal(&ray, (1.0 as Real).next_up(), options.solid)
                else {
                    return Ok(None);
                };
                if hit.time_of_impact.is_finite() && hit.time_of_impact > 1.0 {
                    return Ok(None);
                }
                let c = &self.colliders[handle];
                let fraction = checked_fraction(hit.time_of_impact)?;
                // Rapier may synthesize a normal opposite the ray for an interior
                // hit. That is not an actual surface normal, so do not expose it.
                let inside = (!options.solid || hit.time_of_impact == 0.0) && {
                    if let Some(compound) = c.shape().as_compound() {
                        // Hollow compound casts select a boundary of a part, not
                        // the union's exit. The origin may be inside a DIFFERENT
                        // overlapping part: classify the actual part that was hit.
                        compound
                            .shapes()
                            .get(hit.subshape as usize)
                            .is_some_and(|(pose, shape)| {
                                shape.contains_point(&(*c.position() * *pose), ray.origin)
                            })
                    } else {
                        c.shape().contains_point(c.position(), ray.origin)
                    }
                };
                let initial_solid = options.solid && hit.time_of_impact == 0.0 && inside;
                // Some backend shapes orient exit normals against the ray while
                // others already return an outward normal. Unify closed-shape exits.
                let outward = if inside && hit.normal.dot(delta) < 0.0 {
                    -hit.normal
                } else {
                    hit.normal
                };
                Ok(Some($ray_hit {
                    target: self.query_target(handle, c),
                    fraction,
                    point: checked_point(ray.origin + delta * hit.time_of_impact)?,
                    normal: if initial_solid { None } else { normal(outward) },
                }))
            }

            /// Nearest hit while translating a primitive through displacement.
            /// Rotation stays fixed; other colliders are tested at their current
            /// physical poses, without predicting their motion. No simulation runs.
            /// The query shape does not restrict the kinds of scene colliders hit.
            pub fn cast_shape(
                &self,
                shape: &ShapeDesc,
                pose: QueryPose,
                delta: GameVector,
                options: ShapeCastOptions,
                filter: QueryFilter<'_>,
            ) -> Result<Option<$shape_hit>, PhysicsQueryError> {
                let native_pose = checked_pose(pose)?;
                let delta = displacement(pose.translation, delta)?;
                if !options.target_distance.is_finite() || options.target_distance < 0.0 {
                    return Err(PhysicsQueryError::InvalidTargetDistance);
                }
                with_shape(shape, |shape| {
                    let pipeline = self.broad_phase.as_query_pipeline(
                        self.narrow_phase.query_dispatcher(),
                        &self.bodies,
                        &self.colliders,
                        filter,
                    );
                    let Some((handle, hit)) = pipeline.cast_shape(
                        &native_pose,
                        delta,
                        shape,
                        BackendOptions {
                            max_time_of_impact: (1.0 as Real).next_up(),
                            target_distance: options.target_distance as Real,
                            stop_at_penetration: options.stop_at_penetration,
                            compute_impact_geometry_on_penetration: true,
                        },
                    ) else {
                        return Ok(None);
                    };
                    if hit.time_of_impact.is_finite() && hit.time_of_impact > 1.0 {
                        return Ok(None);
                    }
                    let fraction = checked_fraction(hit.time_of_impact)?;
                    let status = match hit.status {
                        BackendStatus::Converged => ShapeCastStatus::Converged,
                        BackendStatus::PenetratingOrWithinTargetDist => {
                            ShapeCastStatus::InitialContact
                        }
                        BackendStatus::OutOfIterations => ShapeCastStatus::OutOfIterations,
                        BackendStatus::Failed => ShapeCastStatus::Failed,
                    };
                    // The obstacle witness is already in world space. The query
                    // witness is shape-local, and must follow the pose AT IMPACT.
                    let c = &self.colliders[handle];
                    let geometry = if status == ShapeCastStatus::InitialContact {
                        // Ball sweeps in Parry 0.31 return penetration-scaled
                        // witnesses even with compute_impact_geometry enabled.
                        // Resolve surface points instead of exposing those as contacts.
                        let mut impact_pose = native_pose;
                        impact_pose.translation += delta * hit.time_of_impact;
                        backend_contact(
                            c.position(),
                            c.shape(),
                            &impact_pose,
                            shape,
                            options.target_distance as Real,
                        )
                        .ok()
                        .flatten()
                        .and_then(|contact| {
                            Some($geometry {
                                point_on_collider: checked_point(contact.point1).ok()?,
                                point_on_shape: checked_point(contact.point2).ok()?,
                                normal: normal(contact.normal1)?,
                            })
                        })
                    } else {
                        (|| {
                            Some($geometry {
                                point_on_collider: checked_point(hit.witness1).ok()?,
                                point_on_shape: checked_point(
                                    native_pose * hit.witness2 + delta * hit.time_of_impact,
                                )
                                .ok()?,
                                normal: normal(hit.normal1)?,
                            })
                        })()
                    };
                    Ok(Some($shape_hit {
                        target: self.query_target(handle, c),
                        fraction,
                        geometry,
                        status,
                    }))
                })?
            }

            /// Visits exact shape intersections once per collider, in unspecified
            /// order. The visitor can break early or append into a caller-owned Vec.
            /// Returns whether traversal completed or the visitor broke. All input
            /// validation precedes the first visit. No result array is allocated.
            pub fn visit_overlaps(
                &self,
                shape: &ShapeDesc,
                pose: QueryPose,
                filter: QueryFilter<'_>,
                mut visitor: impl FnMut($target) -> ControlFlow<()>,
            ) -> Result<ControlFlow<()>, PhysicsQueryError> {
                let pose = checked_pose(pose)?;
                with_shape(shape, |shape| {
                    let pipeline = self.broad_phase.as_query_pipeline(
                        self.narrow_phase.query_dispatcher(),
                        &self.bodies,
                        &self.colliders,
                        filter,
                    );
                    for (handle, c) in pipeline.intersect_shape(pose, shape) {
                        if let ControlFlow::Break(()) = visitor(self.query_target(handle, c)) {
                            return ControlFlow::Break(());
                        }
                    }
                    ControlFlow::Continue(())
                })
            }
        }
    };
}
pub(super) use spatial_queries;
