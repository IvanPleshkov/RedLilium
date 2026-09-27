// Shared tests inside both world modules, including backend-only scene geometry.
macro_rules! spatial_query_tests {
    () => {
        use crate::physics::{
            PhysicsQueryError as QueryError, RayCastOptions, ShapeCastOptions as SweepOptions,
            ShapeCastStatus as SweepStatus,
        };
        use std::ops::ControlFlow;

        fn gv(x: f32, y: f32) -> GameVector {
            let mut p = GameVector::zeros();
            p.x = x;
            p.y = y;
            p
        }
        fn query_fixture() -> (Physics, ColliderHandle) {
            let mut w = Physics::default();
            let h = w.add_free_collider(
                ColliderBuilder::ball(1.0)
                    .translation(Vector::X * 5.0)
                    .build(),
            );
            w.step();
            (w, h)
        }

        #[test]
        fn query_ray_fraction_point_normal_and_segment_bounds() {
            let (w, h) = query_fixture();
            for length in [4.0, 5.0, 10.0] {
                let hit = w
                    .cast_ray(
                        gv(0.0, 0.0),
                        gv(length, 0.0),
                        RayCastOptions::default(),
                        QueryFilter::default(),
                    )
                    .unwrap()
                    .unwrap();
                assert_eq!(hit.target.collider, h);
                assert!((hit.fraction - 4.0 / length).abs() < 1e-5);
                assert!((hit.point - gv(4.0, 0.0)).norm() < 1e-5);
                assert!((hit.normal.unwrap() - gv(-1.0, 0.0)).norm() < 1e-5);
            }
            for delta in [gv(3.9, 0.0), gv(-10.0, 0.0), gv(0.0, 10.0)] {
                assert!(
                    w.cast_ray(
                        gv(0.0, 0.0),
                        delta,
                        RayCastOptions::default(),
                        QueryFilter::default()
                    )
                    .unwrap()
                    .is_none()
                );
            }
        }

        #[test]
        fn query_ray_interior_and_exit_normals_are_consistent_across_shapes() {
            for shape in [
                SharedShape::ball(1.0),
                SharedShape::new(Cuboid::new(Vector::splat(1.0))),
            ] {
                let mut w = Physics::default();
                w.add_free_collider(
                    ColliderBuilder::new(shape)
                        .translation(Vector::X * 5.0)
                        .build(),
                );
                w.step();
                let inside = w
                    .cast_ray(
                        gv(5.0, 0.0),
                        gv(2.0, 0.0),
                        RayCastOptions::default(),
                        QueryFilter::default(),
                    )
                    .unwrap()
                    .unwrap();
                assert_eq!(inside.fraction, 0.0);
                assert_eq!(inside.point, gv(5.0, 0.0));
                assert_eq!(inside.normal, None);
                for sign in [-1.0, 1.0] {
                    let exit = w
                        .cast_ray(
                            gv(5.0, 0.0),
                            gv(2.0 * sign, 0.0),
                            RayCastOptions { solid: false },
                            QueryFilter::default(),
                        )
                        .unwrap()
                        .unwrap();
                    assert!((exit.fraction - 0.5).abs() < 1e-5);
                    assert!((exit.point - gv(5.0 + sign, 0.0)).norm() < 1e-5);
                    assert!((exit.normal.unwrap() - gv(sign, 0.0)).norm() < 1e-5);
                }
            }
        }

        #[test]
        fn query_shape_world_witnesses_include_rotation_translation_and_hit_fraction() {
            let mut w = Physics::default();
            w.add_free_collider(
                ColliderBuilder::ball(1.0)
                    .translation(Vector::X * 5.0 + Vector::Y * 3.0)
                    .build(),
            );
            w.step();
            let shape = ShapeDesc::CapsuleY {
                half_height: 1.0,
                radius: 0.25,
            };
            let pose = query_pose(gv(0.0, 3.0), std::f32::consts::FRAC_PI_2);
            let hit = w
                .cast_shape(
                    &shape,
                    pose,
                    gv(10.0, 0.0),
                    SweepOptions::default(),
                    QueryFilter::default(),
                )
                .unwrap()
                .unwrap();
            assert!((hit.fraction - 0.275).abs() < 2e-4, "{hit:?}");
            assert_eq!(hit.status, SweepStatus::Converged);
            let g = hit.geometry.unwrap();
            assert!((g.point_on_collider - gv(4.0, 3.0)).norm() < 2e-3, "{g:?}");
            assert!((g.point_on_shape - gv(4.0, 3.0)).norm() < 2e-3, "{g:?}");
            assert!((g.normal - gv(-1.0, 0.0)).norm() < 2e-3);
        }

        #[test]
        fn query_shape_clearance_and_initial_contact_policy() {
            let (w, _) = query_fixture();
            let shape = ShapeDesc::Ball { radius: 0.5 };
            let hit = w
                .cast_shape(
                    &shape,
                    query_pose(gv(0.0, 0.0), 0.0),
                    gv(10.0, 0.0),
                    SweepOptions {
                        target_distance: 0.5,
                        ..Default::default()
                    },
                    QueryFilter::default(),
                )
                .unwrap()
                .unwrap();
            assert!((hit.fraction - 0.3).abs() < 1e-5);
            let g = hit.geometry.unwrap();
            assert!((g.point_on_collider - gv(4.0, 0.0)).norm() < 1e-5);
            assert!((g.point_on_shape - gv(3.5, 0.0)).norm() < 1e-5);
            for clearance_only in [false, true] {
                let x = if clearance_only { 3.4 } else { 4.0 };
                let options = SweepOptions {
                    target_distance: 0.2,
                    stop_at_penetration: true,
                };
                let hit = w
                    .cast_shape(
                        &shape,
                        query_pose(gv(x, 0.0), 0.0),
                        gv(-2.0, 0.0),
                        options,
                        QueryFilter::default(),
                    )
                    .unwrap()
                    .unwrap();
                assert_eq!(hit.fraction, 0.0);
                assert_eq!(hit.status, SweepStatus::InitialContact);
                let g = hit.geometry.unwrap();
                assert!((g.point_on_collider - gv(4.0, 0.0)).norm() < 1e-4, "{g:?}");
                assert!((g.point_on_shape - gv(x + 0.5, 0.0)).norm() < 1e-4, "{g:?}");
                let options = SweepOptions {
                    stop_at_penetration: false,
                    ..options
                };
                assert!(
                    w.cast_shape(
                        &shape,
                        query_pose(gv(x, 0.0), 0.0),
                        gv(-2.0, 0.0),
                        options,
                        QueryFilter::default()
                    )
                    .unwrap()
                    .is_none()
                );
                assert!(
                    w.cast_shape(
                        &shape,
                        query_pose(gv(x, 0.0), 0.0),
                        gv(2.0, 0.0),
                        options,
                        QueryFilter::default()
                    )
                    .unwrap()
                    .is_some()
                );
            }
        }

        #[test]
        fn query_filters_apply_before_selecting_shape_and_ray_hits() {
            let (mut w, far) = query_fixture();
            let sensor = w.add_free_collider(
                ColliderBuilder::ball(0.5)
                    .sensor(true)
                    .translation(Vector::X * 2.0)
                    .build(),
            );
            w.step();
            let shape = ShapeDesc::Ball { radius: 0.25 };
            for (filter, expected) in [
                (QueryFilter::default(), sensor),
                (QueryFilter::default().exclude_sensors(), far),
                (QueryFilter::default().exclude_collider(sensor), far),
            ] {
                let ray = w
                    .cast_ray(
                        gv(0.0, 0.0),
                        gv(10.0, 0.0),
                        RayCastOptions::default(),
                        filter,
                    )
                    .unwrap()
                    .unwrap();
                let sweep = w
                    .cast_shape(
                        &shape,
                        QueryPose::default(),
                        gv(10.0, 0.0),
                        SweepOptions::default(),
                        filter,
                    )
                    .unwrap()
                    .unwrap();
                assert_eq!(ray.target.collider, expected);
                assert_eq!(sweep.target.collider, expected);
            }
            let no_hit = |_: ColliderHandle, _: &Collider| false;
            assert!(
                w.cast_shape(
                    &shape,
                    QueryPose::default(),
                    gv(10.0, 0.0),
                    SweepOptions::default(),
                    QueryFilter::default().predicate(&no_hit)
                )
                .unwrap()
                .is_none()
            );
        }

        #[test]
        fn query_overlaps_are_exact_deduplicate_compounds_and_allow_early_exit() {
            let mut w = Physics::default();
            // AABBs overlap, but the actual circles/spheres do not.
            let false_positive = w.add_free_collider(
                ColliderBuilder::ball(1.0)
                    .translation((Vector::X + Vector::Y) * 1.8)
                    .build(),
            );
            let compound = w.add_free_collider(
                ColliderBuilder::compound(vec![
                    (
                        Pose::from_translation(Vector::X * 0.2),
                        SharedShape::ball(0.5),
                    ),
                    (
                        Pose::from_translation(-Vector::X * 0.2),
                        SharedShape::ball(0.5),
                    ),
                ])
                .build(),
            );
            let sensor = w.add_free_collider(ColliderBuilder::ball(0.5).sensor(true).build());
            w.step();
            let shape = ShapeDesc::Ball { radius: 1.0 };
            let mut found = Vec::new();
            let result = w
                .visit_overlaps(
                    &shape,
                    QueryPose::default(),
                    QueryFilter::default(),
                    |target| {
                        found.push(target);
                        ControlFlow::Continue(())
                    },
                )
                .unwrap();
            assert_eq!(result, ControlFlow::Continue(()));
            assert_eq!(found.len(), 2);
            assert_eq!(found.iter().filter(|t| t.collider == compound).count(), 1);
            assert!(found.iter().all(|t| t.collider != false_positive));
            assert!(found.iter().any(|t| t.collider == sensor));
            let mut visits = 0;
            assert_eq!(
                w.visit_overlaps(&shape, QueryPose::default(), QueryFilter::default(), |_| {
                    visits += 1;
                    ControlFlow::Break(())
                })
                .unwrap(),
                ControlFlow::Break(())
            );
            assert_eq!(visits, 1);
            found.clear();
            let _ = w
                .visit_overlaps(
                    &shape,
                    QueryPose::default(),
                    QueryFilter::default().exclude_sensors(),
                    |t| {
                        found.push(t);
                        ControlFlow::Continue(())
                    },
                )
                .unwrap();
            assert_eq!(found.len(), 1);
            assert_eq!(found[0].collider, compound);
        }

        #[test]
        fn query_validation_precedes_search_and_visitors_even_in_empty_world() {
            let w = Physics::default();
            let shape = ShapeDesc::Ball { radius: 0.5 };
            for v in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
                assert_eq!(
                    w.cast_ray(
                        gv(v, 0.0),
                        gv(1.0, 0.0),
                        RayCastOptions::default(),
                        QueryFilter::default()
                    ),
                    Err(QueryError::InvalidOrigin)
                );
                assert_eq!(
                    w.cast_ray(
                        gv(0.0, 0.0),
                        gv(v, 0.0),
                        RayCastOptions::default(),
                        QueryFilter::default()
                    ),
                    Err(QueryError::InvalidDisplacement)
                );
                assert_eq!(
                    w.cast_shape(
                        &shape,
                        query_pose(gv(v, 0.0), 0.0),
                        gv(1.0, 0.0),
                        SweepOptions::default(),
                        QueryFilter::default()
                    ),
                    Err(QueryError::InvalidPose)
                );
                assert_eq!(
                    w.visit_overlaps(
                        &shape,
                        query_pose(gv(0.0, 0.0), v),
                        QueryFilter::default(),
                        |_| panic!("invalid pose visited")
                    ),
                    Err(QueryError::InvalidPose)
                );
            }
            for delta in [gv(0.0, 0.0), gv(f32::NAN, 0.0)] {
                assert_eq!(
                    w.cast_shape(
                        &shape,
                        QueryPose::default(),
                        delta,
                        SweepOptions::default(),
                        QueryFilter::default()
                    ),
                    Err(QueryError::InvalidDisplacement)
                );
            }
            for radius in [0.0, -1.0, f32::NAN, f32::INFINITY] {
                let invalid = ShapeDesc::Ball { radius };
                assert_eq!(
                    w.cast_shape(
                        &invalid,
                        QueryPose::default(),
                        gv(1.0, 0.0),
                        SweepOptions::default(),
                        QueryFilter::default()
                    ),
                    Err(QueryError::InvalidShape)
                );
                assert_eq!(
                    w.visit_overlaps(
                        &invalid,
                        QueryPose::default(),
                        QueryFilter::default(),
                        |_| panic!("invalid shape visited")
                    ),
                    Err(QueryError::InvalidShape)
                );
            }
            for target_distance in [-1.0, f32::NAN, f32::INFINITY] {
                assert_eq!(
                    w.cast_shape(
                        &shape,
                        QueryPose::default(),
                        gv(1.0, 0.0),
                        SweepOptions {
                            target_distance,
                            ..Default::default()
                        },
                        QueryFilter::default()
                    ),
                    Err(QueryError::InvalidTargetDistance)
                );
            }
        }

        #[test]
        fn query_results_survive_removal_without_rebinding_to_recycled_handles() {
            let (mut w, handle) = query_fixture();
            let hit = w
                .cast_shape(
                    &ShapeDesc::Ball { radius: 0.5 },
                    QueryPose::default(),
                    gv(10.0, 0.0),
                    SweepOptions::default(),
                    QueryFilter::default(),
                )
                .unwrap()
                .unwrap();
            assert!(w.remove_free_collider(handle));
            let next = w.add_free_collider(ColliderBuilder::ball(1.0).build());
            assert_ne!(hit.target.collider, next);
            assert_eq!(hit.target.collider, handle);
            assert_eq!(hit.target.body, None);
            assert_eq!(hit.target.entity, None);
        }

        #[test]
        fn query_compound_ray_normal_belongs_to_the_hit_part() {
            let mut w = Physics::default();
            w.add_free_collider(
                ColliderBuilder::compound(vec![
                    (Pose::IDENTITY, SharedShape::ball(2.0)),
                    (
                        Pose::from_translation(Vector::X * 2.0),
                        SharedShape::ball(1.0),
                    ),
                ])
                .build(),
            );
            w.step();
            // Start inside the large ball. Enter the second ball before exiting
            // the first: its entry normal must not be mistaken for an exit normal.
            let hit = w
                .cast_ray(
                    gv(0.0, 0.0),
                    gv(5.0, 0.0),
                    RayCastOptions { solid: false },
                    QueryFilter::default(),
                )
                .unwrap()
                .unwrap();
            assert!((hit.point - gv(1.0, 0.0)).norm() < 1e-5);
            assert!((hit.normal.unwrap() - gv(-1.0, 0.0)).norm() < 1e-5);
        }

        #[test]
        fn query_all_primitives_and_shape_segment_endpoint() {
            let (w, collider) = query_fixture();
            let mut shapes = vec![
                ShapeDesc::Ball { radius: 0.5 },
                ShapeDesc::Cuboid {
                    half_extents: GameVector::repeat(0.5),
                },
                ShapeDesc::CapsuleY {
                    half_height: 0.5,
                    radius: 0.5,
                },
                ShapeDesc::CapsuleY {
                    half_height: 0.0,
                    radius: 0.5,
                },
            ];
            extra_query_shapes(&mut shapes);
            for shape in shapes {
                let hit = w
                    .cast_shape(
                        &shape,
                        QueryPose::default(),
                        gv(10.0, 0.0),
                        SweepOptions::default(),
                        QueryFilter::default(),
                    )
                    .unwrap()
                    .unwrap();
                assert_eq!(hit.target.collider, collider);
                assert!(hit.geometry.is_some());
                let mut count = 0;
                let _ = w
                    .visit_overlaps(
                        &shape,
                        query_pose(gv(5.0, 0.0), 0.0),
                        QueryFilter::default(),
                        |_| {
                            count += 1;
                            ControlFlow::Continue(())
                        },
                    )
                    .unwrap();
                assert_eq!(count, 1);
            }
            let shape = ShapeDesc::Ball { radius: 0.5 };
            let hit = w
                .cast_shape(
                    &shape,
                    QueryPose::default(),
                    gv(3.5, 0.0),
                    SweepOptions::default(),
                    QueryFilter::default(),
                )
                .unwrap()
                .unwrap();
            assert_eq!(hit.fraction, 1.0);
            assert!(
                w.cast_shape(
                    &shape,
                    QueryPose::default(),
                    gv(3.4, 0.0),
                    SweepOptions::default(),
                    QueryFilter::default()
                )
                .unwrap()
                .is_none()
            );
            for shape in [
                ShapeDesc::Cuboid {
                    half_extents: gv(0.0, 0.5),
                },
                ShapeDesc::Cuboid {
                    half_extents: GameVector::repeat(f32::INFINITY),
                },
                ShapeDesc::CapsuleY {
                    half_height: -0.1,
                    radius: 0.5,
                },
                ShapeDesc::CapsuleY {
                    half_height: f32::NAN,
                    radius: 0.5,
                },
            ] {
                assert_eq!(
                    w.visit_overlaps(
                        &shape,
                        QueryPose::default(),
                        QueryFilter::default(),
                        |_| panic!("invalid shape visited")
                    ),
                    Err(QueryError::InvalidShape)
                );
            }
        }

        #[test]
        fn query_sweep_compound_points_and_group_filtering() {
            let mut w = Physics::default();
            let c = w.add_free_collider(
                ColliderBuilder::compound(vec![
                    (
                        Pose::from_translation(Vector::X * 2.0),
                        SharedShape::ball(0.5),
                    ),
                    (
                        Pose::from_translation(-Vector::X * 2.0),
                        SharedShape::ball(0.5),
                    ),
                ])
                .translation(Vector::X * 5.0 + Vector::Y * 3.0)
                .collision_groups(crate::physics::CollisionGroups::new(2, 1).into())
                .build(),
            );
            w.step();
            let shape = ShapeDesc::Ball { radius: 0.5 };
            let pose = query_pose(gv(0.0, 3.0), 0.0);
            let allowed =
                QueryFilter::default().groups(crate::physics::CollisionGroups::new(1, 2).into());
            let hit = w
                .cast_shape(
                    &shape,
                    pose,
                    gv(10.0, 0.0),
                    SweepOptions::default(),
                    allowed,
                )
                .unwrap()
                .unwrap();
            assert_eq!(hit.target.collider, c);
            assert!((hit.fraction - 0.2).abs() < 1e-5);
            let g = hit.geometry.unwrap();
            assert!((g.point_on_collider - gv(2.5, 3.0)).norm() < 1e-5);
            assert!((g.point_on_shape - gv(2.5, 3.0)).norm() < 1e-5);
            let rejected =
                QueryFilter::default().groups(crate::physics::CollisionGroups::new(4, 2).into());
            assert!(
                w.cast_shape(
                    &shape,
                    pose,
                    gv(10.0, 0.0),
                    SweepOptions::default(),
                    rejected
                )
                .unwrap()
                .is_none()
            );
            let _ = w
                .visit_overlaps(&shape, query_pose(gv(3.0, 3.0), 0.0), rejected, |_| {
                    panic!("rejected group visited")
                })
                .unwrap();
        }
    };
}
pub(super) use spatial_query_tests;
