// Shared tests run inside each world's private test module so raw Rapier fixtures
// can represent unmapped bodies without adding a public structural escape hatch.
macro_rules! world_boundary_tests {
    () => {
        #[test]
        fn manual_sleep_holds_pose_until_explicit_wake() {
            for reposition in [false, true] {
                let mut physics = Physics::default();
                let body = physics.add_body(RigidBodyBuilder::dynamic().build());
                physics.add_collider(ColliderBuilder::ball(0.5).build(), body);
                physics.step();
                assert_ne!(physics.bodies[body].linvel(), Vector::ZERO);
                if reposition {
                    let mut position = physics.bodies[body].translation();
                    position.x += 2.0;
                    physics.bodies[body].set_translation(position, true);
                }
                physics.body_motion(body).unwrap().sleep();
                let pose = *physics.bodies[body].position();
                for _ in 0..5 {
                    physics.step();
                    assert!(physics.bodies[body].is_sleeping());
                    assert_eq!(*physics.bodies[body].position(), pose);
                    assert_eq!(physics.bodies[body].linvel(), Vector::ZERO);
                }
                physics.body_motion(body).unwrap().wake_up(true);
                physics.step();
                assert!(!physics.bodies[body].is_sleeping());
                assert!(physics.bodies[body].translation().y < pose.translation.y);
            }
        }

        fn ray_fixture() -> (
            Physics,
            crate::Entity,
            ColliderHandle,
            RigidBodyHandle,
            ColliderHandle,
        ) {
            let mut physics = Physics::default();
            let entity = crate::Entity::new(42, 0);
            let mut far = Vector::ZERO;
            far.x = 5.0;
            let body = physics.add_body(RigidBodyBuilder::fixed().translation(far).build());
            let collider = physics.add_collider(ColliderBuilder::ball(0.5).build(), body);
            physics.entity_to_body.insert(entity, body);
            physics.body_to_entity.insert(body, entity);
            let mut near = Vector::ZERO;
            near.x = 2.0;
            let free =
                physics.add_free_collider(ColliderBuilder::ball(0.5).translation(near).build());
            physics.step();
            (physics, entity, free, body, collider)
        }

        #[test]
        fn collision_groups_filter_ray_hits_and_free_colliders() {
            use crate::physics::CollisionGroups;
            let (mut physics, entity, free, _, collider) = ray_fixture();
            physics.colliders[free].set_collision_groups(CollisionGroups::new(1, 4).into());
            physics.colliders[collider].set_collision_groups(CollisionGroups::new(2, 4).into());
            physics.step();
            let origin = GameVector::zeros();
            let mut dir = GameVector::zeros();
            dir.x = 1.0;
            let hit = physics
                .cast_ray_filtered(
                    origin,
                    dir,
                    10.0,
                    QueryFilter::default().groups(CollisionGroups::new(4, 2).into()),
                )
                .unwrap();
            assert_eq!(hit.collider, collider);
            assert_eq!(hit.entity, Some(entity));
            // Query membership must also be accepted by the collider.
            assert!(
                physics
                    .cast_ray_filtered(
                        origin,
                        dir,
                        10.0,
                        QueryFilter::default().groups(CollisionGroups::new(8, 2).into()),
                    )
                    .is_none()
            );
            // Queries without a group filter still see every collider.
            assert_eq!(physics.cast_ray(origin, dir, 10.0).unwrap().collider, free);
        }

        #[test]
        fn ray_hit_keeps_unowned_colliders_and_filters_before_selecting_nearest() {
            let (physics, entity, free, body, collider) = ray_fixture();
            let origin = GameVector::zeros();
            let mut dir = GameVector::zeros();
            dir.x = 1.0;
            let hit = physics.cast_ray(origin, dir, 10.0).unwrap();
            assert_eq!(hit.collider, free);
            assert_eq!(hit.body, None);
            assert_eq!(hit.entity, None);
            assert!((hit.toi - 1.5).abs() < 1e-5);
            let only_ecs = |_: ColliderHandle, c: &Collider| {
                c.parent()
                    .and_then(|h| physics.entity_for_body(h))
                    .is_some()
            };
            let hit = physics
                .cast_ray_filtered(
                    origin,
                    dir,
                    10.0,
                    QueryFilter::default().predicate(&only_ecs),
                )
                .unwrap();
            assert_eq!(hit.collider, collider);
            assert_eq!(hit.body, Some(body));
            assert_eq!(hit.entity, Some(entity));
            assert!((hit.toi - 4.5).abs() < 1e-5);
            // TOI parameterizes the ray; a non-unit direction is not normalized silently.
            let hit = physics.cast_ray(origin, dir * 2.0, 10.0).unwrap();
            assert!((hit.toi - 0.75).abs() < 1e-5);
            assert!(physics.cast_ray(origin, dir, 1.0).is_none());
            assert!(physics.cast_ray(origin, -dir, 10.0).is_none());
        }

        #[test]
        fn ray_hit_keeps_body_handle_without_entity_mapping() {
            let (mut physics, entity, free, body, collider) = ray_fixture();
            assert!(physics.remove_free_collider(free));
            physics.body_to_entity.remove(&body);
            physics.entity_to_body.remove(&entity);
            physics.step();
            let mut dir = GameVector::zeros();
            dir.x = 1.0;
            let hit = physics.cast_ray(GameVector::zeros(), dir, 10.0).unwrap();
            assert_eq!(hit.collider, collider);
            assert_eq!(hit.body, Some(body));
            assert_eq!(hit.entity, None);
        }

        #[test]
        fn free_collider_removal_cannot_remove_a_managed_collider() {
            let (mut physics, entity, free, body, collider) = ray_fixture();
            assert!(!physics.remove_free_collider(collider));
            assert_eq!(physics.body_for_entity(entity), Some(body));
            assert!(physics.colliders().contains(collider));
            assert_eq!(physics.bodies()[body].colliders(), &[collider]);
            assert!(physics.remove_free_collider(free));
            assert!(!physics.remove_free_collider(free));
            let replacement = physics.add_free_collider(ColliderBuilder::ball(0.7).build());
            assert!(!physics.remove_free_collider(free));
            assert!(physics.colliders().contains(replacement));
        }

        #[test]
        fn motion_access_preserves_structure_and_rejects_removed_handles() {
            let mut physics = Physics::default();
            let body = physics.add_body(RigidBodyBuilder::dynamic().build());
            let collider = physics.add_collider(ColliderBuilder::ball(0.5).build(), body);
            {
                let mut motion = physics.body_motion(body).unwrap();
                motion.sleep();
                assert!(motion.is_sleeping());
                motion.apply_impulse(Vector::X, true);
                assert!(!motion.is_sleeping());
                assert!(motion.linvel().x > 0.0);
                motion.set_linvel(Vector::X * 2.0, true);
                assert_eq!(motion.linvel().x, 2.0);
                motion.add_force(Vector::X, true);
                motion.reset_forces(true);
                assert_eq!(motion.user_force(), Vector::ZERO);
                assert_eq!(motion.colliders(), &[collider]);
            }
            physics.remove_body(body);
            assert!(physics.body_motion(body).is_none());
            let replacement = physics.add_body(RigidBodyBuilder::dynamic().build());
            assert!(physics.body_motion(body).is_none());
            assert!(physics.body_motion(replacement).is_some());
        }
    };
}
pub(super) use world_boundary_tests;
