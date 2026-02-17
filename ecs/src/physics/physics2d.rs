//! 2D physics resources, systems, and handle components.
//!
//! Uses rapier2d (f64 by default, f32 with `physics-2d-f32` feature).

use std::collections::HashMap;

use super::rapier2d::prelude::*;

// ---- Handle components ----

/// ECS component holding a rapier 2D rigid body handle.
#[derive(Debug, Clone, Copy)]
pub struct RigidBody2DHandle(pub RigidBodyHandle);

/// ECS component holding a rapier 2D collider handle.
#[derive(Debug, Clone, Copy)]
pub struct Collider2DHandle(pub ColliderHandle);

/// ECS component holding a rapier 2D impulse joint handle.
#[derive(Debug, Clone, Copy)]
pub struct ImpulseJoint2DHandle(pub ImpulseJointHandle);

// ---- PhysicsWorld2D resource ----

/// Single ECS resource holding all rapier 2D physics state plus entity mapping.
///
/// The entity mapping (`entity_to_body`, `body_to_entity`, `entity_to_joint`)
/// allows any system to look up the correspondence between ECS entities and
/// rapier handles — useful for raycasting, contact queries, etc.
pub struct PhysicsWorld2D {
    pub gravity: Vector,
    pub integration_parameters: IntegrationParameters,
    pub pipeline: PhysicsPipeline,
    pub island_manager: IslandManager,
    pub broad_phase: DefaultBroadPhase,
    pub narrow_phase: NarrowPhase,
    pub bodies: RigidBodySet,
    pub colliders: ColliderSet,
    pub impulse_joints: ImpulseJointSet,
    pub multibody_joints: MultibodyJointSet,
    pub ccd_solver: CCDSolver,

    /// Maps ECS entity → rapier body handle.
    pub entity_to_body: HashMap<crate::Entity, RigidBodyHandle>,
    /// Maps rapier body handle → ECS entity (reverse lookup for raycasts).
    pub body_to_entity: HashMap<RigidBodyHandle, crate::Entity>,
    /// Maps ECS entity → rapier impulse joint handle.
    pub entity_to_joint: HashMap<crate::Entity, ImpulseJointHandle>,
}

impl Default for PhysicsWorld2D {
    fn default() -> Self {
        Self {
            gravity: Vector::new(0.0, -9.81),
            integration_parameters: IntegrationParameters::default(),
            pipeline: PhysicsPipeline::new(),
            island_manager: IslandManager::new(),
            broad_phase: DefaultBroadPhase::new(),
            narrow_phase: NarrowPhase::new(),
            bodies: RigidBodySet::new(),
            colliders: ColliderSet::new(),
            impulse_joints: ImpulseJointSet::new(),
            multibody_joints: MultibodyJointSet::new(),
            ccd_solver: CCDSolver::new(),
            entity_to_body: HashMap::new(),
            body_to_entity: HashMap::new(),
            entity_to_joint: HashMap::new(),
        }
    }
}

impl PhysicsWorld2D {
    /// Creates a new physics world with the given gravity.
    pub fn with_gravity(gravity: Vector) -> Self {
        Self {
            gravity,
            ..Default::default()
        }
    }

    /// Steps the physics simulation by one timestep.
    pub fn step(&mut self) {
        redlilium_core::profile_scope!("rapier2d: step");
        self.pipeline.step(
            self.gravity,
            &self.integration_parameters,
            &mut self.island_manager,
            &mut self.broad_phase,
            &mut self.narrow_phase,
            &mut self.bodies,
            &mut self.colliders,
            &mut self.impulse_joints,
            &mut self.multibody_joints,
            &mut self.ccd_solver,
            &(),
            &(),
        );
    }

    /// Adds a rigid body and returns its handle.
    pub fn add_body(&mut self, body: RigidBody) -> RigidBodyHandle {
        self.bodies.insert(body)
    }

    /// Adds a collider attached to a rigid body and returns its handle.
    pub fn add_collider(&mut self, collider: Collider, parent: RigidBodyHandle) -> ColliderHandle {
        self.colliders
            .insert_with_parent(collider, parent, &mut self.bodies)
    }

    /// Adds a free collider (not attached to any body) and returns its handle.
    pub fn add_free_collider(&mut self, collider: Collider) -> ColliderHandle {
        self.colliders.insert(collider)
    }

    /// Adds an impulse joint between two bodies and returns its handle.
    pub fn add_impulse_joint(
        &mut self,
        body1: RigidBodyHandle,
        body2: RigidBodyHandle,
        joint: impl Into<GenericJoint>,
    ) -> ImpulseJointHandle {
        self.impulse_joints.insert(body1, body2, joint, true)
    }

    /// Removes a rigid body and all its attached colliders and joints.
    pub fn remove_body(&mut self, handle: RigidBodyHandle) {
        self.bodies.remove(
            handle,
            &mut self.island_manager,
            &mut self.colliders,
            &mut self.impulse_joints,
            &mut self.multibody_joints,
            true,
        );
    }

    /// Removes an impulse joint.
    pub fn remove_impulse_joint(&mut self, handle: ImpulseJointHandle, wake_up: bool) {
        self.impulse_joints.remove(handle, wake_up);
    }

    /// Returns the ECS entity that owns the given body handle.
    pub fn entity_for_body(&self, handle: RigidBodyHandle) -> Option<crate::Entity> {
        self.body_to_entity.get(&handle).copied()
    }

    /// Returns the rapier body handle for the given ECS entity.
    pub fn body_for_entity(&self, entity: crate::Entity) -> Option<RigidBodyHandle> {
        self.entity_to_body.get(&entity).copied()
    }

    /// Casts a ray and returns the first hit as `(entity, toi)`.
    ///
    /// Returns `None` if no collider was hit within `max_toi`.
    pub fn cast_ray(
        &self,
        origin: redlilium_core::math::Vec2,
        dir: redlilium_core::math::Vec2,
        max_toi: f32,
    ) -> Option<(crate::Entity, f32)> {
        use redlilium_core::math::Real;

        let ray = Ray::new(
            Vector::new(origin.x as Real, origin.y as Real),
            Vector::new(dir.x as Real, dir.y as Real),
        );

        let query_pipeline = self.broad_phase.as_query_pipeline(
            self.narrow_phase.query_dispatcher(),
            &self.bodies,
            &self.colliders,
            QueryFilter::default(),
        );

        let (collider_handle, toi) = query_pipeline.cast_ray(&ray, max_toi as Real, true)?;

        let collider = self.colliders.get(collider_handle)?;
        let body_handle = collider.parent()?;
        let entity = self.body_to_entity.get(&body_handle)?;
        Some((*entity, toi as f32))
    }
}

// ---- StepPhysics2D system ----

/// ECS system that steps the 2D physics simulation and syncs body positions
/// back to ECS [`Transform`](crate::Transform) components.
///
/// For 2D, the X/Y rapier position maps to the Transform's X/Y translation,
/// and the rapier rotation angle maps to a Z-axis rotation quaternion.
pub struct StepPhysics2D;

impl crate::System for StepPhysics2D {
    type Result = ();
    fn run<'a>(
        &'a self,
        ctx: &'a crate::SystemContext<'a>,
    ) -> Result<(), crate::system::SystemError> {
        ctx.lock::<(
            crate::ResMut<PhysicsWorld2D>,
            crate::Read<RigidBody2DHandle>,
            crate::Write<crate::Transform>,
        )>()
        .execute(|(mut physics, handles, mut transforms)| {
            redlilium_core::profile_scope!("ecs: step_physics_2d");

            // Step simulation
            physics.step();

            // Sync positions back to transforms
            for (idx, handle) in handles.iter() {
                if let Some(body) = physics.bodies.get(handle.0)
                    && (body.is_dynamic() || body.is_kinematic())
                    && let Some(transform) = transforms.get_mut(idx)
                {
                    let pos = body.position();
                    let t = pos.translation;
                    transform.translation =
                        redlilium_core::math::Vec3::new(t.x as f32, t.y as f32, 0.0);
                    let angle = pos.rotation.angle() as f32;
                    transform.rotation = redlilium_core::math::quat_from_rotation_z(angle);
                }
            }
        });
        Ok(())
    }
}

// ---- SyncPhysicsBodies2D exclusive system ----

/// Exclusive system that creates/removes rapier bodies from ECS descriptor components.
///
/// Detects entities with [`RigidBody2D`](super::components2d::RigidBody2D) +
/// [`Collider2D`](super::components2d::Collider2D) +
/// [`Transform`](crate::Transform) and creates corresponding rapier objects.
/// Also detects removed/despawned entities and cleans up.
pub struct SyncPhysicsBodies2D;

impl crate::ExclusiveSystem for SyncPhysicsBodies2D {
    type Result = ();

    fn run(&mut self, world: &mut crate::World) -> Result<(), crate::system::SystemError> {
        redlilium_core::profile_scope!("ecs: sync_physics_bodies_2d");

        // Ensure resource exists
        if !world.has_resource::<PhysicsWorld2D>() {
            world.insert_resource(PhysicsWorld2D::default());
        }

        // Phase 1: Find stale bodies (entity dead, disabled, or lost RigidBody2D component)
        let stale: Vec<crate::Entity> = {
            let physics = world.resource::<PhysicsWorld2D>();
            physics
                .entity_to_body
                .keys()
                .filter(|e| {
                    !world.is_alive(**e)
                        || world.is_disabled(**e)
                        || world.get::<super::components2d::RigidBody2D>(**e).is_none()
                })
                .copied()
                .collect()
        };

        // Remove stale bodies from rapier and clean mappings
        if !stale.is_empty() {
            // Also find joints that reference stale bodies
            let stale_joints: Vec<crate::Entity> = {
                let physics = world.resource::<PhysicsWorld2D>();
                physics
                    .entity_to_joint
                    .keys()
                    .filter(|je| {
                        if let Some(joint_desc) =
                            world.get::<super::components2d::ImpulseJoint2D>(**je)
                        {
                            stale.contains(&joint_desc.body1) || stale.contains(&joint_desc.body2)
                        } else {
                            false
                        }
                    })
                    .copied()
                    .collect()
            };

            {
                let mut physics = world.resource_mut::<PhysicsWorld2D>();
                for entity in &stale_joints {
                    if let Some(jh) = physics.entity_to_joint.remove(entity) {
                        physics.remove_impulse_joint(jh, true);
                    }
                }
                for entity in &stale {
                    if let Some(bh) = physics.entity_to_body.remove(entity) {
                        physics.body_to_entity.remove(&bh);
                        physics.remove_body(bh);
                    }
                }
            }

            for entity in &stale_joints {
                if world.is_alive(*entity) {
                    world.remove::<ImpulseJoint2DHandle>(*entity);
                }
            }
            for entity in &stale {
                if world.is_alive(*entity) {
                    world.remove::<RigidBody2DHandle>(*entity);
                }
            }
        }

        // Phase 2: Find new bodies (have descriptors, not in mapping, not disabled)
        let new_entities: Vec<(
            crate::Entity,
            super::components2d::RigidBody2D,
            super::components2d::Collider2D,
            crate::Transform,
        )> = {
            let physics = world.resource::<PhysicsWorld2D>();
            world
                .iter_entities()
                .filter(|e| !physics.entity_to_body.contains_key(e) && !world.is_disabled(*e))
                .filter_map(|entity| {
                    let body = world
                        .get::<super::components2d::RigidBody2D>(entity)?
                        .clone();
                    let collider = world
                        .get::<super::components2d::Collider2D>(entity)?
                        .clone();
                    let transform = *world.get::<crate::Transform>(entity)?;
                    Some((entity, body, collider, transform))
                })
                .collect()
        };

        if !new_entities.is_empty() {
            let mut handles = Vec::with_capacity(new_entities.len());
            {
                let mut physics = world.resource_mut::<PhysicsWorld2D>();
                for (entity, body_desc, collider_desc, transform) in &new_entities {
                    let rapier_body = body_desc.to_rigid_body(transform);
                    let body_handle = physics.add_body(rapier_body);
                    let rapier_collider = collider_desc.to_collider();
                    physics.add_collider(rapier_collider, body_handle);
                    physics.entity_to_body.insert(*entity, body_handle);
                    physics.body_to_entity.insert(body_handle, *entity);
                    handles.push((*entity, body_handle));
                }
            }
            for (entity, handle) in handles {
                let _ = world.insert(entity, RigidBody2DHandle(handle));
            }
        }

        Ok(())
    }
}

// ---- SyncPhysicsJoints2D exclusive system ----

/// Exclusive system that creates/removes rapier joints from ECS descriptor components.
///
/// Detects entities with [`ImpulseJoint2D`](super::components2d::ImpulseJoint2D)
/// and creates corresponding rapier joints. Also detects removed/despawned joints.
///
/// Must run after [`SyncPhysicsBodies2D`] so that body handles are available.
pub struct SyncPhysicsJoints2D;

impl crate::ExclusiveSystem for SyncPhysicsJoints2D {
    type Result = ();

    fn run(&mut self, world: &mut crate::World) -> Result<(), crate::system::SystemError> {
        redlilium_core::profile_scope!("ecs: sync_physics_joints_2d");

        if !world.has_resource::<PhysicsWorld2D>() {
            return Ok(());
        }

        // Phase 1: Find stale joints (entity dead, disabled, or lost ImpulseJoint2D component)
        let stale: Vec<crate::Entity> = {
            let physics = world.resource::<PhysicsWorld2D>();
            physics
                .entity_to_joint
                .keys()
                .filter(|e| {
                    !world.is_alive(**e)
                        || world.is_disabled(**e)
                        || world
                            .get::<super::components2d::ImpulseJoint2D>(**e)
                            .is_none()
                })
                .copied()
                .collect()
        };

        if !stale.is_empty() {
            {
                let mut physics = world.resource_mut::<PhysicsWorld2D>();
                for entity in &stale {
                    if let Some(jh) = physics.entity_to_joint.remove(entity) {
                        physics.remove_impulse_joint(jh, true);
                    }
                }
            }
            for entity in &stale {
                if world.is_alive(*entity) {
                    world.remove::<ImpulseJoint2DHandle>(*entity);
                }
            }
        }

        // Phase 2: Find new joints (not in mapping, not disabled)
        let new_joints: Vec<(crate::Entity, super::components2d::ImpulseJoint2D)> = {
            let physics = world.resource::<PhysicsWorld2D>();
            world
                .iter_entities()
                .filter(|e| !physics.entity_to_joint.contains_key(e) && !world.is_disabled(*e))
                .filter_map(|entity| {
                    let joint = world
                        .get::<super::components2d::ImpulseJoint2D>(entity)?
                        .clone();
                    Some((entity, joint))
                })
                .collect()
        };

        if !new_joints.is_empty() {
            let mut handles = Vec::new();
            {
                let mut physics = world.resource_mut::<PhysicsWorld2D>();
                for (entity, joint_desc) in &new_joints {
                    let body1_handle = match physics.entity_to_body.get(&joint_desc.body1) {
                        Some(h) => *h,
                        None => continue,
                    };
                    let body2_handle = match physics.entity_to_body.get(&joint_desc.body2) {
                        Some(h) => *h,
                        None => continue,
                    };
                    let rapier_joint = joint_desc.to_rapier_joint();
                    let jh = physics.add_impulse_joint(body1_handle, body2_handle, rapier_joint);
                    physics.entity_to_joint.insert(*entity, jh);
                    handles.push((*entity, jh));
                }
            }
            for (entity, handle) in handles {
                let _ = world.insert(entity, ImpulseJoint2DHandle(handle));
            }
        }

        Ok(())
    }
}

// ---- Regular system variants ----

/// Regular system variant of [`SyncPhysicsBodies2D`].
///
/// Uses lock-execute + deferred commands instead of exclusive world access.
/// Allows parallel scheduling but joints may lag 1 frame behind body creation.
pub struct SyncPhysicsBodiesSystem2D;

impl crate::System for SyncPhysicsBodiesSystem2D {
    type Result = ();

    fn run<'a>(
        &'a self,
        ctx: &'a crate::SystemContext<'a>,
    ) -> Result<(), crate::system::SystemError> {
        redlilium_core::profile_scope!("ecs: sync_physics_bodies_system_2d");

        let (new_indices, stale_entities) = ctx
            .lock::<(
                crate::ResMut<PhysicsWorld2D>,
                crate::Read<super::components2d::RigidBody2D>,
                crate::Read<super::components2d::Collider2D>,
                crate::Read<crate::Transform>,
            )>()
            .execute(|(mut physics, bodies, colliders, transforms)| {
                // Remove stale
                let stale: Vec<crate::Entity> = physics
                    .entity_to_body
                    .keys()
                    .filter(|e| bodies.get(e.index()).is_none())
                    .copied()
                    .collect();
                for entity in &stale {
                    if let Some(bh) = physics.entity_to_body.remove(entity) {
                        physics.body_to_entity.remove(&bh);
                        physics.remove_body(bh);
                    }
                }

                // Find tracked indices
                let tracked: std::collections::HashSet<u32> =
                    physics.entity_to_body.keys().map(|e| e.index()).collect();

                // Create new
                let mut new_pairs: Vec<(u32, RigidBodyHandle)> = Vec::new();
                for (idx, body_desc) in bodies.iter() {
                    if !tracked.contains(&idx)
                        && let (Some(collider_desc), Some(transform)) =
                            (colliders.get(idx), transforms.get(idx))
                    {
                        let rapier_body = body_desc.to_rigid_body(transform);
                        let body_handle = physics.add_body(rapier_body);
                        let rapier_collider = collider_desc.to_collider();
                        physics.add_collider(rapier_collider, body_handle);
                        new_pairs.push((idx, body_handle));
                    }
                }

                (new_pairs, stale)
            });

        if !new_indices.is_empty() || !stale_entities.is_empty() {
            ctx.commands(move |world| {
                for entity in stale_entities {
                    if world.is_alive(entity) {
                        world.remove::<RigidBody2DHandle>(entity);
                    }
                }
                for (idx, handle) in new_indices {
                    if let Some(entity) = world.entity_at_index(idx) {
                        let _ = world.insert(entity, RigidBody2DHandle(handle));
                        let mut physics = world.resource_mut::<PhysicsWorld2D>();
                        physics.entity_to_body.insert(entity, handle);
                        physics.body_to_entity.insert(handle, entity);
                    }
                }
            });
        }

        Ok(())
    }
}

/// Regular system variant of [`SyncPhysicsJoints2D`].
///
/// Uses lock-execute + deferred commands. Joint creation may lag 1 frame behind
/// body creation when both are spawned in the same frame.
pub struct SyncPhysicsJointsSystem2D;

impl crate::System for SyncPhysicsJointsSystem2D {
    type Result = ();

    fn run<'a>(
        &'a self,
        ctx: &'a crate::SystemContext<'a>,
    ) -> Result<(), crate::system::SystemError> {
        redlilium_core::profile_scope!("ecs: sync_physics_joints_system_2d");

        let (new_indices, stale_entities) = ctx
            .lock::<(
                crate::ResMut<PhysicsWorld2D>,
                crate::Read<super::components2d::ImpulseJoint2D>,
            )>()
            .execute(|(mut physics, joints)| {
                // Remove stale
                let stale: Vec<crate::Entity> = physics
                    .entity_to_joint
                    .keys()
                    .filter(|e| joints.get(e.index()).is_none())
                    .copied()
                    .collect();
                for entity in &stale {
                    if let Some(jh) = physics.entity_to_joint.remove(entity) {
                        physics.remove_impulse_joint(jh, true);
                    }
                }

                // Find tracked
                let tracked: std::collections::HashSet<u32> =
                    physics.entity_to_joint.keys().map(|e| e.index()).collect();

                // Create new
                let mut new_pairs: Vec<(u32, ImpulseJointHandle)> = Vec::new();
                for (idx, joint_desc) in joints.iter() {
                    if !tracked.contains(&idx) {
                        let body1_handle = match physics.entity_to_body.get(&joint_desc.body1) {
                            Some(h) => *h,
                            None => continue,
                        };
                        let body2_handle = match physics.entity_to_body.get(&joint_desc.body2) {
                            Some(h) => *h,
                            None => continue,
                        };
                        let rapier_joint = joint_desc.to_rapier_joint();
                        let jh =
                            physics.add_impulse_joint(body1_handle, body2_handle, rapier_joint);
                        new_pairs.push((idx, jh));
                    }
                }

                (new_pairs, stale)
            });

        if !new_indices.is_empty() || !stale_entities.is_empty() {
            ctx.commands(move |world| {
                for entity in stale_entities {
                    if world.is_alive(entity) {
                        world.remove::<ImpulseJoint2DHandle>(entity);
                    }
                }
                for (idx, handle) in new_indices {
                    if let Some(entity) = world.entity_at_index(idx) {
                        let _ = world.insert(entity, ImpulseJoint2DHandle(handle));
                        let mut physics = world.resource_mut::<PhysicsWorld2D>();
                        physics.entity_to_joint.insert(entity, handle);
                    }
                }
            });
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn physics_world_2d_default() {
        let world = PhysicsWorld2D::default();
        assert!((world.gravity.y - (-9.81)).abs() < 1e-10);
        assert_eq!(world.bodies.len(), 0);
        assert_eq!(world.colliders.len(), 0);
    }

    #[test]
    fn add_body_and_collider_2d() {
        let mut physics = PhysicsWorld2D::default();

        let body_handle = physics.add_body(
            RigidBodyBuilder::dynamic()
                .translation(Vector::new(0.0, 10.0))
                .build(),
        );
        let _collider_handle =
            physics.add_collider(ColliderBuilder::ball(0.5).build(), body_handle);

        assert_eq!(physics.bodies.len(), 1);
        assert_eq!(physics.colliders.len(), 1);
    }

    #[test]
    fn step_moves_dynamic_body_2d() {
        let mut physics = PhysicsWorld2D::default();

        let body_handle = physics.add_body(
            RigidBodyBuilder::dynamic()
                .translation(Vector::new(0.0, 10.0))
                .build(),
        );
        physics.add_collider(ColliderBuilder::ball(0.5).build(), body_handle);

        let initial_y = physics.bodies[body_handle].position().translation.y;

        for _ in 0..10 {
            physics.step();
        }

        let final_y = physics.bodies[body_handle].position().translation.y;
        assert!(final_y < initial_y);
    }

    #[test]
    fn entity_mapping_roundtrip_2d() {
        let mut physics = PhysicsWorld2D::default();
        let entity = crate::Entity::new(42, 0);
        let bh = physics.add_body(RigidBodyBuilder::dynamic().build());

        physics.entity_to_body.insert(entity, bh);
        physics.body_to_entity.insert(bh, entity);

        assert_eq!(physics.entity_for_body(bh), Some(entity));
        assert_eq!(physics.body_for_entity(entity), Some(bh));
    }

    #[test]
    fn sync_bodies_creates_and_removes_2d() {
        use crate::system::run_exclusive_system_once;
        use redlilium_core::math::Vec3;

        let mut world = crate::World::new();
        crate::register_std_components(&mut world);

        // Spawn a dynamic ball
        let e = world.spawn();
        let _ = world.insert(e, super::super::components2d::RigidBody2D::dynamic());
        let _ = world.insert(e, super::super::components2d::Collider2D::ball(0.5));
        let _ = world.insert(
            e,
            crate::Transform::from_translation(Vec3::new(0.0, 10.0, 0.0)),
        );

        // Run sync
        run_exclusive_system_once(&mut SyncPhysicsBodies2D, &mut world).unwrap();

        // Should have handle
        assert!(world.get::<RigidBody2DHandle>(e).is_some());
        {
            let physics = world.resource::<PhysicsWorld2D>();
            assert_eq!(physics.bodies.len(), 1);
            assert!(physics.entity_to_body.contains_key(&e));
        }

        // Now remove the descriptor
        world.remove::<super::super::components2d::RigidBody2D>(e);

        // Run sync again
        run_exclusive_system_once(&mut SyncPhysicsBodies2D, &mut world).unwrap();

        // Should be cleaned up
        assert!(world.get::<RigidBody2DHandle>(e).is_none());
        {
            let physics = world.resource::<PhysicsWorld2D>();
            assert_eq!(physics.bodies.len(), 0);
            assert!(!physics.entity_to_body.contains_key(&e));
        }
    }
}
