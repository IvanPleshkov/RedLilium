//! 3D physics resources, systems, and handle components.
//!
//! Uses rapier3d (f64 by default, f32 with `physics-3d-f32` feature).

use std::collections::HashMap;

use super::rapier3d::prelude::*;

// ---- Handle components ----

/// ECS component holding a rapier rigid body handle.
#[derive(Debug, Clone, Copy)]
pub struct RigidBody3DHandle(pub RigidBodyHandle);

/// ECS component holding a rapier collider handle.
#[derive(Debug, Clone, Copy)]
pub struct Collider3DHandle(pub ColliderHandle);

/// ECS component holding a rapier impulse joint handle.
#[derive(Debug, Clone, Copy)]
pub struct ImpulseJoint3DHandle(pub ImpulseJointHandle);

// ---- PhysicsWorld3D resource ----

/// Single ECS resource holding all rapier 3D physics state plus entity mapping.
///
/// The entity mapping (`entity_to_body`, `body_to_entity`, `entity_to_joint`)
/// allows any system to look up the correspondence between ECS entities and
/// rapier handles — useful for raycasting, contact queries, etc.
///
/// # Example
///
/// ```ignore
/// // In a system, cast a ray and get the hit entity:
/// ctx.lock::<(Res<PhysicsWorld3D>,)>().execute(|(physics,)| {
///     if let Some((entity, toi)) = physics.cast_ray(origin, dir, 100.0) {
///         // `entity` is the ECS entity that was hit
///     }
/// });
/// ```
pub struct PhysicsWorld3D {
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

impl Default for PhysicsWorld3D {
    fn default() -> Self {
        Self {
            gravity: Vector::new(0.0, -9.81, 0.0),
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

impl PhysicsWorld3D {
    /// Creates a new physics world with the given gravity.
    pub fn with_gravity(gravity: Vector) -> Self {
        Self {
            gravity,
            ..Default::default()
        }
    }

    /// Steps the physics simulation by one timestep.
    pub fn step(&mut self) {
        redlilium_core::profile_scope!("rapier3d: step");
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
        origin: redlilium_core::math::Vec3,
        dir: redlilium_core::math::Vec3,
        max_toi: f32,
    ) -> Option<(crate::Entity, f32)> {
        use redlilium_core::math::Real;

        let ray = Ray::new(
            Vector::new(origin.x as Real, origin.y as Real, origin.z as Real),
            Vector::new(dir.x as Real, dir.y as Real, dir.z as Real),
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

// ---- StepPhysics3D system ----

/// ECS system that steps the 3D physics simulation and syncs body positions
/// back to ECS [`Transform`](crate::Transform) components.
///
/// Requires a [`PhysicsWorld3D`] resource and entities with
/// [`RigidBody3DHandle`] + [`Transform`](crate::Transform) components.
pub struct StepPhysics3D;

impl crate::System for StepPhysics3D {
    type Result = ();
    fn run<'a>(
        &'a self,
        ctx: &'a crate::SystemContext<'a>,
    ) -> Result<(), crate::system::SystemError> {
        ctx.lock::<(
            crate::ResMut<PhysicsWorld3D>,
            crate::Read<RigidBody3DHandle>,
            crate::Write<crate::Transform>,
        )>()
        .execute(|(mut physics, handles, mut transforms)| {
            redlilium_core::profile_scope!("ecs: step_physics_3d");

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
                        redlilium_core::math::Vec3::new(t.x as f32, t.y as f32, t.z as f32);
                    let r = pos.rotation;
                    transform.rotation = redlilium_core::math::quat_from_xyzw(
                        r.x as f32, r.y as f32, r.z as f32, r.w as f32,
                    );
                }
            }
        });
        Ok(())
    }
}

// ---- SyncPhysicsBodies3D exclusive system ----

/// Exclusive system that creates/removes rapier bodies from ECS descriptor components.
///
/// Detects entities with [`RigidBody3D`](super::components3d::RigidBody3D) +
/// [`Collider3D`](super::components3d::Collider3D) +
/// [`Transform`](crate::Transform) and creates corresponding rapier objects.
/// Also detects removed/despawned entities and cleans up.
///
/// # Example
///
/// ```ignore
/// let mut systems = SystemsContainer::new();
/// systems.add_exclusive(SyncPhysicsBodies3D);
/// systems.add(StepPhysics3D);
/// systems.add_edge::<SyncPhysicsBodies3D, StepPhysics3D>().unwrap();
/// ```
pub struct SyncPhysicsBodies3D;

impl crate::ExclusiveSystem for SyncPhysicsBodies3D {
    type Result = ();

    fn run(&mut self, world: &mut crate::World) -> Result<(), crate::system::SystemError> {
        redlilium_core::profile_scope!("ecs: sync_physics_bodies_3d");

        // Ensure resource exists
        if !world.has_resource::<PhysicsWorld3D>() {
            world.insert_resource(PhysicsWorld3D::default());
        }

        // Phase 1: Find stale bodies (entity dead, disabled, or lost RigidBody3D component)
        let stale: Vec<crate::Entity> = {
            let physics = world.resource::<PhysicsWorld3D>();
            physics
                .entity_to_body
                .keys()
                .filter(|e| {
                    !world.is_alive(**e)
                        || world.is_disabled(**e)
                        || world.get::<super::components3d::RigidBody3D>(**e).is_none()
                })
                .copied()
                .collect()
        };

        // Remove stale bodies from rapier and clean mappings
        if !stale.is_empty() {
            // Also find joints that reference stale bodies
            let stale_joints: Vec<crate::Entity> = {
                let physics = world.resource::<PhysicsWorld3D>();
                physics
                    .entity_to_joint
                    .keys()
                    .filter(|je| {
                        if let Some(joint_desc) =
                            world.get::<super::components3d::ImpulseJoint3D>(**je)
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
                let mut physics = world.resource_mut::<PhysicsWorld3D>();
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
                    world.remove::<ImpulseJoint3DHandle>(*entity);
                }
            }
            for entity in &stale {
                if world.is_alive(*entity) {
                    world.remove::<RigidBody3DHandle>(*entity);
                }
            }
        }

        // Phase 2: Find new bodies (have descriptors, not in mapping, not disabled)
        let new_entities: Vec<(
            crate::Entity,
            super::components3d::RigidBody3D,
            super::components3d::Collider3D,
            crate::Transform,
        )> = {
            let physics = world.resource::<PhysicsWorld3D>();
            world
                .iter_entities()
                .filter(|e| !physics.entity_to_body.contains_key(e) && !world.is_disabled(*e))
                .filter_map(|entity| {
                    let body = world
                        .get::<super::components3d::RigidBody3D>(entity)?
                        .clone();
                    let collider = world
                        .get::<super::components3d::Collider3D>(entity)?
                        .clone();
                    let transform = *world.get::<crate::Transform>(entity)?;
                    Some((entity, body, collider, transform))
                })
                .collect()
        };

        if !new_entities.is_empty() {
            let mut handles = Vec::with_capacity(new_entities.len());
            {
                let mut physics = world.resource_mut::<PhysicsWorld3D>();
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
                let _ = world.insert(entity, RigidBody3DHandle(handle));
            }
        }

        Ok(())
    }
}

// ---- SyncPhysicsJoints3D exclusive system ----

/// Exclusive system that creates/removes rapier joints from ECS descriptor components.
///
/// Detects entities with [`ImpulseJoint3D`](super::components3d::ImpulseJoint3D)
/// and creates corresponding rapier joints. Also detects removed/despawned joints.
///
/// Must run after [`SyncPhysicsBodies3D`] so that body handles are available.
pub struct SyncPhysicsJoints3D;

impl crate::ExclusiveSystem for SyncPhysicsJoints3D {
    type Result = ();

    fn run(&mut self, world: &mut crate::World) -> Result<(), crate::system::SystemError> {
        redlilium_core::profile_scope!("ecs: sync_physics_joints_3d");

        if !world.has_resource::<PhysicsWorld3D>() {
            return Ok(());
        }

        // Phase 1: Find stale joints (entity dead, disabled, or lost ImpulseJoint3D component)
        let stale: Vec<crate::Entity> = {
            let physics = world.resource::<PhysicsWorld3D>();
            physics
                .entity_to_joint
                .keys()
                .filter(|e| {
                    !world.is_alive(**e)
                        || world.is_disabled(**e)
                        || world
                            .get::<super::components3d::ImpulseJoint3D>(**e)
                            .is_none()
                })
                .copied()
                .collect()
        };

        if !stale.is_empty() {
            {
                let mut physics = world.resource_mut::<PhysicsWorld3D>();
                for entity in &stale {
                    if let Some(jh) = physics.entity_to_joint.remove(entity) {
                        physics.remove_impulse_joint(jh, true);
                    }
                }
            }
            for entity in &stale {
                if world.is_alive(*entity) {
                    world.remove::<ImpulseJoint3DHandle>(*entity);
                }
            }
        }

        // Phase 2: Find new joints (not in mapping, not disabled)
        let new_joints: Vec<(crate::Entity, super::components3d::ImpulseJoint3D)> = {
            let physics = world.resource::<PhysicsWorld3D>();
            world
                .iter_entities()
                .filter(|e| !physics.entity_to_joint.contains_key(e) && !world.is_disabled(*e))
                .filter_map(|entity| {
                    let joint = world
                        .get::<super::components3d::ImpulseJoint3D>(entity)?
                        .clone();
                    Some((entity, joint))
                })
                .collect()
        };

        if !new_joints.is_empty() {
            let mut handles = Vec::new();
            {
                let mut physics = world.resource_mut::<PhysicsWorld3D>();
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
                let _ = world.insert(entity, ImpulseJoint3DHandle(handle));
            }
        }

        Ok(())
    }
}

// ---- Regular system variants ----

/// Regular system variant of [`SyncPhysicsBodies3D`].
///
/// Uses lock-execute + deferred commands instead of exclusive world access.
/// Allows parallel scheduling but joints may lag 1 frame behind body creation.
pub struct SyncPhysicsBodiesSystem3D;

impl crate::System for SyncPhysicsBodiesSystem3D {
    type Result = ();

    fn run<'a>(
        &'a self,
        ctx: &'a crate::SystemContext<'a>,
    ) -> Result<(), crate::system::SystemError> {
        redlilium_core::profile_scope!("ecs: sync_physics_bodies_system_3d");

        let (new_indices, stale_entities) = ctx
            .lock::<(
                crate::ResMut<PhysicsWorld3D>,
                crate::Read<super::components3d::RigidBody3D>,
                crate::Read<super::components3d::Collider3D>,
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
                // Remove handle components for stale entities
                for entity in stale_entities {
                    if world.is_alive(entity) {
                        world.remove::<RigidBody3DHandle>(entity);
                    }
                }
                // Insert handles and update mapping for new bodies
                for (idx, handle) in new_indices {
                    if let Some(entity) = world.entity_at_index(idx) {
                        let _ = world.insert(entity, RigidBody3DHandle(handle));
                        let mut physics = world.resource_mut::<PhysicsWorld3D>();
                        physics.entity_to_body.insert(entity, handle);
                        physics.body_to_entity.insert(handle, entity);
                    }
                }
            });
        }

        Ok(())
    }
}

/// Regular system variant of [`SyncPhysicsJoints3D`].
///
/// Uses lock-execute + deferred commands. Joint creation may lag 1 frame behind
/// body creation when both are spawned in the same frame.
pub struct SyncPhysicsJointsSystem3D;

impl crate::System for SyncPhysicsJointsSystem3D {
    type Result = ();

    fn run<'a>(
        &'a self,
        ctx: &'a crate::SystemContext<'a>,
    ) -> Result<(), crate::system::SystemError> {
        redlilium_core::profile_scope!("ecs: sync_physics_joints_system_3d");

        let (new_indices, stale_entities) = ctx
            .lock::<(
                crate::ResMut<PhysicsWorld3D>,
                crate::Read<super::components3d::ImpulseJoint3D>,
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
                        world.remove::<ImpulseJoint3DHandle>(entity);
                    }
                }
                for (idx, handle) in new_indices {
                    if let Some(entity) = world.entity_at_index(idx) {
                        let _ = world.insert(entity, ImpulseJoint3DHandle(handle));
                        let mut physics = world.resource_mut::<PhysicsWorld3D>();
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
    fn physics_world_default() {
        let world = PhysicsWorld3D::default();
        assert!((world.gravity.y - (-9.81)).abs() < 1e-10);
        assert_eq!(world.bodies.len(), 0);
        assert_eq!(world.colliders.len(), 0);
    }

    #[test]
    fn add_body_and_collider() {
        let mut physics = PhysicsWorld3D::default();

        let body_handle = physics.add_body(
            RigidBodyBuilder::dynamic()
                .translation(Vector::new(0.0, 10.0, 0.0))
                .build(),
        );
        let _collider_handle =
            physics.add_collider(ColliderBuilder::ball(0.5).build(), body_handle);

        assert_eq!(physics.bodies.len(), 1);
        assert_eq!(physics.colliders.len(), 1);
    }

    #[test]
    fn step_moves_dynamic_body() {
        let mut physics = PhysicsWorld3D::default();

        let body_handle = physics.add_body(
            RigidBodyBuilder::dynamic()
                .translation(Vector::new(0.0, 10.0, 0.0))
                .build(),
        );
        physics.add_collider(ColliderBuilder::ball(0.5).build(), body_handle);

        let initial_y = physics.bodies[body_handle].position().translation.y;

        // Step a few times
        for _ in 0..10 {
            physics.step();
        }

        let final_y = physics.bodies[body_handle].position().translation.y;
        // Ball should have fallen due to gravity
        assert!(final_y < initial_y);
    }

    #[test]
    fn add_impulse_joint() {
        let mut physics = PhysicsWorld3D::default();

        let b1 = physics.add_body(RigidBodyBuilder::dynamic().build());
        physics.add_collider(ColliderBuilder::ball(0.5).build(), b1);

        let b2 = physics.add_body(
            RigidBodyBuilder::dynamic()
                .translation(Vector::new(2.0, 0.0, 0.0))
                .build(),
        );
        physics.add_collider(ColliderBuilder::ball(0.5).build(), b2);

        let joint = SphericalJointBuilder::new()
            .local_anchor1(Vector::new(1.0, 0.0, 0.0))
            .local_anchor2(Vector::new(-1.0, 0.0, 0.0));

        let _handle = physics.add_impulse_joint(b1, b2, joint);
        assert_eq!(physics.impulse_joints.len(), 1);
    }

    #[test]
    fn remove_body_cleans_colliders() {
        let mut physics = PhysicsWorld3D::default();

        let bh = physics.add_body(RigidBodyBuilder::dynamic().build());
        physics.add_collider(ColliderBuilder::ball(0.5).build(), bh);

        assert_eq!(physics.bodies.len(), 1);
        assert_eq!(physics.colliders.len(), 1);

        physics.remove_body(bh);

        assert_eq!(physics.bodies.len(), 0);
        assert_eq!(physics.colliders.len(), 0);
    }

    #[test]
    fn entity_mapping_roundtrip() {
        let mut physics = PhysicsWorld3D::default();
        let entity = crate::Entity::new(42, 0);
        let bh = physics.add_body(RigidBodyBuilder::dynamic().build());

        physics.entity_to_body.insert(entity, bh);
        physics.body_to_entity.insert(bh, entity);

        assert_eq!(physics.entity_for_body(bh), Some(entity));
        assert_eq!(physics.body_for_entity(entity), Some(bh));
    }

    #[test]
    fn sync_bodies_creates_and_removes() {
        use crate::system::run_exclusive_system_once;
        use redlilium_core::math::Vec3;

        let mut world = crate::World::new();
        crate::register_std_components(&mut world);

        // Spawn a dynamic ball
        let e = world.spawn();
        let _ = world.insert(e, super::super::components3d::RigidBody3D::dynamic());
        let _ = world.insert(e, super::super::components3d::Collider3D::ball(0.5));
        let _ = world.insert(
            e,
            crate::Transform::from_translation(Vec3::new(0.0, 10.0, 0.0)),
        );

        // Run sync
        run_exclusive_system_once(&mut SyncPhysicsBodies3D, &mut world).unwrap();

        // Should have handle
        assert!(world.get::<RigidBody3DHandle>(e).is_some());
        {
            let physics = world.resource::<PhysicsWorld3D>();
            assert_eq!(physics.bodies.len(), 1);
            assert!(physics.entity_to_body.contains_key(&e));
        }

        // Now remove the descriptor
        world.remove::<super::super::components3d::RigidBody3D>(e);

        // Run sync again
        run_exclusive_system_once(&mut SyncPhysicsBodies3D, &mut world).unwrap();

        // Should be cleaned up
        assert!(world.get::<RigidBody3DHandle>(e).is_none());
        {
            let physics = world.resource::<PhysicsWorld3D>();
            assert_eq!(physics.bodies.len(), 0);
            assert!(!physics.entity_to_body.contains_key(&e));
        }
    }

    #[test]
    fn sync_bodies_handles_disabled_entities() {
        use crate::system::run_exclusive_system_once;
        use redlilium_core::math::Vec3;

        let mut world = crate::World::new();
        crate::register_std_components(&mut world);

        // Spawn a dynamic ball
        let e = world.spawn();
        let _ = world.insert(e, super::super::components3d::RigidBody3D::dynamic());
        let _ = world.insert(e, super::super::components3d::Collider3D::ball(0.5));
        let _ = world.insert(
            e,
            crate::Transform::from_translation(Vec3::new(0.0, 10.0, 0.0)),
        );

        // Run sync — body should be created
        run_exclusive_system_once(&mut SyncPhysicsBodies3D, &mut world).unwrap();
        assert!(world.get::<RigidBody3DHandle>(e).is_some());
        {
            let physics = world.resource::<PhysicsWorld3D>();
            assert_eq!(physics.bodies.len(), 1);
        }

        // Disable the entity
        let _ = world.insert(e, crate::Disabled);

        // Run sync — body should be removed from rapier
        run_exclusive_system_once(&mut SyncPhysicsBodies3D, &mut world).unwrap();
        {
            let physics = world.resource::<PhysicsWorld3D>();
            assert_eq!(physics.bodies.len(), 0);
            assert!(!physics.entity_to_body.contains_key(&e));
        }

        // Re-enable the entity
        world.remove::<crate::Disabled>(e);

        // Run sync — body should be re-created
        run_exclusive_system_once(&mut SyncPhysicsBodies3D, &mut world).unwrap();
        assert!(world.get::<RigidBody3DHandle>(e).is_some());
        {
            let physics = world.resource::<PhysicsWorld3D>();
            assert_eq!(physics.bodies.len(), 1);
            assert!(physics.entity_to_body.contains_key(&e));
        }
    }
}
