//! 3D physics world resource and handle components.

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
}
