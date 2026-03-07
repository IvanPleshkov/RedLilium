//! 2D physics world resource and handle components.

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
}
