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

pub use super::queries2d::{QueryTarget2D, RayHit2D, ShapeCastGeometry2D, ShapeCastHit2D};

/// Temporary access to a body's motion. Dereferences to a read-only Rapier body;
/// only the explicit motion methods below can mutate it. Descriptor settings
/// are edited through ECS components, poses through PhysicsWorld::teleport.
///
/// ```compile_fail
/// use redlilium_ecs::physics::{world2d::PhysicsWorld2D, rapier2d::prelude::*};
/// fn replace(world: &mut PhysicsWorld2D, handle: RigidBodyHandle) {
///     let mut body = world.body_motion(handle).unwrap();
///     *body = RigidBodyBuilder::dynamic().build();
/// }
/// ```
pub struct BodyMotion2D<'a> {
    body: &'a mut RigidBody,
}

impl std::ops::Deref for BodyMotion2D<'_> {
    type Target = RigidBody;
    fn deref(&self) -> &RigidBody {
        self.body
    }
}

impl BodyMotion2D<'_> {
    /// Sets world-space linear velocity, discarding locked axes on dynamic bodies.
    pub fn set_linvel(&mut self, velocity: Vector, wake_up: bool) {
        self.body.set_linvel(
            super::locked_axes::dim2::linear(self.body, velocity),
            wake_up,
        );
    }
    /// Sets world-space angular velocity in radians per second, discarding
    /// locked axes on dynamic bodies.
    pub fn set_angvel(&mut self, velocity: Real, wake_up: bool) {
        self.body.set_angvel(
            super::locked_axes::dim2::angular(self.body, velocity),
            wake_up,
        );
    }
    /// Adds a persistent world-space force.
    pub fn add_force(&mut self, force: Vector, wake_up: bool) {
        self.body.add_force(force, wake_up);
    }
    /// Adds a persistent torque.
    pub fn add_torque(&mut self, torque: Real, wake_up: bool) {
        self.body.add_torque(torque, wake_up);
    }
    /// Adds a force at a world-space point.
    pub fn add_force_at_point(&mut self, force: Vector, point: Vector, wake_up: bool) {
        self.body.add_force_at_point(force, point, wake_up);
    }
    /// Applies an instantaneous world-space impulse.
    pub fn apply_impulse(&mut self, impulse: Vector, wake_up: bool) {
        self.body.apply_impulse(impulse, wake_up);
    }
    /// Applies an instantaneous angular impulse.
    pub fn apply_torque_impulse(&mut self, impulse: Real, wake_up: bool) {
        self.body.apply_torque_impulse(impulse, wake_up);
    }
    /// Applies an impulse at a world-space point.
    pub fn apply_impulse_at_point(&mut self, impulse: Vector, point: Vector, wake_up: bool) {
        self.body.apply_impulse_at_point(impulse, point, wake_up);
    }
    /// Clears accumulated forces.
    pub fn reset_forces(&mut self, wake_up: bool) {
        self.body.reset_forces(wake_up);
    }
    /// Clears accumulated torques.
    pub fn reset_torques(&mut self, wake_up: bool) {
        self.body.reset_torques(wake_up);
    }
    /// Wakes the body using Rapier’s strong/weak wake policy.
    pub fn wake_up(&mut self, strong: bool) {
        self.body.wake_up(strong);
    }
    /// Puts the body to sleep and zeros its velocities.
    pub fn sleep(&mut self) {
        self.body.sleep();
    }
}

// ---- PhysicsWorld2D resource ----

/// Single ECS resource holding all rapier 2D physics state plus entity mapping.
///
/// ECS descriptors and sync systems own creation and removal of physics objects.
/// Collections are exposed read-only. Use [`Self::body_motion`] for forces,
/// impulses and velocities, and [`Self::teleport`] for pose changes.
/// Entity/handle lookup is available through [`Self::body_for_entity`],
/// [`Self::entity_for_body`] and [`Self::joint_for_entity`].
pub struct PhysicsWorld2D {
    pub(super) collision_events: super::events2d::EventState,
    pub(super) teleports: HashMap<
        crate::Entity,
        (
            RigidBodyHandle,
            super::control2d::PhysicsPose2D,
            super::TeleportVelocity,
        ),
    >,
    pub(super) pose_resets: std::collections::HashSet<RigidBodyHandle>,
    pub(super) applied_bodies: HashMap<
        RigidBodyHandle,
        (
            super::components2d::RigidBody2D,
            super::components2d::Collider2D,
            ColliderHandle,
        ),
    >,
    pub(super) applied_joints: HashMap<ImpulseJointHandle, super::components2d::ImpulseJoint2D>,
    pub gravity: Vector,
    pub integration_parameters: IntegrationParameters,
    pub(super) pipeline: PhysicsPipeline,
    pub(super) island_manager: IslandManager,
    pub(super) broad_phase: DefaultBroadPhase,
    pub(super) narrow_phase: NarrowPhase,
    pub(super) bodies: RigidBodySet,
    pub(super) colliders: ColliderSet,
    pub(super) impulse_joints: ImpulseJointSet,
    pub(super) multibody_joints: MultibodyJointSet,
    pub(super) ccd_solver: CCDSolver,
    // Required by Rapier; soft bodies are not exposed by the ECS integration.
    soft_bodies: SoftBodySet,

    /// Maps ECS entity → rapier body handle.
    pub(super) entity_to_body: HashMap<crate::Entity, RigidBodyHandle>,
    /// Maps rapier body handle → ECS entity (reverse lookup for raycasts).
    pub(super) body_to_entity: HashMap<RigidBodyHandle, crate::Entity>,
    /// Maps ECS entity → rapier impulse joint handle.
    pub(super) entity_to_joint: HashMap<crate::Entity, ImpulseJointHandle>,
}

impl Default for PhysicsWorld2D {
    fn default() -> Self {
        Self {
            collision_events: Default::default(),
            teleports: HashMap::new(),
            pose_resets: Default::default(),
            applied_bodies: HashMap::new(),
            applied_joints: HashMap::new(),
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
            soft_bodies: SoftBodySet::new(),
            entity_to_body: HashMap::new(),
            body_to_entity: HashMap::new(),
            entity_to_joint: HashMap::new(),
        }
    }
}

impl PhysicsWorld2D {
    /// Read-only Rapier bodies, including their current poses and velocities.
    ///
    /// ```compile_fail
    /// use redlilium_ecs::physics::{world2d::PhysicsWorld2D, rapier2d::prelude::*};
    /// fn mutate(world: &mut PhysicsWorld2D, handle: RigidBodyHandle) {
    ///     world.bodies().get_mut(handle).unwrap().set_body_type(RigidBodyType::Fixed, true);
    /// }
    /// ```
    pub fn bodies(&self) -> &RigidBodySet {
        &self.bodies
    }

    /// Read-only Rapier colliders.
    pub fn colliders(&self) -> &ColliderSet {
        &self.colliders
    }

    /// Read-only live impulse joints.
    pub fn impulse_joints(&self) -> &ImpulseJointSet {
        &self.impulse_joints
    }

    /// Read-only contact/intersection information from the last physics step.
    pub fn narrow_phase(&self) -> &NarrowPhase {
        &self.narrow_phase
    }

    /// Controls motion without allowing replacement of the body, structural
    /// edits, or changes to descriptor-owned settings. A removed handle returns None.
    pub fn body_motion(&mut self, handle: RigidBodyHandle) -> Option<BodyMotion2D<'_>> {
        self.bodies
            .get_mut(handle)
            .map(|body| BodyMotion2D { body })
    }

    /// Returns the live joint handle owned by an ECS entity.
    pub fn joint_for_entity(&self, entity: crate::Entity) -> Option<ImpulseJointHandle> {
        self.entity_to_joint
            .get(&entity)
            .copied()
            .filter(|handle| self.impulse_joints.contains(*handle))
    }

    /// Creates a new physics world with the given gravity.
    pub fn with_gravity(gravity: Vector) -> Self {
        Self {
            gravity,
            ..Default::default()
        }
    }

    /// Steps the physics simulation by one timestep.
    pub(super) fn step(&mut self) {
        redlilium_core::profile_scope!("rapier2d: step");
        self.collision_events.prepare(&self.colliders, &self.bodies);
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
            &mut self.soft_bodies,
            &mut self.ccd_solver,
            &super::collision_types::CollisionTypeHooks,
            self.collision_events.handler(),
        );
        self.collision_events.finish(
            self.integration_parameters.dt,
            &self.colliders,
            &self.bodies,
            &self.narrow_phase,
            &self.body_to_entity,
        );
    }

    /// Inserts a standalone static collider (e.g. terrain) with no ECS owner.
    /// Its lifetime is caller-owned: remove it explicitly or drop this physics world.
    /// The pose is world-space. It is not serialized or removed by ECS body sync.
    pub fn add_free_collider(&mut self, collider: Collider) -> ColliderHandle {
        let handle = self.colliders.insert(collider);
        self.collision_events
            .forces
            .configure(handle, &mut self.colliders[handle]);
        self.collision_events
            .collider_changed(handle, &self.colliders[handle]);
        handle
    }

    /// Removes only a standalone collider. Returns false for a stale handle or
    /// a collider attached to a body; managed colliders are removed through ECS.
    pub fn remove_free_collider(&mut self, handle: ColliderHandle) -> bool {
        if !self
            .colliders
            .get(handle)
            .is_some_and(|c| c.parent().is_none())
        {
            return false;
        }
        self.collision_events.collider_removed(handle);
        self.colliders
            .remove(
                handle,
                &mut self.island_manager,
                &mut self.bodies,
                &mut self.soft_bodies,
                true,
            )
            .is_some()
    }

    /// Adds a rigid body and returns its handle.
    pub(super) fn add_body(&mut self, body: RigidBody) -> RigidBodyHandle {
        self.bodies.insert(body)
    }

    /// Adds a collider attached to a rigid body and returns its handle.
    pub(super) fn add_collider(
        &mut self,
        collider: Collider,
        parent: RigidBodyHandle,
    ) -> ColliderHandle {
        let handle = self
            .colliders
            .insert_with_parent(collider, parent, &mut self.bodies);
        self.collision_events
            .forces
            .configure(handle, &mut self.colliders[handle]);
        self.collision_events
            .collider_changed(handle, &self.colliders[handle]);
        handle
    }

    /// Adds an impulse joint between two bodies and returns its handle.
    pub(super) fn add_impulse_joint(
        &mut self,
        body1: RigidBodyHandle,
        body2: RigidBodyHandle,
        joint: impl Into<GenericJoint>,
    ) -> ImpulseJointHandle {
        self.impulse_joints.insert(body1, body2, joint, true)
    }

    /// Removes a rigid body and all its attached colliders and joints.
    pub(super) fn remove_body(&mut self, handle: RigidBodyHandle) {
        self.collision_events.pending_entities.remove(&handle);
        if let Some(body) = self.bodies.get(handle) {
            for &collider in body.colliders() {
                self.collision_events.collider_removed(collider);
            }
        }
        self.applied_bodies.remove(&handle);
        self.pose_resets.remove(&handle);
        self.teleports.retain(|_, (body, _, _)| *body != handle);
        self.bodies.remove(
            handle,
            &mut self.island_manager,
            &mut self.colliders,
            &mut self.impulse_joints,
            &mut self.multibody_joints,
            &mut self.soft_bodies,
            true,
        );
    }

    /// Removes an impulse joint.
    pub(super) fn remove_impulse_joint(&mut self, handle: ImpulseJointHandle, wake_up: bool) {
        self.applied_joints.remove(&handle);
        self.impulse_joints.remove(handle, wake_up);
    }

    /// Returns the ECS entity that owns the given body handle.
    pub fn entity_for_body(&self, handle: RigidBodyHandle) -> Option<crate::Entity> {
        self.body_to_entity
            .get(&handle)
            .copied()
            .filter(|_| self.bodies.contains(handle))
    }

    /// Returns the rapier body handle for the given ECS entity.
    pub fn body_for_entity(&self, entity: crate::Entity) -> Option<RigidBodyHandle> {
        self.entity_to_body
            .get(&entity)
            .copied()
            .filter(|handle| self.bodies.contains(*handle))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use redlilium_core::math::Vec2 as GameVector;
    type Physics = PhysicsWorld2D;
    super::super::boundary_tests::world_boundary_tests!();
    use super::super::components2d::ColliderShape2D as ShapeDesc;
    use super::super::control2d::PhysicsPose2D as QueryPose;
    fn query_pose(translation: GameVector, angle: f32) -> QueryPose {
        QueryPose {
            translation,
            rotation: angle,
        }
    }
    fn extra_query_shapes(_: &mut Vec<ShapeDesc>) {}
    super::super::query_tests::spatial_query_tests!();

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
