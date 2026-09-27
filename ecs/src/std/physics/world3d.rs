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

/// Render-side interpolation history for a physics body: the authoritative
/// poses of the two most recent fixed steps.
///
/// Physics is stepped at a fixed rate that need not match the render rate, so
/// showing the raw latest step would judder (a "staircase") whenever the two
/// disagree. [`RecordPhysicsPose`](super::systems3d::RecordPhysicsPose) shifts
/// `cur → prev` and records the new pose each fixed step;
/// [`InterpolatePhysics`](super::systems3d::InterpolatePhysics) blends the two
/// by [`Time::fixed_alpha`](crate::Time::fixed_alpha) into `Transform` for
/// rendering. Dynamic and kinematic bodies never read `Transform` back, so
/// interpolation cannot perturb the simulation. Fixed bodies are not interpolated.
#[derive(Debug, Clone, Copy)]
pub struct PhysicsInterpolation {
    /// Translation after the previous fixed step.
    pub prev_translation: redlilium_core::math::Vec3,
    /// Rotation after the previous fixed step.
    pub prev_rotation: redlilium_core::math::Quat,
    /// Translation after the latest fixed step.
    pub cur_translation: redlilium_core::math::Vec3,
    /// Rotation after the latest fixed step.
    pub cur_rotation: redlilium_core::math::Quat,
}

/// A ray hit independent of whether the collider belongs to an ECS entity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RayHit3D {
    pub collider: ColliderHandle,
    pub body: Option<RigidBodyHandle>,
    pub entity: Option<crate::Entity>,
    /// Ray parameter; distance only when the direction has unit length.
    pub toi: f32,
}

/// Temporary access to a body's motion. Dereferences to a read-only Rapier body;
/// only the explicit motion methods below can mutate it. Descriptor settings
/// are edited through ECS components, poses through PhysicsWorld::teleport.
///
/// ```compile_fail
/// use redlilium_ecs::physics::{world3d::PhysicsWorld3D, rapier3d::prelude::*};
/// fn replace(world: &mut PhysicsWorld3D, handle: RigidBodyHandle) {
///     let mut body = world.body_motion(handle).unwrap();
///     *body = RigidBodyBuilder::dynamic().build();
/// }
/// ```
pub struct BodyMotion3D<'a> {
    body: &'a mut RigidBody,
}

impl std::ops::Deref for BodyMotion3D<'_> {
    type Target = RigidBody;
    fn deref(&self) -> &RigidBody {
        self.body
    }
}

impl BodyMotion3D<'_> {
    /// Sets world-space linear velocity.
    pub fn set_linvel(&mut self, velocity: Vector, wake_up: bool) {
        self.body.set_linvel(velocity, wake_up);
    }
    /// Sets angular velocity in radians per second.
    pub fn set_angvel(&mut self, velocity: Vector, wake_up: bool) {
        self.body.set_angvel(velocity, wake_up);
    }
    /// Adds a persistent world-space force.
    pub fn add_force(&mut self, force: Vector, wake_up: bool) {
        self.body.add_force(force, wake_up);
    }
    /// Adds a persistent torque.
    pub fn add_torque(&mut self, torque: Vector, wake_up: bool) {
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
    pub fn apply_torque_impulse(&mut self, impulse: Vector, wake_up: bool) {
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

// ---- PhysicsWorld3D resource ----

/// Single ECS resource holding all rapier 3D physics state plus entity mapping.
///
/// ECS descriptors and sync systems own creation and removal of physics objects.
/// Collections are exposed read-only. Use [`Self::body_motion`] for forces,
/// impulses and velocities, and [`Self::teleport`] for pose changes.
/// Entity/handle lookup is available through [`Self::body_for_entity`],
/// [`Self::entity_for_body`] and [`Self::joint_for_entity`].
///
/// # Example
///
/// ```ignore
/// // In a system, cast a ray and get the hit entity:
/// ctx.lock::<(Res<PhysicsWorld3D>,)>().execute(|(physics,)| {
///     if let Some(hit) = physics.cast_ray(origin, dir, 100.0) {
///         // `hit.entity` is Some(entity) for an ECS-managed body
///     }
/// });
/// ```
pub struct PhysicsWorld3D {
    pub(super) collision_events: super::events3d::EventState,
    pub(super) teleports: HashMap<
        crate::Entity,
        (
            RigidBodyHandle,
            super::control3d::PhysicsPose3D,
            super::TeleportVelocity,
        ),
    >,
    pub(super) pose_resets: std::collections::HashSet<RigidBodyHandle>,
    pub(super) applied_bodies: HashMap<
        RigidBodyHandle,
        (
            super::components3d::RigidBody3D,
            super::components3d::Collider3D,
            ColliderHandle,
        ),
    >,
    pub(super) applied_joints: HashMap<ImpulseJointHandle, super::components3d::ImpulseJoint3D>,
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

    /// Maps ECS entity → rapier body handle.
    pub(super) entity_to_body: HashMap<crate::Entity, RigidBodyHandle>,
    /// Maps rapier body handle → ECS entity (reverse lookup for raycasts).
    pub(super) body_to_entity: HashMap<RigidBodyHandle, crate::Entity>,
    /// Maps ECS entity → rapier impulse joint handle.
    pub(super) entity_to_joint: HashMap<crate::Entity, ImpulseJointHandle>,
}

impl Default for PhysicsWorld3D {
    fn default() -> Self {
        Self {
            collision_events: Default::default(),
            teleports: HashMap::new(),
            pose_resets: Default::default(),
            applied_bodies: HashMap::new(),
            applied_joints: HashMap::new(),
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
    /// Read-only Rapier bodies, including their current poses and velocities.
    ///
    /// ```compile_fail
    /// use redlilium_ecs::physics::{world3d::PhysicsWorld3D, rapier3d::prelude::*};
    /// fn mutate(world: &mut PhysicsWorld3D, handle: RigidBodyHandle) {
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
    pub fn body_motion(&mut self, handle: RigidBodyHandle) -> Option<BodyMotion3D<'_>> {
        self.bodies
            .get_mut(handle)
            .map(|body| BodyMotion3D { body })
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
        redlilium_core::profile_scope!("rapier3d: step");
        self.collision_events.prepare(&self.colliders);
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
            self.collision_events.handler(),
        );
        self.collision_events
            .finish(&self.colliders, &self.narrow_phase, &self.body_to_entity);
    }

    /// Inserts a standalone static collider (e.g. terrain) with no ECS owner.
    /// Its lifetime is caller-owned: remove it explicitly or drop this physics world.
    /// The pose is world-space. It is not serialized or removed by ECS body sync.
    pub fn add_free_collider(&mut self, collider: Collider) -> ColliderHandle {
        let handle = self.colliders.insert(collider);
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
            .remove(handle, &mut self.island_manager, &mut self.bodies, true)
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

    /// Returns the nearest solid ray hit, including colliders without an ECS entity.
    /// `toi` parameterizes `origin + dir * toi`; it is a distance only for unit `dir`.
    /// Queries use the broad phase from the last physics step: run StepPhysics3D
    /// after syncing bodies or applying teleports before querying their new positions.
    pub fn cast_ray(
        &self,
        origin: redlilium_core::math::Vec3,
        dir: redlilium_core::math::Vec3,
        max_toi: f32,
    ) -> Option<RayHit3D> {
        self.cast_ray_filtered(origin, dir, max_toi, QueryFilter::default())
    }

    /// Like [`Self::cast_ray`], with Rapier collision groups, body/sensor exclusions
    /// and a custom collider predicate applied before choosing the nearest hit.
    pub fn cast_ray_filtered(
        &self,
        origin: redlilium_core::math::Vec3,
        dir: redlilium_core::math::Vec3,
        max_toi: f32,
        filter: QueryFilter<'_>,
    ) -> Option<RayHit3D> {
        let ray = Ray::new(
            Vector::new(origin.x as Real, origin.y as Real, origin.z as Real),
            Vector::new(dir.x as Real, dir.y as Real, dir.z as Real),
        );
        let query_pipeline = self.broad_phase.as_query_pipeline(
            self.narrow_phase.query_dispatcher(),
            &self.bodies,
            &self.colliders,
            filter,
        );
        let (collider, toi) = query_pipeline.cast_ray(&ray, max_toi as Real, true)?;
        let body = self.colliders.get(collider)?.parent();
        let entity = body.and_then(|handle| self.entity_for_body(handle));
        Some(RayHit3D {
            collider,
            body,
            entity,
            toi: toi as f32,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use redlilium_core::math::Vec3 as GameVector;
    type Physics = PhysicsWorld3D;
    super::super::boundary_tests::world_boundary_tests!();

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
