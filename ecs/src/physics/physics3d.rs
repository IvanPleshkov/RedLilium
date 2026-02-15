//! 3D physics resources, systems, and handle components.
//!
//! Uses rapier3d (f64 by default, f32 with `physics-3d-f32` feature).

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

/// Single ECS resource holding all rapier 3D physics state.
///
/// Insert this as a resource in the ECS world. The [`StepPhysics3D`] system
/// steps the simulation and syncs positions back to ECS transforms.
///
/// # Example
///
/// ```ignore
/// let mut world = World::new();
/// world.insert_resource(PhysicsWorld3D::default());
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
}
