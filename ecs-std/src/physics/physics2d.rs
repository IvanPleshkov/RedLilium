//! 2D physics resources, systems, and handle components.
//!
//! Uses rapier2d (f64 by default, f32 with `physics-2d-f32` feature).

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

/// Single ECS resource holding all rapier 2D physics state.
///
/// Insert this as a resource in the ECS world. The [`StepPhysics2D`] system
/// steps the simulation and syncs positions back to ECS transforms.
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
}

// ---- StepPhysics2D system ----

/// ECS system that steps the 2D physics simulation and syncs body positions
/// back to ECS [`Transform`](crate::Transform) components.
///
/// For 2D, the X/Y rapier position maps to the Transform's X/Y translation,
/// and the rapier rotation angle maps to a Z-axis rotation quaternion.
pub struct StepPhysics2D;

impl redlilium_ecs::System for StepPhysics2D {
    async fn run<'a>(&'a self, ctx: &'a redlilium_ecs::SystemContext<'a>) {
        ctx.lock::<(
            redlilium_ecs::ResMut<PhysicsWorld2D>,
            redlilium_ecs::Read<RigidBody2DHandle>,
            redlilium_ecs::Write<crate::Transform>,
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
                    transform.translation = glam::Vec3::new(t.x as f32, t.y as f32, 0.0);
                    let angle = pos.rotation.angle() as f32;
                    transform.rotation = glam::Quat::from_rotation_z(angle);
                }
            }
        })
        .await;
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
}
