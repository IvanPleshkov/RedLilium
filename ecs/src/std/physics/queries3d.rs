//! Read-only 3D spatial queries. Run StepPhysics3D after sync/teleports before
//! querying updated positions. Queries never advance or synchronize the world.
use super::components3d::ColliderShape3D as ShapeDesc;
use super::control3d::PhysicsPose3D as QueryPose;
use super::rapier3d::parry::query::{
    ShapeCastOptions as BackendOptions, ShapeCastStatus as BackendStatus,
    contact as backend_contact,
};
pub use super::rapier3d::prelude::QueryFilter;
use super::rapier3d::prelude::*;
use super::world3d::PhysicsWorld3D as Physics;
use super::{PhysicsQueryError, RayCastOptions, ShapeCastOptions, ShapeCastStatus};
use redlilium_core::math::Vec3 as GameVector;
use std::ops::ControlFlow;

fn to_vector(v: GameVector) -> Vector {
    Vector::new(v.x as Real, v.y as Real, v.z as Real)
}
fn to_game(v: Vector) -> GameVector {
    GameVector::new(v.x as f32, v.y as f32, v.z as f32)
}
fn checked_pose(pose: QueryPose) -> Result<Pose, PhysicsQueryError> {
    if !pose.translation.iter().all(|x| x.is_finite())
        || !(pose.rotation.coords.iter().all(|x| x.is_finite())
            && pose.rotation.norm_squared().is_finite()
            && pose.rotation.norm_squared() >= 1e-12)
    {
        return Err(PhysicsQueryError::InvalidPose);
    }
    Ok(pose.to_rapier())
}

// Borrow stack-allocated primitives for the duration of a query. No SharedShape
// construction, temporary collider or per-query geometry allocation is needed.
fn with_shape<R>(
    shape: &ShapeDesc,
    f: impl FnOnce(&dyn Shape) -> R,
) -> Result<R, PhysicsQueryError> {
    let positive = |v: f32| {
        if v.is_finite() && v > 0.0 {
            Ok(())
        } else {
            Err(PhysicsQueryError::InvalidShape)
        }
    };
    match shape {
        ShapeDesc::Compound { .. } => {
            shape
                .validate(crate::Entity::DANGLING)
                .map_err(|_| PhysicsQueryError::InvalidShape)?;
            let shared = shape.to_shared_shape();
            Ok(f(shared.as_ref()))
        }

        ShapeDesc::Ball { radius } => {
            positive(*radius)?;
            Ok(f(&Ball::new(*radius as Real)))
        }
        ShapeDesc::Cuboid { half_extents } => {
            for &v in half_extents.iter() {
                positive(v)?;
            }
            Ok(f(&Cuboid::new(to_vector(*half_extents))))
        }
        ShapeDesc::CapsuleY {
            half_height,
            radius,
        } => {
            positive(*radius)?;
            if !half_height.is_finite() || *half_height < 0.0 {
                return Err(PhysicsQueryError::InvalidShape);
            }
            Ok(f(&Capsule::new_y(*half_height as Real, *radius as Real)))
        }

        ShapeDesc::Cylinder {
            half_height,
            radius,
        } => {
            positive(*half_height)?;
            positive(*radius)?;
            Ok(f(&Cylinder::new(*half_height as Real, *radius as Real)))
        }
    }
}
super::query_support::spatial_queries!(
    QueryTarget3D,
    RayHit3D,
    ShapeCastHit3D,
    ShapeCastGeometry3D
);
