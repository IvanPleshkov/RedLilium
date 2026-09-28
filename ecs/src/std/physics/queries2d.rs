//! Read-only 2D spatial queries. Run StepPhysics2D after sync/teleports before
//! querying updated positions. Queries never advance or synchronize the world.
use super::components2d::ColliderShape2D as ShapeDesc;
use super::control2d::PhysicsPose2D as QueryPose;
use super::rapier2d::parry::query::{
    ShapeCastOptions as BackendOptions, ShapeCastStatus as BackendStatus,
    contact as backend_contact,
};
pub use super::rapier2d::prelude::QueryFilter;
use super::rapier2d::prelude::*;
use super::world2d::PhysicsWorld2D as Physics;
use super::{PhysicsQueryError, RayCastOptions, ShapeCastOptions, ShapeCastStatus};
use redlilium_core::math::Vec2 as GameVector;
use std::ops::ControlFlow;

fn to_vector(v: GameVector) -> Vector {
    Vector::new(v.x as Real, v.y as Real)
}
fn to_game(v: Vector) -> GameVector {
    GameVector::new(v.x as f32, v.y as f32)
}
fn checked_pose(pose: QueryPose) -> Result<Pose, PhysicsQueryError> {
    if !pose.translation.iter().all(|x| x.is_finite()) || !pose.rotation.is_finite() {
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
    }
}
super::query_support::spatial_queries!(
    QueryTarget2D,
    RayHit2D,
    ShapeCastHit2D,
    ShapeCastGeometry2D
);
