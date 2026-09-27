//! 3D collision and contact-force events, read through the standard ECS Events/EventCursor API.
use super::rapier3d::prelude::{CollisionEvent as RapierEvent, *};
super::event_support::collision_events!(CollisionParticipant3D, CollisionEvent3D);

use redlilium_core::math::Vec3 as EventVector;
fn event_vector(v: Vector) -> EventVector {
    EventVector::new(v.x as f32, v.y as f32, v.z as f32)
}
super::force_support::contact_force_events!(
    CollisionParticipant3D,
    ContactForceEvent3D,
    ContactForceSample3D
);
