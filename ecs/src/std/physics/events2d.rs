//! 2D collision and contact-force events, read through the standard ECS Events/EventCursor API.
use super::rapier2d::prelude::{CollisionEvent as RapierEvent, *};
super::event_support::collision_events!(CollisionParticipant2D, CollisionEvent2D);

use redlilium_core::math::Vec2 as EventVector;
fn event_vector(v: Vector) -> EventVector {
    EventVector::new(v.x as f32, v.y as f32)
}
super::force_support::contact_force_events!(
    CollisionParticipant2D,
    ContactForceEvent2D,
    ContactForceSample2D
);
