//! 2D collision events, read through the standard ECS Events/EventCursor API.
use super::rapier2d::prelude::{CollisionEvent as RapierEvent, *};
super::event_support::collision_events!(CollisionParticipant2D, CollisionEvent2D);
