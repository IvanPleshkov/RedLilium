//! 3D collision events, read through the standard ECS Events/EventCursor API.
use super::rapier3d::prelude::{CollisionEvent as RapierEvent, *};
super::event_support::collision_events!(CollisionParticipant3D, CollisionEvent3D);
