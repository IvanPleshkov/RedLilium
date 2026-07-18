//! Graph evaluation: the canonical road-patch builder with auto-tangents.
//!
//! A [`RoadSegment`]'s `tangent_*` fields are overrides: any value `≤ 0`
//! means **auto**. Auto follows Catmull-Rom: a lone segment uses
//! `|chord| / 3`; at a node shared by exactly two segments the magnitude is
//! `|far_next − far_prev| / 6` — identical on both sides of the node, so
//! chains are C1 (equal parametric speed across the joint), not merely
//! direction-continuous. Directions always stay the nodes' +Z headings —
//! authoring controls where a road *points*, auto only controls how far it
//! keeps pointing there.

use redlilium_core::math::Vec3;
use redlilium_ecs::{Entity, GlobalTransform, World};

use crate::bezier::{self, Patch};
use crate::{RoadNode, RoadSegment};

/// A node's world-space center (the cross-section midpoint).
pub(crate) fn node_center(world: &World, node: Entity) -> Option<Vec3> {
    let gt = world.get::<GlobalTransform>(node)?;
    Some(Vec3::new(gt.0[(0, 3)], gt.0[(1, 3)], gt.0[(2, 3)]))
}

/// Resolve one end's tangent length. `node` is the end being resolved,
/// `far` the segment's other end.
fn auto_tangent(world: &World, road: Entity, node: Entity, far: Entity) -> f32 {
    let Some(node_c) = node_center(world, node) else {
        return 1.0;
    };
    let Some(far_c) = node_center(world, far) else {
        return 1.0;
    };
    let chord_third = ((far_c - node_c).norm() / 3.0).max(0.1);

    // Exactly one *other* segment sharing this node → Catmull-Rom through
    // the two far ends. Zero or several: fall back to the chord rule.
    let mut other_far: Option<Entity> = None;
    let Ok(segments) = world.read_all::<RoadSegment>() else {
        return chord_third;
    };
    for (index, seg) in segments.iter() {
        let Some(entity) = world.entity_at_index(index) else {
            continue;
        };
        if entity == road {
            continue;
        }
        let far_end = if seg.a == node {
            Some(seg.b)
        } else if seg.b == node {
            Some(seg.a)
        } else {
            None
        };
        if let Some(far_end) = far_end {
            if other_far.is_some() {
                return chord_third; // 3+ segments meet here: ambiguous
            }
            other_far = Some(far_end);
        }
    }
    match other_far.and_then(|q| node_center(world, q)) {
        Some(q_c) => {
            let cr = (q_c - far_c).norm() / 6.0;
            if cr > 0.1 { cr } else { chord_third }
        }
        None => chord_third,
    }
}

/// Both resolved tangent lengths for a road segment (overrides honored).
pub fn segment_tangents(world: &World, road: Entity, seg: &RoadSegment) -> (f32, f32) {
    let a = if seg.tangent_a > 0.0 {
        seg.tangent_a
    } else {
        auto_tangent(world, road, seg.a, seg.b)
    };
    let b = if seg.tangent_b > 0.0 {
        seg.tangent_b
    } else {
        auto_tangent(world, road, seg.b, seg.a)
    };
    (a, b)
}

/// The canonical road-patch builder: nodes + resolved tangents. `None` when
/// the segment's ends are gone.
pub fn road_patch(world: &World, road: Entity) -> Option<Patch> {
    let seg = world.get::<RoadSegment>(road)?;
    let (a_gt, a_node) = (
        world.get::<GlobalTransform>(seg.a)?,
        world.get::<RoadNode>(seg.a)?,
    );
    let (b_gt, b_node) = (
        world.get::<GlobalTransform>(seg.b)?,
        world.get::<RoadNode>(seg.b)?,
    );
    let (tangent_a, tangent_b) = segment_tangents(world, road, seg);
    Some(bezier::patch_from_nodes(
        &a_gt.0,
        a_node.half_width,
        tangent_a,
        &b_gt.0,
        b_node.half_width,
        tangent_b,
        seg.b_from_front,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;
    use redlilium_core::math::quat_from_rotation_y;
    use redlilium_ecs::Transform;

    fn world_with_types() -> World {
        let mut world = World::new();
        redlilium_ecs::register_std_components(&mut world);
        world.register_inspector_default::<RoadNode>();
        world.register_inspector_default::<RoadSegment>();
        world
    }

    fn node(world: &mut World, x: f32, z: f32, yaw: f32) -> Entity {
        let e = world.spawn();
        let t = Transform::new(
            Vec3::new(x, 0.0, z),
            quat_from_rotation_y(yaw),
            Vec3::new(1.0, 1.0, 1.0),
        );
        world.insert(e, t).unwrap();
        world.insert(e, GlobalTransform(t.to_matrix())).unwrap();
        world.insert(e, RoadNode::default()).unwrap();
        e
    }

    fn road(world: &mut World, a: Entity, b: Entity) -> Entity {
        let e = world.spawn();
        world
            .insert(
                e,
                RoadSegment {
                    a,
                    b,
                    ..RoadSegment::default()
                },
            )
            .unwrap();
        e
    }

    #[test]
    fn lone_segment_uses_chord_third() {
        let mut world = world_with_types();
        let a = node(&mut world, 0.0, 0.0, 0.0);
        let b = node(&mut world, 0.0, 12.0, 0.0);
        let r = road(&mut world, a, b);
        let seg = world.get::<RoadSegment>(r).unwrap().clone();
        let (ta, tb) = segment_tangents(&world, r, &seg);
        assert!((ta - 4.0).abs() < 1e-4);
        assert!((tb - 4.0).abs() < 1e-4);
    }

    #[test]
    fn chain_joint_is_c1() {
        let mut world = world_with_types();
        // Uneven chain: spans of 8 and 20 meters.
        let n1 = node(&mut world, 0.0, 0.0, 0.0);
        let n2 = node(&mut world, 0.0, 8.0, 0.0);
        let n3 = node(&mut world, 0.0, 28.0, 0.0);
        let r1 = road(&mut world, n1, n2);
        let r2 = road(&mut world, n2, n3);

        let s1 = world.get::<RoadSegment>(r1).unwrap().clone();
        let s2 = world.get::<RoadSegment>(r2).unwrap().clone();
        let (_, t1_at_n2) = segment_tangents(&world, r1, &s1);
        let (t2_at_n2, _) = segment_tangents(&world, r2, &s2);
        // Same magnitude on both sides of the shared node (|n3−n1|/6),
        // same direction (n2's +Z) → C1 across the joint.
        assert!((t1_at_n2 - t2_at_n2).abs() < 1e-4);
        assert!((t1_at_n2 - 28.0 / 6.0).abs() < 1e-4);
    }

    #[test]
    fn manual_override_wins() {
        let mut world = world_with_types();
        let a = node(&mut world, 0.0, 0.0, 0.0);
        let b = node(&mut world, 0.0, 12.0, 0.0);
        let r = world.spawn();
        world
            .insert(
                r,
                RoadSegment {
                    a,
                    b,
                    tangent_a: 7.5,
                    ..RoadSegment::default()
                },
            )
            .unwrap();
        let seg = world.get::<RoadSegment>(r).unwrap().clone();
        let (ta, tb) = segment_tangents(&world, r, &seg);
        assert!((ta - 7.5).abs() < 1e-4, "manual override");
        assert!((tb - 4.0).abs() < 1e-4, "auto on the other end");
    }

    #[test]
    fn three_way_share_falls_back_to_chord() {
        let mut world = world_with_types();
        let hub = node(&mut world, 0.0, 0.0, 0.0);
        let e1 = node(&mut world, 0.0, 9.0, 0.0);
        let e2 = node(&mut world, 9.0, 0.0, 1.5);
        let e3 = node(&mut world, -9.0, 0.0, -1.5);
        let r1 = road(&mut world, hub, e1);
        road(&mut world, hub, e2);
        road(&mut world, hub, e3);
        let seg = world.get::<RoadSegment>(r1).unwrap().clone();
        let (t_hub, _) = segment_tangents(&world, r1, &seg);
        assert!((t_hub - 3.0).abs() < 1e-4, "ambiguous joint → chord/3");
    }
}
