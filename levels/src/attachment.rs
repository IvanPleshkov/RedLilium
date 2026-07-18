//! Edge attachments (design P5): a connection landing on **part of a road's
//! boundary curve** instead of on a node — the driveway/building-exit case.
//!
//! The attachment patch's road-side row is sampled directly from the parent
//! road's edge curve on `[u_min, u_max]` — the seam is watertight by
//! construction, and it *stays* watertight as the road's nodes move, because
//! the interval is parametric, not positional. The patch leaves the edge
//! along the surface's cross derivative (`∂P/∂v`), continuing the road's
//! cross-slope, and ends at the outer node. Socket convention (same as
//! junction connectors): the outer node's **+Z faces the road network** —
//! the driveway departs it along +Z like a road departs its `a` node, and
//! the structure that owns the socket (the future building) fills the −Z
//! side behind it.

use redlilium_core::abstract_editor::{EditAction, EditActionError, EditActionResult};
use redlilium_core::math::Vec3;
use redlilium_ecs::{Component, Entity, GlobalTransform, World};

use crate::bezier::{self, Patch};
use crate::{RoadNode, RoadSegment};

/// A patch hung off a road's side edge, ending at an outer [`RoadNode`].
#[derive(Debug, Clone, Component)]
pub struct EdgeAttachment {
    /// The road entity (carries [`RoadSegment`]) whose edge this lands on.
    pub road: Entity,
    /// The outer cross-section node (carries [`RoadNode`]); its +Z faces
    /// the road (socket convention — the owning structure fills −Z).
    pub node: Entity,
    /// Which side edge: `true` = the `v = 1` edge (the cross-sections'
    /// local +X ends), `false` = the `v = 0` edge.
    pub right_edge: bool,
    /// Landing interval on the road's `u` axis, `0 ≤ u_min < u_max ≤ 1`.
    pub u_min: f32,
    pub u_max: f32,
    /// Tangent length at the road edge, meters (along `∂P/∂v`, outward).
    pub tangent_edge: f32,
    /// Tangent length at the outer node, meters (along its +Z, toward the
    /// road).
    pub tangent_node: f32,
}

impl Default for EdgeAttachment {
    fn default() -> Self {
        Self {
            road: Entity::DANGLING,
            node: Entity::DANGLING,
            right_edge: true,
            u_min: 0.4,
            u_max: 0.6,
            tangent_edge: 2.0,
            tangent_node: 2.0,
        }
    }
}

/// Rebuild the parent road's patch from the world. `None` when the road or
/// its nodes are gone.
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
    Some(bezier::patch_from_nodes(
        &a_gt.0,
        a_node.half_width,
        seg.tangent_a,
        &b_gt.0,
        b_node.half_width,
        seg.tangent_b,
    ))
}

/// Derive the attachment's own patch. Row 0 lies on the road's edge curve;
/// row 3 is the outer node's cross-section (uncrossed if needed).
pub fn attachment_patch(world: &World, att: &EdgeAttachment) -> Option<Patch> {
    let road = road_patch(world, att.road)?;
    let node = world.get::<RoadNode>(att.node)?;
    let node_gt = world.get::<GlobalTransform>(att.node)?;

    let v_edge = if att.right_edge { 1.0 } else { 0.0 };
    let (u_min, u_max) = (att.u_min.min(att.u_max), att.u_min.max(att.u_max));
    let row0 = bezier::sample_edge_row(&road, v_edge, u_min, u_max);

    // Leave the edge continuing the road's cross-slope: ∂P/∂v, flipped on
    // the v = 0 side so the tangent always points outward.
    let sign = if att.right_edge { 1.0 } else { -1.0 };
    let row1: [Vec3; 4] = std::array::from_fn(|i| {
        let u = u_min + (u_max - u_min) * (i as f32 / 3.0);
        let dv = bezier::eval_dv(&road, u, v_edge);
        let out = dv * sign;
        let out = if out.norm() > 1e-6 {
            out.normalize()
        } else {
            Vec3::zeros()
        };
        row0[i] + out * att.tangent_edge
    });

    let mut row3 = bezier::cross_section(&node_gt.0, node.half_width);
    let straight = (row3[0] - row0[0]).norm() + (row3[3] - row0[3]).norm();
    let crossed = (row3[3] - row0[0]).norm() + (row3[0] - row0[3]).norm();
    if crossed < straight {
        row3.reverse();
    }
    // The node's +Z faces the road (socket convention): the control row
    // extends from the node toward the road, like a road leaving its `a`.
    let toward_road = bezier::heading(&node_gt.0);
    let row2 = row3.map(|p| p + toward_road * att.tangent_node);

    Some([row0, row1, row2, row3])
}

/// An edge point under the cursor: which road, which side, where along it.
#[derive(Debug, Clone, Copy)]
pub struct EdgeHit {
    pub road: Entity,
    pub right_edge: bool,
    pub u: f32,
    pub point: Vec3,
}

/// Tessellation used for edge picking.
const PICK_STEPS: usize = 24;

/// CPU pick against every road's two side edges: the closest edge point
/// within `radius` of the ray.
pub fn edge_under_cursor(
    world: &World,
    ray: &redlilium_ecs::ui::ViewportRay,
    radius: f32,
) -> Option<EdgeHit> {
    let roads = world.read_all::<RoadSegment>().ok()?;
    let mut best: Option<(f32, EdgeHit)> = None;
    for (index, _) in roads.iter() {
        let Some(road) = world.entity_at_index(index) else {
            continue;
        };
        let Some(patch) = road_patch(world, road) else {
            continue;
        };
        for (right_edge, v_edge) in [(false, 0.0f32), (true, 1.0f32)] {
            let mut prev = bezier::eval(&patch, 0.0, v_edge);
            for step in 1..=PICK_STEPS {
                let u1 = step as f32 / PICK_STEPS as f32;
                let next = bezier::eval(&patch, u1, v_edge);
                let (dist, s) = crate::tool::ray_segment_closest(ray, prev, next);
                if dist <= radius && best.is_none_or(|(d, _)| dist < d) {
                    let u0 = (step - 1) as f32 / PICK_STEPS as f32;
                    let u = u0 + (u1 - u0) * s;
                    best = Some((
                        dist,
                        EdgeHit {
                            road,
                            right_edge,
                            u,
                            point: prev + (next - prev) * s,
                        },
                    ));
                }
                prev = next;
            }
        }
    }
    best.map(|(_, hit)| hit)
}

/// The `u` half-interval an outer node's width covers around a hit point:
/// the node's full width mapped through the local edge-length density.
pub fn interval_around(world: &World, hit: &EdgeHit, node_half_width: f32) -> (f32, f32) {
    let Some(patch) = road_patch(world, hit.road) else {
        return (hit.u - 0.1, hit.u + 0.1);
    };
    let v_edge = if hit.right_edge { 1.0 } else { 0.0 };
    // Local edge speed |dP/du| via finite difference.
    let h = 1e-3;
    let u = hit.u.clamp(h, 1.0 - h);
    let speed = (bezier::eval(&patch, u + h, v_edge) - bezier::eval(&patch, u - h, v_edge)).norm()
        / (2.0 * h);
    let half_u = if speed > 1e-4 {
        (node_half_width / speed).min(0.45)
    } else {
        0.1
    };
    ((hit.u - half_u).max(0.0), (hit.u + half_u).min(1.0))
}

// ---------------------------------------------------------------------------
// Edit action
// ---------------------------------------------------------------------------

/// Undoable "hang an attachment between an outer node and a road edge".
#[derive(Debug)]
pub struct AttachToEdgeAction {
    attachment: EdgeAttachment,
    created: Option<Entity>,
}

impl AttachToEdgeAction {
    pub fn new(attachment: EdgeAttachment) -> Self {
        Self {
            attachment,
            created: None,
        }
    }
}

impl EditAction<World> for AttachToEdgeAction {
    fn apply(&mut self, world: &mut World) -> EditActionResult {
        if !world.is_alive(self.attachment.road) || !world.is_alive(self.attachment.node) {
            return Err(EditActionError::TargetNotFound(
                "attachment road or node despawned".into(),
            ));
        }
        let entity = world.spawn();
        world.insert(entity, self.attachment.clone()).map_err(|e| {
            world.despawn(entity);
            EditActionError::Custom(e.to_string())
        })?;
        self.created = Some(entity);
        Ok(())
    }

    fn undo(&mut self, world: &mut World) -> EditActionResult {
        if let Some(entity) = self.created.take() {
            world.despawn(entity);
        }
        Ok(())
    }

    fn description(&self) -> &str {
        "Attach to road edge"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use redlilium_core::math::quat_from_rotation_y;
    use redlilium_ecs::Transform;

    fn setup() -> (World, Entity, Entity) {
        let mut world = World::new();
        redlilium_ecs::register_std_components(&mut world);
        world.register_inspector_default::<RoadNode>();
        world.register_inspector_default::<RoadSegment>();
        world.register_inspector_default::<EdgeAttachment>();

        let node = |world: &mut World, x: f32, z: f32, yaw: f32| {
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
        };
        // Straight road along +Z from origin to (0, 0, 20).
        let a = node(&mut world, 0.0, 0.0, 0.0);
        let b = node(&mut world, 0.0, 20.0, 0.0);
        let road = world.spawn();
        world
            .insert(
                road,
                RoadSegment {
                    a,
                    b,
                    ..RoadSegment::default()
                },
            )
            .unwrap();
        // Outer node to the right (+X side); +Z faces the road (socket
        // convention), i.e. toward −X.
        let outer = node(&mut world, 10.0, 10.0, -std::f32::consts::FRAC_PI_2);
        (world, road, outer)
    }

    #[test]
    fn attachment_row_sits_on_road_edge() {
        let (world, road, outer) = setup();
        let att = EdgeAttachment {
            road,
            node: outer,
            right_edge: true,
            u_min: 0.3,
            u_max: 0.6,
            ..EdgeAttachment::default()
        };
        let patch = attachment_patch(&world, &att).expect("patch");
        let road_p = road_patch(&world, road).unwrap();
        for (i, p) in patch[0].iter().enumerate() {
            let u = 0.3 + 0.3 * (i as f32 / 3.0);
            assert!((p - bezier::eval(&road_p, u, 1.0)).norm() < 1e-5);
        }
        // Outer row sits on the node's cross-section, and the patch spans
        // from the road (x = 3) toward the node (x = 10).
        assert!(patch[3].iter().all(|p| (p.x - 10.0).abs() < 3.5));
    }

    #[test]
    fn attachment_survives_road_node_moves_parametrically() {
        let (mut world, road, outer) = setup();
        let att = EdgeAttachment {
            road,
            node: outer,
            right_edge: true,
            u_min: 0.4,
            u_max: 0.6,
            ..EdgeAttachment::default()
        };
        let before = attachment_patch(&world, &att).expect("patch");
        // Move road end b: the edge curve changes, the attachment follows.
        let seg_b = world.get::<RoadSegment>(road).unwrap().b;
        let t = Transform::new(
            Vec3::new(5.0, 0.0, 20.0),
            quat_from_rotation_y(0.3),
            Vec3::new(1.0, 1.0, 1.0),
        );
        world.insert(seg_b, t).unwrap();
        world.insert(seg_b, GlobalTransform(t.to_matrix())).unwrap();
        let after = attachment_patch(&world, &att).expect("patch");
        assert!(
            (before[0][3] - after[0][3]).norm() > 0.1,
            "edge row follows the moved road"
        );
        let road_p = road_patch(&world, road).unwrap();
        assert!((after[0][0] - bezier::eval(&road_p, 0.4, 1.0)).norm() < 1e-5);
    }

    #[test]
    fn dead_road_or_node_dissolves_attachment() {
        let (mut world, road, outer) = setup();
        let att = EdgeAttachment {
            road,
            node: outer,
            ..EdgeAttachment::default()
        };
        assert!(attachment_patch(&world, &att).is_some());
        world.despawn(road);
        assert!(attachment_patch(&world, &att).is_none());
        let att2 = EdgeAttachment {
            road,
            node: outer,
            ..EdgeAttachment::default()
        };
        world.despawn(outer);
        assert!(attachment_patch(&world, &att2).is_none());
    }

    #[test]
    fn attach_action_roundtrip() {
        let (mut world, road, outer) = setup();
        let mut action = AttachToEdgeAction::new(EdgeAttachment {
            road,
            node: outer,
            ..EdgeAttachment::default()
        });
        action.apply(&mut world).unwrap();
        assert_eq!(
            world.read_all::<EdgeAttachment>().unwrap().iter().count(),
            1
        );
        action.undo(&mut world).unwrap();
        assert_eq!(
            world.read_all::<EdgeAttachment>().unwrap().iter().count(),
            0
        );
    }
}
