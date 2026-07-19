//! Junctions: N road connectors closed into one boundary loop.
//!
//! A junction is semantic, not geometric (design P1/P5): an entity
//! referencing N ordinary [`RoadNode`] connectors. The boundary is derived
//! every evaluation — cross-sections alternating with **corner curves**,
//! cubic Béziers G1-continuous with the attached roads' side edges (they
//! leave a connector along its inward −Z and arrive along the next
//! connector's outward +Z). The three authoring cases — convex-quad
//! 4-ways, connectors that don't touch, 3-way T/Y — are one continuum:
//! touching connectors just degenerate their corner curves to points.
//! Convention: a connector's **+Z faces outward** (the road side); the
//! junction fills the −Z side.

use redlilium_core::abstract_editor::{EditAction, EditActionError, EditActionResult};
use redlilium_core::math::Vec3;
use redlilium_ecs::{Component, Entity, GlobalTransform, World};

use crate::{RoadNode, bezier};

/// A junction over N connector nodes. Connector order is irrelevant — the
/// loop is re-derived by angle around the centroid on every evaluation, so
/// dragging connectors around never corrupts authored data.
#[derive(Clone, Component)]
pub struct Junction {
    /// Connector entities (each must carry [`RoadNode`]). Dead or non-node
    /// references are skipped at evaluation.
    pub connectors: Vec<Entity>,
    /// Corner-curve tangent length, meters — how far corners keep the
    /// roads' heading before turning (one value per junction for now).
    pub corner_tangent: f32,
}

impl Default for Junction {
    fn default() -> Self {
        Self {
            connectors: Vec::new(),
            corner_tangent: 2.0,
        }
    }
}

/// One resolved connector in loop order.
pub struct LoopConnector {
    pub entity: Entity,
    /// Cross-section boundary points (4, uniform along the segment).
    pub section: [Vec3; 4],
    /// World +Z — outward, toward the road.
    pub outward: Vec3,
}

/// A corner between two adjacent connectors: a cubic Bézier from an
/// endpoint of one cross-section to an endpoint of the next.
pub struct Corner {
    pub points: [Vec3; 4],
}

/// The derived junction boundary: connectors in CCW order (viewed from +Y)
/// with a corner curve after each.
pub struct JunctionLoop {
    pub centroid: Vec3,
    pub connectors: Vec<LoopConnector>,
    /// `corners[i]` joins `connectors[i]` to `connectors[(i + 1) % n]`.
    pub corners: Vec<Corner>,
}

/// Resolve a junction's boundary loop against the world. `None` when fewer
/// than 3 live connectors remain — not enough to enclose an area.
pub fn junction_loop(world: &World, junction: &Junction) -> Option<JunctionLoop> {
    let mut resolved: Vec<LoopConnector> = junction
        .connectors
        .iter()
        .filter_map(|&entity| {
            let node = world.get::<RoadNode>(entity)?;
            let gt = world.get::<GlobalTransform>(entity)?;
            Some(LoopConnector {
                entity,
                section: bezier::cross_section(&gt.0, node.half_width),
                outward: bezier::heading(&gt.0),
            })
        })
        .collect();
    if resolved.len() < 3 {
        return None;
    }

    let centroid = resolved
        .iter()
        .map(center)
        .fold(Vec3::zeros(), |acc, c| acc + c)
        / resolved.len() as f32;

    // CCW around the centroid in the ground (XZ) plane.
    resolved.sort_by(|a, b| {
        let angle = |c: &LoopConnector| {
            let d = center(c) - centroid;
            d.z.atan2(d.x)
        };
        angle(a).total_cmp(&angle(b))
    });

    let n = resolved.len();
    let corners = (0..n)
        .map(|i| {
            let from = &resolved[i];
            let to = &resolved[(i + 1) % n];
            // Adjacent corners meet at the *near* pair of endpoints. Ends of
            // a cross-section are section[0] and section[3].
            let (p0, p3) = nearest_ends(from, to);
            Corner {
                points: [
                    p0,
                    p0 - from.outward * junction.corner_tangent,
                    p3 - to.outward * junction.corner_tangent,
                    p3,
                ],
            }
        })
        .collect();

    Some(JunctionLoop {
        centroid,
        connectors: resolved,
        corners,
    })
}

/// Evaluate a corner's cubic Bézier at `t`.
pub fn eval_corner(corner: &Corner, t: f32) -> Vec3 {
    let s = 1.0 - t;
    let [p0, p1, p2, p3] = corner.points;
    p0 * (s * s * s) + p1 * (3.0 * t * s * s) + p2 * (3.0 * t * t * s) + p3 * (t * t * t)
}

fn center(c: &LoopConnector) -> Vec3 {
    (c.section[0] + c.section[3]) * 0.5
}

/// The closest pair of cross-section endpoints between two connectors.
fn nearest_ends(from: &LoopConnector, to: &LoopConnector) -> (Vec3, Vec3) {
    let mut best = (f32::MAX, from.section[0], to.section[0]);
    for &a in &[from.section[0], from.section[3]] {
        for &b in &[to.section[0], to.section[3]] {
            let d = (a - b).norm_squared();
            if d < best.0 {
                best = (d, a, b);
            }
        }
    }
    (best.1, best.2)
}

/// Transform for a connector `distance` from `center`, `yaw` radians around
/// Y, +Z pointing outward (away from the center) — the 4-way stamp's
/// building block.
pub fn stamp_connector_transform(
    center: Vec3,
    yaw: f32,
    distance: f32,
) -> redlilium_ecs::Transform {
    let rotation = redlilium_core::math::quat_from_rotation_y(yaw);
    let outward = redlilium_core::math::quat_rotate_vec3(rotation, Vec3::new(0.0, 0.0, 1.0));
    redlilium_ecs::Transform::new(
        center + outward * distance,
        rotation,
        Vec3::new(1.0, 1.0, 1.0),
    )
}

// ---------------------------------------------------------------------------
// Edit actions
// ---------------------------------------------------------------------------

/// Undoable "close the selected connector nodes into a junction".
#[derive(Debug)]
pub struct CreateJunctionAction {
    connectors: Vec<Entity>,
    created: Option<Entity>,
}

impl CreateJunctionAction {
    pub fn new(connectors: Vec<Entity>) -> Self {
        Self {
            connectors,
            created: None,
        }
    }
}

impl EditAction<World> for CreateJunctionAction {
    fn apply(&mut self, world: &mut World) -> EditActionResult {
        let live: Vec<Entity> = self
            .connectors
            .iter()
            .copied()
            .filter(|&e| world.is_alive(e) && world.get::<RoadNode>(e).is_some())
            .collect();
        if live.len() < 3 {
            return Err(EditActionError::Custom(
                "a junction needs at least 3 road-node connectors".into(),
            ));
        }
        let entity = world.spawn();
        world
            .insert(
                entity,
                Junction {
                    connectors: live,
                    ..Junction::default()
                },
            )
            .map_err(|e| {
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
        "Create junction"
    }
}

/// Undoable "stamp a 4-way junction template at a point": four outward
/// connectors around `center` plus the junction over them — drag them into
/// shape afterwards, attach roads with the connect tool.
#[derive(Debug)]
pub struct StampJunctionAction {
    center: Vec3,
    created: Vec<Entity>,
}

/// Stamp geometry: connector distance from the stamp center.
const STAMP_ARM: f32 = 8.0;

impl StampJunctionAction {
    pub fn new(center: Vec3) -> Self {
        Self {
            center,
            created: Vec::new(),
        }
    }
}

impl EditAction<World> for StampJunctionAction {
    fn apply(&mut self, world: &mut World) -> EditActionResult {
        let undo_partial = |world: &mut World, created: &mut Vec<Entity>| {
            for e in created.drain(..) {
                world.despawn(e);
            }
        };
        let mut connectors = Vec::with_capacity(4);
        for i in 0..4 {
            let yaw = i as f32 * std::f32::consts::FRAC_PI_2;
            let transform = stamp_connector_transform(self.center, yaw, STAMP_ARM);
            let node = world.spawn();
            self.created.push(node);
            let inserted = world
                .insert(node, transform)
                .and_then(|_| world.insert(node, GlobalTransform(transform.to_matrix())))
                .and_then(|_| world.insert(node, RoadNode::default()));
            if let Err(e) = inserted {
                undo_partial(world, &mut self.created);
                return Err(EditActionError::Custom(e.to_string()));
            }
            connectors.push(node);
        }
        let junction = world.spawn();
        self.created.push(junction);
        if let Err(e) = world.insert(
            junction,
            Junction {
                connectors,
                ..Junction::default()
            },
        ) {
            undo_partial(world, &mut self.created);
            return Err(EditActionError::Custom(e.to_string()));
        }
        Ok(())
    }

    fn undo(&mut self, world: &mut World) -> EditActionResult {
        for entity in self.created.drain(..) {
            world.despawn(entity);
        }
        Ok(())
    }

    fn description(&self) -> &str {
        "Add 4-way junction"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use redlilium_core::math::quat_from_rotation_y;
    use redlilium_ecs::Transform;

    fn world_with_nodes() -> World {
        let mut world = World::new();
        redlilium_ecs::register_std_components(&mut world);
        world.register_inspector_default::<RoadNode>();
        world.register_inspector_default::<Junction>();
        world
    }

    fn spawn_connector(world: &mut World, x: f32, z: f32, yaw: f32) -> Entity {
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

    /// A symmetric 4-way cross: 4 connectors 8 m out, facing outward.
    fn cross(world: &mut World) -> Junction {
        use std::f32::consts::FRAC_PI_2;
        let connectors = (0..4)
            .map(|i| {
                let yaw = i as f32 * FRAC_PI_2;
                let (sin, cos) = yaw.sin_cos();
                spawn_connector(world, sin * 8.0, cos * 8.0, yaw)
            })
            .collect();
        Junction {
            connectors,
            corner_tangent: 2.0,
        }
    }

    #[test]
    fn four_way_loop_has_four_corners_between_near_ends() {
        let mut world = world_with_nodes();
        let junction = cross(&mut world);
        let lp = junction_loop(&world, &junction).expect("loop");
        assert_eq!(lp.connectors.len(), 4);
        assert_eq!(lp.corners.len(), 4);
        assert!(
            lp.centroid.norm() < 1e-4,
            "symmetric cross centers at origin"
        );
        for corner in &lp.corners {
            // Near-end pairing: corner endpoints are much closer than the
            // far-end pairing would be (~16 m across the junction).
            let span = (corner.points[0] - corner.points[3]).norm();
            assert!(span < 10.0, "corner spans the short gap, got {span}");
        }
    }

    #[test]
    fn corner_tangents_follow_road_headings() {
        let mut world = world_with_nodes();
        let junction = cross(&mut world);
        let lp = junction_loop(&world, &junction).expect("loop");
        for (i, corner) in lp.corners.iter().enumerate() {
            let from = &lp.connectors[i];
            let to = &lp.connectors[(i + 1) % lp.connectors.len()];
            let depart = (corner.points[1] - corner.points[0]).normalize();
            let arrive = (corner.points[3] - corner.points[2]).normalize();
            // Leaves along the road's inward direction, arrives against the
            // next road's outward — G1 with both side edges.
            assert!((depart + from.outward).norm() < 1e-4);
            assert!((arrive - to.outward).norm() < 1e-4);
        }
    }

    #[test]
    fn dead_connectors_degrade_and_below_three_dissolves() {
        let mut world = world_with_nodes();
        let mut junction = cross(&mut world);
        world.despawn(junction.connectors[0]);
        let lp = junction_loop(&world, &junction).expect("3 live connectors");
        assert_eq!(lp.connectors.len(), 3);
        assert_eq!(lp.corners.len(), 3);
        world.despawn(junction.connectors[1]);
        assert!(junction_loop(&world, &junction).is_none());
        junction.connectors.clear();
        assert!(junction_loop(&world, &junction).is_none());
    }

    #[test]
    fn three_way_loop_works() {
        let mut world = world_with_nodes();
        let connectors = vec![
            spawn_connector(&mut world, 0.0, 8.0, 0.0),
            spawn_connector(&mut world, 7.0, -5.0, 2.2),
            spawn_connector(&mut world, -7.0, -5.0, -2.2),
        ];
        let junction = Junction {
            connectors,
            corner_tangent: 2.0,
        };
        let lp = junction_loop(&world, &junction).expect("loop");
        assert_eq!(lp.corners.len(), 3);
    }

    #[test]
    fn stamp_transform_points_outward() {
        let t = stamp_connector_transform(Vec3::new(10.0, 0.0, 0.0), 0.0, 8.0);
        // yaw 0 → outward is +Z: placed at center + 8 on Z.
        assert!((t.translation - Vec3::new(10.0, 0.0, 8.0)).norm() < 1e-4);
    }
}
