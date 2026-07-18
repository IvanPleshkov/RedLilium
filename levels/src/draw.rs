//! Editing-view overlay: draws the road graph through the `DebugDrawer`.

use redlilium_core::math::{Mat4, Vec3, Vec4};
use redlilium_debug_drawer::{DebugDrawer, DebugDrawerContext};
use redlilium_ecs::{GlobalTransform, Res, System, SystemContext, SystemError};

use crate::attachment::{self, EdgeAttachment};
use crate::bezier::{self, Patch};
use crate::junction::{self, Junction};
use crate::{RoadNode, RoadSegment};

/// Node cross-sections + heading arrows (amber).
const NODE_COLOR: [f32; 4] = [1.0, 0.75, 0.1, 1.0];
/// Road patch side edges (cyan) — the curves attachments will land on.
const EDGE_COLOR: [f32; 4] = [0.15, 0.85, 1.0, 1.0];
/// Interior wireframe (dim teal).
const GRID_COLOR: [f32; 4] = [0.05, 0.4, 0.5, 1.0];

/// Longitudinal tessellation of the preview wireframe.
const U_STEPS: usize = 16;
/// Cross-curve tessellation of each rung.
const V_STEPS: usize = 8;

/// Read-only editing-world system: visualizes [`RoadNode`]s and
/// [`RoadSegment`] patches. Segments with a missing/dangling end draw
/// nothing — the graph, not this overlay, is the source of truth.
pub struct DrawLevelGraph;

impl System for DrawLevelGraph {
    type Result = ();

    fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) -> Result<(), SystemError> {
        ctx.lock::<(Res<DebugDrawer>,)>().execute(|(drawer,)| {
            let world = ctx.raw_world();
            let mut draw = drawer.context();

            if let Ok(nodes) = world.read_all::<RoadNode>() {
                for (index, node) in nodes.iter() {
                    let Some(entity) = world.entity_at_index(index) else {
                        continue;
                    };
                    let Some(gt) = world.get::<GlobalTransform>(entity) else {
                        continue;
                    };
                    draw_node(&mut draw, &gt.0, node.half_width);
                }
            }

            if let Ok(attachments) = world.read_all::<EdgeAttachment>() {
                for (_, att) in attachments.iter() {
                    let Some(patch) = attachment::attachment_patch(world, att) else {
                        continue;
                    };
                    draw_patch(&mut draw, &patch);
                    // Interval ticks: mark where the landing cuts the
                    // parent edge (the future curb break).
                    if let Some(road) = attachment::road_patch(world, att.road) {
                        let v_edge = if att.right_edge { 1.0 } else { 0.0 };
                        for u in [att.u_min, att.u_max] {
                            let p = bezier::eval(&road, u, v_edge);
                            let dv = bezier::eval_dv(&road, u, v_edge);
                            let out = dv * if att.right_edge { 1.0 } else { -1.0 };
                            if out.norm() > 1e-6 {
                                let out = out.normalize() * 0.5;
                                let pt = |v: &redlilium_core::math::Vec3| [v.x, v.y, v.z];
                                draw.draw_line(pt(&(p - out)), pt(&(p + out)), NODE_COLOR);
                            }
                        }
                    }
                }
            }

            if let Ok(junctions) = world.read_all::<Junction>() {
                for (_, j) in junctions.iter() {
                    let Some(lp) = junction::junction_loop(world, j) else {
                        continue;
                    };
                    draw_junction(&mut draw, &lp);
                }
            }

            if let Ok(segments) = world.read_all::<RoadSegment>() {
                for (index, _) in segments.iter() {
                    let Some(road) = world.entity_at_index(index) else {
                        continue;
                    };
                    let Some(patch) = crate::graph::road_patch(world, road) else {
                        continue;
                    };
                    draw_patch(&mut draw, &patch);
                }
            }
        });
        Ok(())
    }
}

fn pt(v: &Vec3) -> [f32; 3] {
    [v.x, v.y, v.z]
}

/// Cross-section segment with end ticks and a heading arrow out of its center.
fn draw_node(draw: &mut DebugDrawerContext<'_>, world: &Mat4, half_width: f32) {
    let section = bezier::cross_section(world, half_width);
    let (left, right) = (section[0], section[3]);
    draw.draw_line(pt(&left), pt(&right), NODE_COLOR);

    let fwd = bezier::heading(world);
    for end in [left, right] {
        draw.draw_line(pt(&(end - fwd * 0.4)), pt(&(end + fwd * 0.4)), NODE_COLOR);
    }

    let center4 = world * Vec4::new(0.0, 0.0, 0.0, 1.0);
    let center = Vec3::new(center4.x, center4.y, center4.z);
    let tip = center + fwd * 1.5;
    let side = (right - left).normalize();
    draw.draw_line(pt(&center), pt(&tip), NODE_COLOR);
    draw.draw_line(pt(&tip), pt(&(tip - fwd * 0.5 + side * 0.3)), NODE_COLOR);
    draw.draw_line(pt(&tip), pt(&(tip - fwd * 0.5 - side * 0.3)), NODE_COLOR);
}

/// Junction boundary: corner curves in the edge color (they continue the
/// roads' curb lines), plus a faint fan from the centroid so the enclosed
/// area reads as a surface.
fn draw_junction(draw: &mut DebugDrawerContext<'_>, lp: &junction::JunctionLoop) {
    let pt = |v: &redlilium_core::math::Vec3| [v.x, v.y, v.z];
    const CORNER_STEPS: usize = 10;
    for corner in &lp.corners {
        let mut prev = junction::eval_corner(corner, 0.0);
        for step in 1..=CORNER_STEPS {
            let next = junction::eval_corner(corner, step as f32 / CORNER_STEPS as f32);
            draw.draw_line(pt(&prev), pt(&next), EDGE_COLOR);
            prev = next;
        }
        draw.draw_line(pt(&lp.centroid), pt(&corner.points[0]), GRID_COLOR);
    }
    for connector in &lp.connectors {
        let mid = (connector.section[0] + connector.section[3]) * 0.5;
        draw.draw_line(pt(&lp.centroid), pt(&mid), GRID_COLOR);
    }
}

/// Patch wireframe: bright side edges (v = 0, 1), dim centerline and rungs.
fn draw_patch(draw: &mut DebugDrawerContext<'_>, patch: &Patch) {
    for (v, color) in [(0.0, EDGE_COLOR), (1.0, EDGE_COLOR), (0.5, GRID_COLOR)] {
        let mut prev = bezier::eval(patch, 0.0, v);
        for step in 1..=U_STEPS {
            let next = bezier::eval(patch, step as f32 / U_STEPS as f32, v);
            draw.draw_line(pt(&prev), pt(&next), color);
            prev = next;
        }
    }
    for step in 1..U_STEPS {
        let u = step as f32 / U_STEPS as f32;
        let mut prev = bezier::eval(patch, u, 0.0);
        for sub in 1..=V_STEPS {
            let next = bezier::eval(patch, u, sub as f32 / V_STEPS as f32);
            draw.draw_line(pt(&prev), pt(&next), GRID_COLOR);
            prev = next;
        }
    }
}
