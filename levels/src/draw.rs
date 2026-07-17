//! Editing-view overlay: draws the road graph through the `DebugDrawer`.

use redlilium_core::math::{Mat4, Vec3, Vec4};
use redlilium_debug_drawer::{DebugDrawer, DebugDrawerContext};
use redlilium_ecs::{GlobalTransform, Res, System, SystemContext, SystemError};

use crate::bezier::{self, Patch};
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

            if let Ok(segments) = world.read_all::<RoadSegment>() {
                for (_, seg) in segments.iter() {
                    let ends = (
                        world.get::<GlobalTransform>(seg.a),
                        world.get::<RoadNode>(seg.a),
                        world.get::<GlobalTransform>(seg.b),
                        world.get::<RoadNode>(seg.b),
                    );
                    let (Some(gta), Some(na), Some(gtb), Some(nb)) = ends else {
                        continue;
                    };
                    let patch = bezier::patch_from_nodes(
                        &gta.0,
                        na.half_width,
                        seg.tangent_a,
                        &gtb.0,
                        nb.half_width,
                        seg.tangent_b,
                    );
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
