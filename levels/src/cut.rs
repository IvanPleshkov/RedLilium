//! Cuts: terrain-discontinuity lines — where the landscape's continuity
//! deliberately breaks.
//!
//! The terrain default is "flow continuously everywhere" (see
//! [`stroke`](crate::stroke)); a [`Cut`] is the entity that overrides it.
//! Along a cut the surface may crease (C1 break with C0 kept) or step
//! (C0 break — a height jump): pedestal rims, pool walls, moats, small
//! cliffs, embankments. A [`Stroke`](crate::Stroke) stays bare geometry
//! with no obligations; a cut *obliges* the fill — the two are separate
//! entities precisely so the contract is visible in the type.
//!
//! **Master + profile.** The authored path is the **upper lip**, built on
//! the shared vertex machinery — ordered child entities carrying
//! [`StrokeVertex`](crate::StrokeVertex) pen handles. Each vertex adds a
//! [`CutVertex`] profile: `drop` sinks the **derived lower lip** straight
//! down in world Y, `offset` pushes it toward the path's right-hand side
//! (a battered slope/embankment instead of a vertical face; negative
//! overhangs). `drop == 0` merges the lips — the pure-crease case. There
//! is never a second authored curve: the lower lip is derived, so the
//! two can't desync and every attachment keeps a single path
//! parameterization.
//!
//! **Crossings**: a [`Gate`](crate::Gate) dropped on a cut sits on the
//! lip of the side it faces (`flip` → the derived lip) — the socket at
//! the foot of the face a road arrives at. The face between the lips —
//! and the crossing's volume, stairs or a ramp — is generator geometry
//! (tyroxine); the cut owns only the boundary description the generator
//! consumes.

use redlilium_core::abstract_editor::{EditAction, EditActionError, EditActionResult};
use redlilium_core::math::{Vec3, quat_from_rotation_y};
use redlilium_ecs::{
    Component, Entity, GlobalTransform, Transform, World, remove_parent, set_parent,
};

use crate::stroke::StrokeVertex;

/// An open terrain-discontinuity polyline: `points` lists the vertex
/// children in path order, exactly like [`Stroke::points`](crate::Stroke).
/// Each vertex carries [`StrokeVertex`] (path shape) plus [`CutVertex`]
/// (the step profile).
#[derive(Debug, Clone, Default, Component)]
pub struct Cut {
    pub points: Vec<Entity>,
}

/// A cut vertex's step profile. Positive `drop` sinks the lower lip
/// `drop` meters straight down (world Y) beneath the master path; the
/// terrain on the path's **right-hand side** (of travel direction) meets
/// the lower lip, the left side the upper. Zero collapses the step into a
/// crease. A vertex missing this component counts as `drop = 0`.
#[derive(Debug, Clone, Default, Component)]
pub struct CutVertex {
    /// Height of the step at this vertex, meters, interpolated along the
    /// adjacent segments.
    pub drop: f32,
    /// Plan displacement of the derived lip at this vertex, meters,
    /// toward the path's **right-hand side** (the side the drop lowers):
    /// the face batters into a slope/embankment instead of a vertical
    /// wall; negative overhangs. Beware concave bends — an offset beyond
    /// the local curvature radius self-intersects (the classic
    /// offset-curve caveat); the authoring layer does not clamp it.
    pub offset: f32,
}

/// Displace a master-path sample onto the derived lip: sunk `drop`
/// straight down in world Y and pushed `offset` toward the tangent's
/// horizontal right-hand normal (a vertical tangent leaves the offset
/// unapplied).
pub(crate) fn lip_point(p: Vec3, tangent: Vec3, drop: f32, offset: f32) -> Vec3 {
    let mut q = p - Vec3::new(0.0, drop, 0.0);
    let h = Vec3::new(tangent.x, 0.0, tangent.z);
    if h.norm() > 1e-4 {
        let h = h.normalize();
        q += Vec3::new(h.z, 0.0, -h.x) * offset;
    }
    q
}

/// Both lips of the cut, tessellated in world space: `(upper, lower)`,
/// same sample count, index-aligned (sample `i` of the lower lip hangs
/// beneath sample `i` of the upper — the face rungs). Drop and offset
/// interpolate linearly in the segment parameter between the two
/// vertices' profiles. `None` with fewer than 2 live vertices.
pub fn cut_paths(world: &World, cut: &Cut) -> Option<(Vec<Vec3>, Vec<Vec3>)> {
    let corners = crate::stroke::corners_of(world, &cut.points)?;
    let profile: Vec<(f32, f32)> = corners
        .iter()
        .map(|(v, _, _, _)| {
            world
                .get::<CutVertex>(*v)
                .map(|c| (c.drop, c.offset))
                .unwrap_or((0.0, 0.0))
        })
        .collect();
    let geometry: crate::stroke::Corners = corners
        .into_iter()
        .map(|(_, p, h_out, h_in)| (p, h_out, h_in))
        .collect();
    let samples = crate::stroke::tessellate(&geometry);
    let upper: Vec<Vec3> = samples.iter().map(|(_, _, p)| *p).collect();
    let lower: Vec<Vec3> = samples
        .iter()
        .map(|(i, t, p)| {
            let (d0, o0) = profile[*i];
            let (d1, o1) = profile[*i + 1];
            let (_, tangent) = crate::stroke::eval_segment(&geometry, *i, *t);
            lip_point(*p, tangent, d0 + (d1 - d0) * *t, o0 + (o1 - o0) * *t)
        })
        .collect();
    Some((upper, lower))
}

/// [`corners_of`](crate::stroke::corners_of) for a cut's point list —
/// the vertex handle positions (drawing, picking).
pub(crate) fn cut_corners(world: &World, cut: &Cut) -> Option<Vec<(Entity, Vec3, Vec3, Vec3)>> {
    crate::stroke::corners_of(world, &cut.points)
}

/// Default path for a freshly stamped cut: a straight 12 m line along
/// local X — drag the vertices (and tune each `drop`) into shape after.
const DEFAULT_CUT: [[f32; 2]; 3] = [[-6.0, 0.0], [0.0, 0.0], [6.0, 0.0]];

/// Default step height of a freshly stamped cut, meters.
const DEFAULT_DROP: f32 = 2.0;

/// Undoable "stamp a cut": the root entity plus a default straight line of
/// vertex children, each with the default step profile.
#[derive(Debug)]
pub struct AddCutAction {
    transform: Transform,
    created: Vec<Entity>,
}

impl AddCutAction {
    pub fn at_point(point: Vec3) -> Self {
        Self {
            transform: Transform::new(point, quat_from_rotation_y(0.0), Vec3::new(1.0, 1.0, 1.0)),
            created: Vec::new(),
        }
    }
}

impl EditAction<World> for AddCutAction {
    fn apply(&mut self, world: &mut World) -> EditActionResult {
        let undo_partial = |world: &mut World, created: &mut Vec<Entity>| {
            for e in created.drain(..).rev() {
                remove_parent(world, e);
                world.despawn(e);
            }
        };
        let cut = world.spawn();
        self.created.push(cut);
        let inserted = world
            .insert(cut, self.transform)
            .and_then(|_| world.insert(cut, GlobalTransform(self.transform.to_matrix())))
            .and_then(|_| world.insert(cut, redlilium_ecs::Name("Cut".to_owned())));
        if let Err(e) = inserted {
            undo_partial(world, &mut self.created);
            return Err(EditActionError::Custom(e.to_string()));
        }

        let mut points = Vec::with_capacity(DEFAULT_CUT.len());
        for (n, [x, z]) in DEFAULT_CUT.into_iter().enumerate() {
            let local = Transform::new(
                Vec3::new(x, 0.0, z),
                quat_from_rotation_y(0.0),
                Vec3::new(1.0, 1.0, 1.0),
            );
            let vertex = world.spawn();
            self.created.push(vertex);
            let inserted = world
                .insert(vertex, local)
                .and_then(|_| {
                    world.insert(
                        vertex,
                        GlobalTransform(self.transform.to_matrix() * local.to_matrix()),
                    )
                })
                .and_then(|_| world.insert(vertex, StrokeVertex::default()))
                .and_then(|_| {
                    world.insert(
                        vertex,
                        CutVertex {
                            drop: DEFAULT_DROP,
                            offset: 0.0,
                        },
                    )
                })
                .and_then(|_| {
                    world.insert(vertex, redlilium_ecs::Name(format!("Point {}", n + 1)))
                });
            if let Err(e) = inserted {
                undo_partial(world, &mut self.created);
                return Err(EditActionError::Custom(e.to_string()));
            }
            set_parent(world, vertex, cut);
            points.push(vertex);
        }
        if let Err(e) = world.insert(cut, Cut { points }) {
            undo_partial(world, &mut self.created);
            return Err(EditActionError::Custom(e.to_string()));
        }
        Ok(())
    }

    fn undo(&mut self, world: &mut World) -> EditActionResult {
        for e in self.created.drain(..).rev() {
            remove_parent(world, e);
            world.despawn(e);
        }
        Ok(())
    }

    fn description(&self) -> &str {
        "Add cut"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn world() -> World {
        let mut world = World::new();
        redlilium_ecs::register_std_components(&mut world);
        world.register_inspector_default::<Cut>();
        world.register_inspector_default::<CutVertex>();
        world.register_inspector_default::<StrokeVertex>();
        world
    }

    fn spawned_cut(world: &World) -> (Entity, Cut) {
        world
            .read_all::<Cut>()
            .unwrap()
            .iter()
            .filter_map(|(index, c)| Some((world.entity_at_index(index)?, c.clone())))
            .next()
            .unwrap()
    }

    #[test]
    fn add_cut_stamps_two_lips_and_undo_reverts() {
        let mut world = world();
        let mut action = AddCutAction::at_point(Vec3::new(10.0, 1.0, 5.0));
        action.apply(&mut world).unwrap();

        let (cut, component) = spawned_cut(&world);
        assert_eq!(component.points.len(), 3);
        for v in &component.points {
            assert_eq!(world.get::<redlilium_ecs::Parent>(*v).unwrap().0, cut);
            assert!((world.get::<CutVertex>(*v).unwrap().drop - DEFAULT_DROP).abs() < 1e-6);
        }

        let (upper, lower) = cut_paths(&world, &component).expect("lips");
        assert_eq!(upper.len(), 3);
        assert_eq!(lower.len(), 3);
        // Upper lip = the master path at the stamp height; lower lip hangs
        // DEFAULT_DROP straight beneath it — same XZ, world-Y offset only.
        assert!((upper[0] - Vec3::new(4.0, 1.0, 5.0)).norm() < 1e-4);
        assert!((upper[2] - Vec3::new(16.0, 1.0, 5.0)).norm() < 1e-4);
        for (u, l) in upper.iter().zip(&lower) {
            assert!((l - (u - Vec3::new(0.0, DEFAULT_DROP, 0.0))).norm() < 1e-4);
        }

        action.undo(&mut world).unwrap();
        assert!(world.read_all::<Cut>().unwrap().iter().next().is_none());
        assert!(
            world
                .read_all::<CutVertex>()
                .unwrap()
                .iter()
                .next()
                .is_none()
        );
    }

    #[test]
    fn drop_interpolates_along_a_curved_segment() {
        let mut world = world();
        AddCutAction::at_point(Vec3::zeros())
            .apply(&mut world)
            .unwrap();
        let (_, component) = spawned_cut(&world);

        // Curve the first segment ((−6,0,0) → (0,0,0)) with a +Z bulge and
        // ramp the profile: drop 1 m at the start vertex, 3 m at the next.
        let (v0, v1) = (component.points[0], component.points[1]);
        world
            .insert(
                v0,
                StrokeVertex {
                    handle_out: Vec3::new(2.0, 0.0, 2.0),
                    handle_in: Vec3::zeros(),
                },
            )
            .unwrap();
        world
            .insert(
                v1,
                StrokeVertex {
                    handle_in: Vec3::new(-2.0, 0.0, 2.0),
                    handle_out: Vec3::zeros(),
                },
            )
            .unwrap();
        world
            .insert(
                v0,
                CutVertex {
                    drop: 1.0,
                    offset: 0.0,
                },
            )
            .unwrap();
        world
            .insert(
                v1,
                CutVertex {
                    drop: 3.0,
                    offset: 0.0,
                },
            )
            .unwrap();

        let (upper, lower) = cut_paths(&world, &component).expect("lips");
        // One curved segment → its Bézier fan joins the 3 vertices.
        assert_eq!(upper.len(), 3 + (crate::stroke::CURVE_STEPS - 1));
        // Mid-segment sample (t = 0.5): the drop is halfway through the
        // ramp (2 m) and the lower lip hangs beneath the bulged curve.
        let mid = crate::stroke::CURVE_STEPS / 2;
        assert!(
            (upper[mid].z - 1.5).abs() < 1e-3,
            "bulge, got {:?}",
            upper[mid]
        );
        assert!((upper[mid] - lower[mid] - Vec3::new(0.0, 2.0, 0.0)).norm() < 1e-3);
        // Vertex samples take their authored drops exactly.
        assert!((upper[0] - lower[0] - Vec3::new(0.0, 1.0, 0.0)).norm() < 1e-4);
        let v1_sample = crate::stroke::CURVE_STEPS;
        assert!((upper[v1_sample] - lower[v1_sample] - Vec3::new(0.0, 3.0, 0.0)).norm() < 1e-4);
    }

    #[test]
    fn offset_batters_the_face_toward_the_right_hand_side() {
        let mut world = world();
        AddCutAction::at_point(Vec3::zeros())
            .apply(&mut world)
            .unwrap();
        let (_, component) = spawned_cut(&world);
        // Travel is +X, so the right-hand side is −Z: with drop 2 and
        // offset 1 the derived lip runs parallel one meter south, two
        // meters down — a battered slope's foot line.
        for &v in &component.points {
            world
                .insert(
                    v,
                    CutVertex {
                        drop: 2.0,
                        offset: 1.0,
                    },
                )
                .unwrap();
        }
        let (upper, lower) = cut_paths(&world, &component).expect("lips");
        for (u, l) in upper.iter().zip(&lower) {
            assert!((l - (u + Vec3::new(0.0, -2.0, -1.0))).norm() < 1e-4);
        }
        // A negative offset overhangs — the lip tucks under the crest.
        world
            .insert(
                component.points[0],
                CutVertex {
                    drop: 2.0,
                    offset: -1.0,
                },
            )
            .unwrap();
        let (upper, lower) = cut_paths(&world, &component).expect("lips");
        assert!((lower[0] - (upper[0] + Vec3::new(0.0, -2.0, 1.0))).norm() < 1e-4);
    }

    #[test]
    fn zero_drop_merges_the_lips_into_a_crease() {
        let mut world = world();
        AddCutAction::at_point(Vec3::zeros())
            .apply(&mut world)
            .unwrap();
        let (_, component) = spawned_cut(&world);
        // Zero the whole profile — the C0-continuous crease case (C1
        // still breaks in the fill; the lips coincide).
        for &v in &component.points {
            world
                .insert(
                    v,
                    CutVertex {
                        drop: 0.0,
                        offset: 0.0,
                    },
                )
                .unwrap();
        }
        let (upper, lower) = cut_paths(&world, &component).expect("lips");
        for (u, l) in upper.iter().zip(&lower) {
            assert!((u - l).norm() < 1e-6);
        }

        // A vertex missing its profile component counts as drop = 0 too.
        let _ = world.remove::<CutVertex>(component.points[0]);
        world
            .insert(
                component.points[1],
                CutVertex {
                    drop: 2.0,
                    offset: 0.0,
                },
            )
            .unwrap();
        let (upper, lower) = cut_paths(&world, &component).expect("lips");
        assert!((upper[0] - lower[0]).norm() < 1e-6, "missing profile = 0");
        assert!((upper[1] - lower[1] - Vec3::new(0.0, 2.0, 0.0)).norm() < 1e-4);
    }
}
