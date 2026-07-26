//! Generic scene-viewport extension points: context-menu operations and
//! pluggable viewport tools.
//!
//! Both registries are world resources filled by the editor's built-ins and
//! by game plugins (via `Plugin::build_editing_view`); the editor shell only
//! renders/routes them and knows nothing about what they do. All mutation
//! flows through the [`ActionQueue`] — neither ops nor tools ever receive a
//! mutable world (HARD RULE 1 is structural here, not a convention).

use redlilium_core::abstract_editor::ActionQueue;
use redlilium_core::math::Vec3;

use crate::{Entity, World};

// ---------------------------------------------------------------------------
// Context-menu operations
// ---------------------------------------------------------------------------

/// Context handed to a [`ViewportOp`]: read-only world, the action queue,
/// the current selection, and a slot to request tool activation.
pub struct ViewportOpCtx<'a> {
    pub world: &'a World,
    pub actions: &'a ActionQueue<World>,
    pub selection: &'a [Entity],
    /// World-space ray under the position the menu was opened at (`None`
    /// when the shell cannot unproject, e.g. no camera yet). Lets ops place
    /// things at the click point ("add X here" stamps).
    pub cursor_ray: Option<ViewportRay>,
    /// An op can set this to a registered tool's label; the shell activates
    /// that tool after the menu closes (e.g. "Add road" arms the connect
    /// tool instead of editing anything itself).
    pub request_tool: Option<String>,
}

/// One entry in the viewport's right-click menu.
pub struct ViewportOp {
    label: String,
    enabled: Box<dyn Fn(&ViewportOpCtx<'_>) -> bool + Send + Sync>,
    run: Box<dyn Fn(&mut ViewportOpCtx<'_>) + Send + Sync>,
}

impl ViewportOp {
    pub fn label(&self) -> &str {
        &self.label
    }

    pub fn is_enabled(&self, ctx: &ViewportOpCtx<'_>) -> bool {
        (self.enabled)(ctx)
    }

    pub fn run(&self, ctx: &mut ViewportOpCtx<'_>) {
        (self.run)(ctx)
    }
}

/// Registry behind the viewport context menu. Created with the editor
/// built-ins; plugins append their own operations.
#[derive(Default)]
pub struct ViewportOps {
    ops: Vec<ViewportOp>,
}

impl ViewportOps {
    /// Editor built-ins only.
    pub fn with_builtins() -> Self {
        let mut ops = Self::default();
        ops.add(
            "Delete selected",
            |ctx| !ctx.selection.is_empty(),
            |ctx| {
                for &entity in top_most(ctx.world, ctx.selection) {
                    let parent = ctx.world.get::<crate::Parent>(entity).map(|p| p.0);
                    ctx.actions
                        .push(Box::new(super::DeleteEntityAction::new(entity, parent)));
                }
            },
        );
        ops
    }

    /// Register a menu operation. `enabled` greys the entry out; `run` fires
    /// on click — push [`EditAction`](redlilium_core::abstract_editor::EditAction)s
    /// or request a tool, never mutate directly.
    pub fn add(
        &mut self,
        label: impl Into<String>,
        enabled: impl Fn(&ViewportOpCtx<'_>) -> bool + Send + Sync + 'static,
        run: impl Fn(&mut ViewportOpCtx<'_>) + Send + Sync + 'static,
    ) {
        self.ops.push(ViewportOp {
            label: label.into(),
            enabled: Box::new(enabled),
            run: Box::new(run),
        });
    }

    pub fn iter(&self) -> impl Iterator<Item = &ViewportOp> {
        self.ops.iter()
    }

    pub fn is_empty(&self) -> bool {
        self.ops.is_empty()
    }
}

/// Filter a selection down to its top-most entities: an entity whose
/// ancestor is also selected is dropped (deleting the ancestor's subtree
/// already covers it — a second delete action would fail on a dead entity).
fn top_most<'a>(world: &World, selection: &'a [Entity]) -> impl Iterator<Item = &'a Entity> {
    selection.iter().filter(move |&&entity| {
        let mut cursor = entity;
        loop {
            let Some(parent) = world.get::<crate::Parent>(cursor).map(|p| p.0) else {
                return true;
            };
            if selection.contains(&parent) {
                return false;
            }
            cursor = parent;
        }
    })
}

// ---------------------------------------------------------------------------
// Viewport tools
// ---------------------------------------------------------------------------

/// A world-space ray under the cursor.
#[derive(Debug, Clone, Copy)]
pub struct ViewportRay {
    pub origin: Vec3,
    pub dir: Vec3,
}

/// Per-frame input the shell/runner prepares for the active tool. The tool
/// never sees raw window events — clicks reach it only when the gesture
/// router granted the tool ownership (egui and an armed gizmo rank higher).
#[derive(Default)]
pub struct ViewportToolInput {
    /// Cursor in scene-image pixels; `None` while outside the scene view.
    pub cursor: Option<[f32; 2]>,
    /// Scene-image size in pixels.
    pub scene_size: [f32; 2],
    /// World-space ray under the cursor (camera unprojection).
    pub ray: Option<ViewportRay>,
    /// A primary click was routed to the tool this frame (at [`cursor`](Self::cursor)).
    pub clicked: bool,
    /// Escape was pressed this frame (edge).
    pub escape: bool,
}

/// Context for [`ViewportTool::update`]. Read-only world + action queue;
/// tools fetch shared editor resources (e.g. the `DebugDrawer` for previews)
/// from the world themselves.
pub struct ViewportToolCtx<'a> {
    pub world: &'a World,
    pub actions: &'a ActionQueue<World>,
    pub input: &'a ViewportToolInput,
}

/// What the tool wants after an update tick.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolFlow {
    Continue,
    /// Deactivate the tool (the runner calls [`ViewportTool::deactivate`]).
    Finished,
}

/// An interactive viewport mode. While active it owns scene-view clicks:
/// selection is untouched, so multi-step interactions (connect A to B) keep
/// their own state without fighting the selection-driven inspector.
pub trait ViewportTool: Send + Sync {
    /// Stable display/activation name (menu ops request tools by this label).
    fn label(&self) -> &str;
    /// One-line usage hint shown while the tool is active (status bar).
    fn status_hint(&self) -> &str {
        ""
    }
    /// Called every frame while active.
    fn update(&mut self, ctx: &mut ViewportToolCtx<'_>) -> ToolFlow;
    /// Called when the tool is deactivated (Escape, tool switch, `Finished`).
    fn deactivate(&mut self) {}
}

/// Registry + activation state for viewport tools. The shell routes input by
/// [`is_active`](Self::is_active); the runner system drives the active tool.
#[derive(Default)]
pub struct ViewportTools {
    tools: Vec<Box<dyn ViewportTool>>,
    active: Option<usize>,
    pending: Option<String>,
    /// Set by the shell when a tool-owned click is released; consumed by the
    /// runner into [`ViewportToolInput::clicked`].
    pub pending_click: bool,
    /// Runner-side edge detector for Escape.
    pub prev_escape: bool,
}

impl ViewportTools {
    pub fn add(&mut self, tool: Box<dyn ViewportTool>) {
        self.tools.push(tool);
    }

    /// Request activation by label (unknown labels are ignored on resolve).
    pub fn request(&mut self, label: impl Into<String>) {
        self.pending = Some(label.into());
    }

    pub fn is_active(&self) -> bool {
        self.active.is_some()
    }

    pub fn active_label(&self) -> Option<&str> {
        self.active.map(|i| self.tools[i].label())
    }

    /// `(label, hint)` of the active tool, for the shell's mode indicator.
    pub fn active_status(&self) -> Option<(String, String)> {
        self.active.map(|i| {
            (
                self.tools[i].label().to_owned(),
                self.tools[i].status_hint().to_owned(),
            )
        })
    }

    /// Apply a pending activation request (deactivating any current tool).
    pub fn resolve_pending(&mut self) {
        let Some(label) = self.pending.take() else {
            return;
        };
        let target = self.tools.iter().position(|t| t.label() == label);
        if target == self.active {
            return;
        }
        self.deactivate_active();
        self.active = target;
    }

    pub fn deactivate_active(&mut self) {
        if let Some(i) = self.active.take() {
            self.tools[i].deactivate();
        }
    }

    /// Run the active tool for one frame; deactivates it on `Finished`.
    pub fn run_active(&mut self, ctx: &mut ViewportToolCtx<'_>) {
        let Some(i) = self.active else {
            return;
        };
        if self.tools[i].update(ctx) == ToolFlow::Finished {
            self.tools[i].deactivate();
            self.active = None;
        }
    }
}

// ---------------------------------------------------------------------------
// Viewport pickers
// ---------------------------------------------------------------------------

/// What the editor resolved under a select click, handed to the plugins'
/// pickers: the cursor ray, the scene entity the GPU entity-index pass
/// found (a mesh), and the exact world-space point of that hit (the picking
/// pass's depth, unprojected) so a picker can compare depth.
pub struct ViewportPickQuery {
    pub ray: ViewportRay,
    /// The scene entity under the cursor per the GPU pass, if any.
    pub scene_entity: Option<Entity>,
    /// World-space surface point under the pick, from the picking pass's
    /// depth output; `None` when the pick hit no rendered geometry.
    pub scene_point: Option<Vec3>,
}

/// A picker's answer: the entity it claims, and the hit's distance along
/// the ray (`origin + dir * t`) — comparable against other pickers' hits.
#[derive(Debug, Clone, Copy)]
pub struct ViewportPickHit {
    pub entity: Entity,
    pub t: f32,
}

/// Signature of a registered CPU picker.
pub type ViewportPicker =
    Box<dyn Fn(&World, &ViewportPickQuery) -> Option<ViewportPickHit> + Send + Sync>;

/// CPU pick overrides for viewport click-selection (a world resource).
///
/// The editor's click-select resolves through the GPU entity-index pass,
/// which only sees mesh renderers. Entities that draw as overlay lines
/// (road nodes, graph handles, …) register a picker here. Each picker sees
/// the editor's own resolution ([`ViewportPickQuery`]) and decides: return
/// a hit to override it — overlay controls may legitimately sit in front
/// of scene meshes — or `None` to leave the editor's result alone (e.g.
/// when the scene hit is closer to the camera than the control).
#[derive(Default)]
pub struct ViewportPickers {
    pickers: Vec<ViewportPicker>,
}

impl ViewportPickers {
    pub fn add(&mut self, picker: ViewportPicker) {
        self.pickers.push(picker);
    }

    /// Final click resolution: the nearest plugin hit (along the ray) wins;
    /// with no plugin hits the editor's scene entity stands.
    pub fn resolve(&self, world: &World, query: &ViewportPickQuery) -> Option<Entity> {
        self.pickers
            .iter()
            .filter_map(|p| p(world, query))
            .min_by(|a, b| a.t.total_cmp(&b.t))
            .map(|hit| hit.entity)
            .or(query.scene_entity)
    }

    pub fn is_empty(&self) -> bool {
        self.pickers.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct CountingTool {
        label: &'static str,
        updates: u32,
        deactivations: u32,
        finish_after: u32,
    }

    impl ViewportTool for CountingTool {
        fn label(&self) -> &str {
            self.label
        }
        fn update(&mut self, _ctx: &mut ViewportToolCtx<'_>) -> ToolFlow {
            self.updates += 1;
            if self.updates >= self.finish_after {
                ToolFlow::Finished
            } else {
                ToolFlow::Continue
            }
        }
        fn deactivate(&mut self) {
            self.deactivations += 1;
        }
    }

    fn ctx_world() -> (World, ActionQueue<World>) {
        (World::new(), ActionQueue::new())
    }

    #[test]
    fn pickers_override_or_keep_the_scene_hit() {
        let mut world = World::new();
        let scene = world.spawn();
        let control = world.spawn();
        let near_control = world.spawn();
        let query = |scene_entity| ViewportPickQuery {
            ray: ViewportRay {
                origin: Vec3::new(0.0, 10.0, 0.0),
                dir: Vec3::new(0.0, -1.0, 0.0),
            },
            scene_entity,
            scene_point: None,
        };

        // No pickers: the editor's resolution stands.
        let mut pickers = ViewportPickers::default();
        assert_eq!(pickers.resolve(&world, &query(Some(scene))), Some(scene));
        assert_eq!(pickers.resolve(&world, &query(None)), None);

        // A declining picker leaves the scene hit alone.
        pickers.add(Box::new(|_, _| None));
        assert_eq!(pickers.resolve(&world, &query(Some(scene))), Some(scene));

        // A claiming picker overrides; the nearest claim wins.
        pickers.add(Box::new(move |_, _| {
            Some(ViewportPickHit {
                entity: control,
                t: 5.0,
            })
        }));
        assert_eq!(pickers.resolve(&world, &query(Some(scene))), Some(control));
        pickers.add(Box::new(move |_, _| {
            Some(ViewportPickHit {
                entity: near_control,
                t: 2.0,
            })
        }));
        assert_eq!(
            pickers.resolve(&world, &query(Some(scene))),
            Some(near_control)
        );
    }

    #[test]
    fn activation_and_finish_lifecycle() {
        let (world, actions) = ctx_world();
        let input = ViewportToolInput::default();
        let mut tools = ViewportTools::default();
        tools.add(Box::new(CountingTool {
            label: "connect",
            updates: 0,
            deactivations: 0,
            finish_after: 2,
        }));

        tools.request("connect");
        tools.resolve_pending();
        assert_eq!(tools.active_label(), Some("connect"));

        let mut ctx = ViewportToolCtx {
            world: &world,
            actions: &actions,
            input: &input,
        };
        tools.run_active(&mut ctx); // 1st update: Continue
        assert!(tools.is_active());
        tools.run_active(&mut ctx); // 2nd update: Finished
        assert!(!tools.is_active());
    }

    #[test]
    fn unknown_label_activates_nothing() {
        let mut tools = ViewportTools::default();
        tools.request("no-such-tool");
        tools.resolve_pending();
        assert!(!tools.is_active());
    }

    #[test]
    fn delete_selected_skips_nested_children() {
        let mut world = World::new();
        crate::register_std_components(&mut world);
        let parent = world.spawn();
        let child = world.spawn();
        crate::std::hierarchy::set_parent(&mut world, child, parent);
        let selection = vec![parent, child];
        let survivors: Vec<Entity> = top_most(&world, &selection).copied().collect();
        assert_eq!(survivors, vec![parent]);
    }
}
