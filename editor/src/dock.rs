use egui_dock::{DockState, NodeIndex, TabViewer};
use redlilium_ecs::World;
use redlilium_ecs::ui::InspectorState;

/// Identifiers for editor dock tabs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Tab {
    WorldInspector,
    SceneView,
    ComponentInspector,
    Assets,
}

impl Tab {
    pub fn title(&self) -> &str {
        match self {
            Tab::WorldInspector => "World Inspector",
            Tab::SceneView => "Scene View",
            Tab::ComponentInspector => "Component Inspector",
            Tab::Assets => "Assets",
        }
    }
}

/// Create the default editor dock layout.
///
/// ```text
/// +------------------+--------------------+------------------+
/// |                  |                    |                  |
/// | World Inspector  |    Scene View      | Component Insp.  |
/// | (20% width)      |    (center)        | (25% width)      |
/// |                  |                    |                  |
/// +------------------+--------------------+------------------+
/// |                     Assets (25% height)                  |
/// +----------------------------------------------------------+
/// ```
pub fn create_default_layout() -> DockState<Tab> {
    let mut dock_state = DockState::new(vec![Tab::SceneView]);
    let surface = dock_state.main_surface_mut();

    // Left: World Inspector (20% of total width)
    let [_left, remainder] = surface.split_left(NodeIndex::root(), 0.20, vec![Tab::WorldInspector]);

    // Right: Component Inspector (25% of remaining width => ~20% of total)
    let [center, _right] = surface.split_right(remainder, 0.75, vec![Tab::ComponentInspector]);

    // Bottom of center: Assets (25% of center height)
    let [_scene, _assets] = surface.split_below(center, 0.75, vec![Tab::Assets]);

    dock_state
}

/// Provides content for each docked tab.
pub struct EditorTabViewer<'a> {
    pub world: &'a mut World,
    pub inspector_state: &'a mut InspectorState,
    /// Output: the SceneView panel rect from this frame (egui logical points).
    pub scene_view_rect: Option<egui::Rect>,
}

impl TabViewer for EditorTabViewer<'_> {
    type Tab = Tab;

    fn title(&mut self, tab: &mut Tab) -> egui::WidgetText {
        tab.title().into()
    }

    fn ui(&mut self, ui: &mut egui::Ui, tab: &mut Tab) {
        match tab {
            Tab::WorldInspector => {
                redlilium_ecs::ui::show_world_inspector(ui, self.world, self.inspector_state);
            }
            Tab::ComponentInspector => {
                redlilium_ecs::ui::show_component_inspector(ui, self.world, self.inspector_state);
            }
            Tab::SceneView => {
                // Record the available rect; the scene pass renders directly
                // to the swapchain in this area.
                self.scene_view_rect = Some(ui.available_rect_before_wrap());
            }
            Tab::Assets => {
                ui.centered_and_justified(|ui| {
                    ui.label("Assets Browser");
                });
            }
        }
    }

    fn clear_background(&self, tab: &Self::Tab) -> bool {
        // SceneView renders directly to the swapchain — don't paint over it.
        !matches!(tab, Tab::SceneView)
    }

    fn closeable(&mut self, _tab: &mut Tab) -> bool {
        false
    }
}
