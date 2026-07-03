use egui_dock::{DockState, NodeIndex, TabViewer};
use redlilium_core::abstract_editor::EditActionHistory;
use redlilium_ecs::World;
use redlilium_ecs::ui::InspectorState;
use redlilium_vfs::Vfs;

use crate::asset_browser::AssetBrowser;
use crate::console::ConsolePanel;

/// Identifiers for editor dock tabs.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Tab {
    WorldInspector,
    SceneView,
    ComponentInspector,
    Assets,
    Console,
    History,
}

impl Tab {
    pub fn title(&self) -> &str {
        match self {
            Tab::WorldInspector => "World Inspector",
            Tab::SceneView => "Scene View",
            Tab::ComponentInspector => "Component Inspector",
            Tab::Assets => "Assets",
            Tab::Console => "Console",
            Tab::History => "History",
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

    // Bottom: Assets + Console tabs (25% of total height, spanning full width)
    let [top, _bottom] = surface.split_below(
        NodeIndex::root(),
        0.75,
        vec![Tab::Assets, Tab::Console, Tab::History],
    );

    // Left: World Inspector (20% of total width)
    // split_left returns [old, new] — old (SceneView) stays right, new (WorldInspector) goes left
    let [scene_area, _left] = surface.split_left(top, 0.20, vec![Tab::WorldInspector]);

    // Right: Component Inspector (25% of remaining width => ~20% of total)
    let [_center, _right] = surface.split_right(scene_area, 0.75, vec![Tab::ComponentInspector]);

    dock_state
}

/// Provides content for each docked tab.
pub struct EditorTabViewer<'a> {
    pub world: &'a mut World,
    pub inspector_state: &'a mut InspectorState,
    pub vfs: &'a Vfs,
    pub asset_browser: &'a mut AssetBrowser,
    pub console: &'a mut ConsolePanel,
    pub history: &'a EditActionHistory<World>,
    /// Output: the SceneView panel rect from this frame (egui logical points).
    pub scene_view_rect: Option<egui::Rect>,
    /// Optional drag selection rectangle to draw over the SceneView (egui logical points).
    pub drag_rect: Option<egui::Rect>,
    /// egui texture id of the off-screen scene color target (the SceneView image).
    pub scene_texture: Option<egui::TextureId>,
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
                // An asset selected in the browser takes precedence over the
                // entity inspector.
                if let Some((source, path)) = self.asset_browser.selected_file().cloned() {
                    let edited = crate::asset_inspector::show_asset_inspector(
                        ui, self.world, &source, &path,
                    );
                    if edited {
                        self.asset_browser.mark_db_dirty(source);
                    }
                } else {
                    redlilium_ecs::ui::show_component_inspector(
                        ui,
                        self.world,
                        self.inspector_state,
                    );
                }
            }
            Tab::SceneView => {
                // Record the panel rect (drives picking + next frame's target
                // size), then composite the off-screen scene color target here.
                let rect = ui.available_rect_before_wrap();
                self.scene_view_rect = Some(rect);

                if let Some(texture_id) = self.scene_texture {
                    ui.painter().image(
                        texture_id,
                        rect,
                        egui::Rect::from_min_max(egui::pos2(0.0, 0.0), egui::pos2(1.0, 1.0)),
                        egui::Color32::WHITE,
                    );
                }

                // Draw box selection rectangle overlay if dragging.
                if let Some(rect) = self.drag_rect {
                    ui.painter().rect_filled(
                        rect,
                        egui::CornerRadius::ZERO,
                        egui::Color32::from_rgba_unmultiplied(70, 130, 230, 30),
                    );
                    ui.painter().rect_stroke(
                        rect,
                        egui::CornerRadius::ZERO,
                        egui::Stroke::new(1.0, egui::Color32::from_rgb(70, 130, 230)),
                        egui::StrokeKind::Outside,
                    );
                }
            }
            Tab::Assets => {
                // The DB stamps dragged files with their asset identity so
                // inspector reference fields can accept the drop.
                let db = self
                    .world
                    .has_resource::<redlilium_assets::AssetDb>()
                    .then(|| self.world.resource::<redlilium_assets::AssetDb>());
                self.asset_browser.show(ui, self.vfs, db.as_deref());
            }
            Tab::Console => {
                self.console.show(ui);
            }
            Tab::History => {
                crate::history_panel::show_history(ui, self.history);
            }
        }
    }

    fn clear_background(&self, _tab: &Self::Tab) -> bool {
        // SceneView now composites the scene as an egui image, so paint normally.
        true
    }

    fn closeable(&mut self, _tab: &mut Tab) -> bool {
        false
    }
}
