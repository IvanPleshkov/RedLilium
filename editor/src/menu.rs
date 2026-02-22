#[cfg(not(target_os = "macos"))]
use std::sync::Arc;

#[cfg(not(target_os = "macos"))]
use winit::window::Window;

/// Actions that can be triggered from the menu.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MenuAction {
    About,
    Save,
    Undo,
    Redo,
    #[allow(dead_code)]
    CloseWindow,
}

// --- Native macOS menu via muda ---

#[cfg(target_os = "macos")]
mod native {
    use muda::{MenuEvent, MenuId};

    use super::MenuAction;

    /// Native OS menu bar.
    pub struct NativeMenu {
        #[allow(dead_code)]
        menu: muda::Menu,
        about_id: MenuId,
        save_id: MenuId,
        undo_id: MenuId,
        redo_id: MenuId,
    }

    impl NativeMenu {
        /// Create and install the native menu bar.
        pub fn new() -> Self {
            use muda::{Menu, MenuItem, PredefinedMenuItem, Submenu, accelerator::Accelerator};

            let menu = Menu::new();

            // RedLilium submenu
            let about_item = MenuItem::new("About RedLilium Editor", true, None::<Accelerator>);
            let about_id = about_item.id().clone();

            let app_submenu = Submenu::with_items(
                "RedLilium",
                true,
                &[
                    &about_item,
                    &PredefinedMenuItem::separator(),
                    &PredefinedMenuItem::quit(None),
                ],
            )
            .expect("failed to create app submenu");

            // File submenu
            let save_item = MenuItem::new(
                "Save",
                true,
                Some(Accelerator::new(
                    Some(muda::accelerator::Modifiers::META),
                    muda::accelerator::Code::KeyS,
                )),
            );
            let save_id = save_item.id().clone();

            let file_submenu = Submenu::with_items("File", true, &[&save_item])
                .expect("failed to create file submenu");

            // Edit submenu
            let undo_item = MenuItem::new(
                "Undo",
                true,
                Some(Accelerator::new(
                    Some(muda::accelerator::Modifiers::META),
                    muda::accelerator::Code::KeyZ,
                )),
            );
            let redo_item = MenuItem::new(
                "Redo",
                true,
                Some(Accelerator::new(
                    Some(muda::accelerator::Modifiers::META | muda::accelerator::Modifiers::SHIFT),
                    muda::accelerator::Code::KeyZ,
                )),
            );
            let undo_id = undo_item.id().clone();
            let redo_id = redo_item.id().clone();

            let edit_submenu = Submenu::with_items("Edit", true, &[&undo_item, &redo_item])
                .expect("failed to create edit submenu");

            menu.append(&app_submenu)
                .expect("failed to append app submenu");
            menu.append(&file_submenu)
                .expect("failed to append file submenu");
            menu.append(&edit_submenu)
                .expect("failed to append edit submenu");

            menu.init_for_nsapp();

            NativeMenu {
                menu,
                about_id,
                save_id,
                undo_id,
                redo_id,
            }
        }

        /// Poll for menu events and return an action if one was triggered.
        pub fn poll_event(&self) -> Option<MenuAction> {
            if let Ok(event) = MenuEvent::receiver().try_recv() {
                if event.id == self.about_id {
                    return Some(MenuAction::About);
                } else if event.id == self.save_id {
                    return Some(MenuAction::Save);
                } else if event.id == self.undo_id {
                    return Some(MenuAction::Undo);
                } else if event.id == self.redo_id {
                    return Some(MenuAction::Redo);
                }
            }
            None
        }
    }
}

#[cfg(target_os = "macos")]
pub use native::NativeMenu;

// --- egui fallback for non-macOS ---

#[cfg(not(target_os = "macos"))]
pub struct MenuBarResult {
    pub action: Option<MenuAction>,
    pub play_state: crate::toolbar::PlayState,
}

#[cfg(not(target_os = "macos"))]
pub fn draw_menu_bar(
    ctx: &egui::Context,
    window: &Arc<Window>,
    custom_titlebar: bool,
    play_state: crate::toolbar::PlayState,
) -> MenuBarResult {
    let mut action = None;
    let mut new_play_state = play_state;
    let window_for_drag = window.clone();

    egui::TopBottomPanel::top("menu_bar").show(ctx, |ui| {
        ui.horizontal(|ui| {
            // Left: menu buttons
            egui::MenuBar::new().ui(ui, |ui| {
                ui.menu_button("RedLilium", |ui| {
                    if ui.button("About RedLilium Editor").clicked() {
                        ui.close();
                    }
                });
                ui.menu_button("File", |ui| {
                    if ui
                        .add(egui::Button::new("Save").shortcut_text("Ctrl+S"))
                        .clicked()
                    {
                        action = Some(MenuAction::Save);
                        ui.close();
                    }
                });
                ui.menu_button("Edit", |ui| {
                    if ui
                        .add(egui::Button::new("Undo").shortcut_text("Ctrl+Z"))
                        .clicked()
                    {
                        action = Some(MenuAction::Undo);
                        ui.close();
                    }
                    if ui
                        .add(egui::Button::new("Redo").shortcut_text("Ctrl+Shift+Z"))
                        .clicked()
                    {
                        action = Some(MenuAction::Redo);
                        ui.close();
                    }
                });
            });

            // Center play controls between the menu (left) and window
            // controls (right).  We use max_rect which is the full
            // horizontal region allocated for this row.
            let full_rect = ui.max_rect();
            let right_width = if custom_titlebar { 3.0 * 36.0 } else { 0.0 };
            let usable_center = (full_rect.left() + full_rect.right() - right_width) / 2.0;

            // Lay out play controls at the computed center using an
            // absolute-position child UI so they don't steal space from
            // the window controls.
            let play_rect = egui::Rect::from_min_size(
                egui::pos2(usable_center - 40.0, full_rect.top()),
                egui::vec2(120.0, full_rect.height()),
            );
            let mut play_ui = ui.new_child(
                egui::UiBuilder::new()
                    .max_rect(play_rect)
                    .layout(egui::Layout::left_to_right(egui::Align::Center)),
            );
            new_play_state = crate::toolbar::draw_play_controls(&mut play_ui, play_state);

            // Right: window control buttons (custom titlebar only)
            if custom_titlebar {
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    draw_window_controls(ui, window, &mut action);
                });
            }
        });

        // Titlebar drag: only when no widget is using the pointer
        if custom_titlebar {
            let titlebar_rect = ui.min_rect();
            let pointer_in_bar = ui
                .input(|i| i.pointer.interact_pos())
                .is_some_and(|pos| titlebar_rect.contains(pos));

            if pointer_in_bar && !ui.ctx().is_using_pointer() {
                if ui.input(|i| i.pointer.button_pressed(egui::PointerButton::Primary)) {
                    let _ = window_for_drag.drag_window();
                }
                if ui.input(|i| {
                    i.pointer
                        .button_double_clicked(egui::PointerButton::Primary)
                }) {
                    window_for_drag.set_maximized(!window_for_drag.is_maximized());
                }
            }
        }
    });

    MenuBarResult {
        action,
        play_state: new_play_state,
    }
}

/// Draw the minimize / maximize / close buttons (Windows/Linux).
#[cfg(not(target_os = "macos"))]
fn draw_window_controls(ui: &mut egui::Ui, window: &Arc<Window>, action: &mut Option<MenuAction>) {
    let btn_size = egui::vec2(36.0, 20.0);

    // Close button — red background on hover
    let close_btn = ui.add_sized(
        btn_size,
        egui::Button::new(
            egui::RichText::new("\u{00D7}")
                .size(16.0)
                .color(crate::theme::TEXT_PRIMARY),
        )
        .frame(false),
    );
    if close_btn.hovered() {
        ui.painter().rect_filled(
            close_btn.rect,
            egui::CornerRadius::ZERO,
            egui::Color32::from_rgb(232, 17, 35),
        );
        // Re-draw the label on top of the red background
        ui.painter().text(
            close_btn.rect.center(),
            egui::Align2::CENTER_CENTER,
            "\u{00D7}",
            egui::FontId::proportional(16.0),
            egui::Color32::WHITE,
        );
    }
    if close_btn.clicked() {
        *action = Some(MenuAction::CloseWindow);
    }

    // Maximize / restore button
    let max_label = if window.is_maximized() {
        "\u{2750}"
    } else {
        "\u{25A1}"
    };
    let max_btn = ui.add_sized(
        btn_size,
        egui::Button::new(
            egui::RichText::new(max_label)
                .size(13.0)
                .color(crate::theme::TEXT_PRIMARY),
        )
        .frame(false),
    );
    if max_btn.hovered() {
        ui.painter().rect_filled(
            max_btn.rect,
            egui::CornerRadius::ZERO,
            crate::theme::SURFACE3,
        );
    }
    if max_btn.clicked() {
        window.set_maximized(!window.is_maximized());
    }

    // Minimize button
    let min_btn = ui.add_sized(
        btn_size,
        egui::Button::new(
            egui::RichText::new("\u{2013}")
                .size(13.0)
                .color(crate::theme::TEXT_PRIMARY),
        )
        .frame(false),
    );
    if min_btn.hovered() {
        ui.painter().rect_filled(
            min_btn.rect,
            egui::CornerRadius::ZERO,
            crate::theme::SURFACE3,
        );
    }
    if min_btn.clicked() {
        window.set_minimized(true);
    }
}
