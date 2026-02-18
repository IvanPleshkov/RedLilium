/// Actions that can be triggered from the menu.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MenuAction {
    About,
    Undo,
    Redo,
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
            menu.append(&edit_submenu)
                .expect("failed to append edit submenu");

            menu.init_for_nsapp();

            NativeMenu {
                menu,
                about_id,
                undo_id,
                redo_id,
            }
        }

        /// Poll for menu events and return an action if one was triggered.
        pub fn poll_event(&self) -> Option<MenuAction> {
            if let Ok(event) = MenuEvent::receiver().try_recv() {
                if event.id == self.about_id {
                    return Some(MenuAction::About);
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
pub fn draw_menu_bar(ctx: &egui::Context) {
    egui::TopBottomPanel::top("menu_bar").show(ctx, |ui| {
        egui::MenuBar::new().ui(ui, |ui| {
            ui.menu_button("RedLilium", |ui| {
                if ui.button("About RedLilium Editor").clicked() {
                    ui.close();
                }
            });
            ui.menu_button("Edit", |ui| {
                if ui
                    .add(egui::Button::new("Undo").shortcut_text("Ctrl+Z"))
                    .clicked()
                {
                    ui.close();
                }
                if ui
                    .add(egui::Button::new("Redo").shortcut_text("Ctrl+Shift+Z"))
                    .clicked()
                {
                    ui.close();
                }
            });
        });
    });
}
