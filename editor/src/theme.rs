//! Editor color theme — dark palette with soft burgundy accent.

use egui::Color32;

// --- Accent (Burgundy Soft) ---

pub const ACCENT: Color32 = Color32::from_rgb(166, 74, 92);
pub const ACCENT_HOVER: Color32 = Color32::from_rgb(186, 88, 108);
pub const ACCENT_PRESSED: Color32 = Color32::from_rgb(138, 60, 76);

// --- Base (dark backgrounds) ---

pub const BG: Color32 = Color32::from_rgb(14, 16, 20);
pub const SURFACE1: Color32 = Color32::from_rgb(20, 23, 29);
pub const SURFACE2: Color32 = Color32::from_rgb(26, 30, 38);
pub const SURFACE3: Color32 = Color32::from_rgb(33, 38, 48);
pub const BORDER: Color32 = Color32::from_rgb(48, 55, 68);

// --- Text ---

pub const TEXT_PRIMARY: Color32 = Color32::from_rgb(232, 236, 242);
pub const TEXT_SECONDARY: Color32 = Color32::from_rgb(170, 178, 191);
pub const TEXT_MUTED: Color32 = Color32::from_rgb(120, 129, 145);

// --- Interactive / states ---

pub const SELECTION: Color32 = Color32::from_rgb(60, 30, 40);
pub const SUCCESS: Color32 = Color32::from_rgb(74, 166, 126);
pub const WARNING: Color32 = Color32::from_rgb(230, 185, 92);
pub const ERROR: Color32 = Color32::from_rgb(220, 88, 88);
pub const INFO: Color32 = Color32::from_rgb(92, 150, 230);

/// Apply the full theme to an egui context.
pub fn apply(ctx: &egui::Context) {
    let mut style = (*ctx.style()).clone();
    let v = &mut style.visuals;

    // Start from dark defaults
    *v = egui::Visuals::dark();

    // --- Window / panel backgrounds ---
    v.window_fill = SURFACE1;
    v.panel_fill = SURFACE1;
    v.extreme_bg_color = BG;
    v.faint_bg_color = SURFACE2;
    v.code_bg_color = SURFACE2;

    // Corner radius
    v.window_corner_radius = egui::CornerRadius::same(4);

    // --- Stroke / border ---
    v.window_stroke = egui::Stroke::new(1.0, BORDER);
    v.widgets.noninteractive.bg_stroke = egui::Stroke::new(1.0, BORDER);
    v.widgets.noninteractive.fg_stroke = egui::Stroke::new(1.0, TEXT_SECONDARY);

    // --- Widget backgrounds ---

    // Inactive (buttons, combo-boxes at rest)
    v.widgets.inactive.bg_fill = SURFACE2;
    v.widgets.inactive.bg_stroke = egui::Stroke::new(1.0, BORDER);
    v.widgets.inactive.fg_stroke = egui::Stroke::new(1.0, TEXT_PRIMARY);
    v.widgets.inactive.weak_bg_fill = SURFACE2;

    // Hovered
    v.widgets.hovered.bg_fill = SURFACE3;
    v.widgets.hovered.bg_stroke = egui::Stroke::new(1.0, ACCENT_HOVER);
    v.widgets.hovered.fg_stroke = egui::Stroke::new(1.0, TEXT_PRIMARY);
    v.widgets.hovered.weak_bg_fill = SURFACE3;

    // Active (pressed)
    v.widgets.active.bg_fill = ACCENT_PRESSED;
    v.widgets.active.bg_stroke = egui::Stroke::new(1.0, ACCENT);
    v.widgets.active.fg_stroke = egui::Stroke::new(1.0, TEXT_PRIMARY);
    v.widgets.active.weak_bg_fill = ACCENT_PRESSED;

    // Open (menus / combo-box popups)
    v.widgets.open.bg_fill = SURFACE3;
    v.widgets.open.bg_stroke = egui::Stroke::new(1.0, ACCENT);
    v.widgets.open.fg_stroke = egui::Stroke::new(1.0, TEXT_PRIMARY);
    v.widgets.open.weak_bg_fill = SURFACE3;

    // Non-interactive (labels, separators)
    v.widgets.noninteractive.bg_fill = SURFACE1;
    v.widgets.noninteractive.weak_bg_fill = SURFACE1;

    // --- Selection ---
    v.selection.bg_fill = SELECTION;
    v.selection.stroke = egui::Stroke::new(1.0, ACCENT);

    // --- Hyperlinks ---
    v.hyperlink_color = ACCENT_HOVER;

    // --- Text colors ---
    v.override_text_color = Some(TEXT_PRIMARY);
    v.warn_fg_color = WARNING;
    v.error_fg_color = ERROR;

    // --- Popup shadow ---
    v.popup_shadow = egui::Shadow {
        offset: [0, 2],
        blur: 8,
        spread: 0,
        color: Color32::from_black_alpha(80),
    };

    ctx.set_style(style);
}
