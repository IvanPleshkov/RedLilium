use redlilium_core::input::KeyCode;

use crate::std::components::{FreeFlyCamera, WindowInput};

/// System that updates the free-fly camera from window input.
///
/// **Orbit mode** (default): mouse drag orbits the camera around
/// the fixed target point.
///
/// **Fly mode** (Ctrl held): mouse drag rotates the view direction.
/// WASD moves forward/back/left/right, Q/E moves down/up. The target
/// point moves with the camera.
///
/// Scroll wheel zooms (adjusts distance to target) in both modes.
///
/// # Access
///
/// - Resource: `Res<WindowInput>`
/// - Writes: `FreeFlyCamera`, `Transform`
pub struct UpdateFreeFlyCamera;

impl crate::System for UpdateFreeFlyCamera {
    type Result = ();

    fn run<'a>(
        &'a self,
        ctx: &'a crate::SystemContext<'a>,
    ) -> Result<(), crate::system::SystemError> {
        ctx.lock::<(
            crate::Res<WindowInput>,
            crate::WriteAll<FreeFlyCamera>,
            crate::WriteAll<crate::Transform>,
        )>()
        .execute(|(input, mut cameras, mut transforms)| {
            if input.ui_wants_input {
                return;
            }

            for (idx, cam) in cameras.iter_mut() {
                let dragging = input.mouse_left || input.mouse_right;

                if dragging {
                    let dx = input.cursor_delta[0] * cam.rotate_sensitivity;
                    let dy = input.cursor_delta[1] * cam.rotate_sensitivity;

                    let ctrl = input.is_key_pressed(KeyCode::ControlLeft)
                        || input.is_key_pressed(KeyCode::ControlRight);
                    if ctrl {
                        // Fly mode: Ctrl held — rotate view, target follows
                        cam.free_rotate(-dx, dy);
                    } else {
                        // Orbit mode: rotate around fixed target
                        cam.orbit_rotate(-dx, dy);
                    }
                }

                // Shift: ramp up speed multiplier; release resets it
                let shift = input.is_key_pressed(KeyCode::ShiftLeft)
                    || input.is_key_pressed(KeyCode::ShiftRight);
                if shift {
                    cam.speed_multiplier = (cam.speed_multiplier * cam.speed_boost_acceleration)
                        .min(cam.max_speed_multiplier);
                } else {
                    cam.speed_multiplier = 1.0;
                }

                // WASD / QE movement
                let mut move_forward = 0.0f32;
                let mut move_right = 0.0f32;
                let mut move_up = 0.0f32;

                if input.is_key_pressed(KeyCode::W) {
                    move_forward += 1.0;
                }
                if input.is_key_pressed(KeyCode::S) {
                    move_forward -= 1.0;
                }
                if input.is_key_pressed(KeyCode::D) {
                    move_right += 1.0;
                }
                if input.is_key_pressed(KeyCode::A) {
                    move_right -= 1.0;
                }
                if input.is_key_pressed(KeyCode::E) {
                    move_up += 1.0;
                }
                if input.is_key_pressed(KeyCode::Q) {
                    move_up -= 1.0;
                }

                if move_forward.abs() > f32::EPSILON
                    || move_right.abs() > f32::EPSILON
                    || move_up.abs() > f32::EPSILON
                {
                    cam.fly_move(move_forward, move_right, move_up);
                }

                // Zoom on scroll
                if input.scroll_delta[1].abs() > f32::EPSILON {
                    cam.zoom(input.scroll_delta[1] * cam.zoom_sensitivity);
                }

                // Update transform from camera parameters
                if let Some(t) = transforms.get_mut(idx) {
                    *t = cam.to_transform();
                }
            }
        });
        Ok(())
    }
}
