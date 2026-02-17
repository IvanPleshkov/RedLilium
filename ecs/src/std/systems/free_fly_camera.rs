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
            crate::Write<FreeFlyCamera>,
            crate::Write<crate::Transform>,
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

                    if input.key_ctrl {
                        // Fly mode: Ctrl held — rotate view, target follows
                        cam.free_rotate(-dx, dy);
                    } else {
                        // Orbit mode: rotate around fixed target
                        cam.orbit_rotate(-dx, dy);
                    }
                }

                // Shift: ramp up speed multiplier; release resets it
                if input.key_shift {
                    cam.speed_multiplier = (cam.speed_multiplier * cam.speed_boost_acceleration)
                        .min(cam.max_speed_multiplier);
                } else {
                    cam.speed_multiplier = 1.0;
                }

                // WASD / QE movement
                let mut move_forward = 0.0f32;
                let mut move_right = 0.0f32;
                let mut move_up = 0.0f32;

                if input.key_w {
                    move_forward += 1.0;
                }
                if input.key_s {
                    move_forward -= 1.0;
                }
                if input.key_d {
                    move_right += 1.0;
                }
                if input.key_a {
                    move_right -= 1.0;
                }
                if input.key_e {
                    move_up += 1.0;
                }
                if input.key_q {
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
