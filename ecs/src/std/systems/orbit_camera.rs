use crate::std::components::{OrbitCamera, WindowInput};

/// System that updates orbit camera parameters from window input.
///
/// Reads the [`WindowInput`] resource. When the left mouse button is down
/// and `ui_wants_input` is false, applies cursor delta as azimuth/elevation
/// rotation. Scroll delta is applied as zoom.
///
/// After updating orbital parameters, writes the computed [`Transform`](crate::Transform)
/// to the entity.
///
/// # Access
///
/// - Resource: `Res<WindowInput>`
/// - Writes: `OrbitCamera`, `Transform`
pub struct UpdateOrbitCamera;

impl crate::System for UpdateOrbitCamera {
    type Result = ();

    fn run<'a>(
        &'a self,
        ctx: &'a crate::SystemContext<'a>,
    ) -> Result<(), crate::system::SystemError> {
        ctx.lock::<(
            crate::Res<WindowInput>,
            crate::Write<OrbitCamera>,
            crate::Write<crate::Transform>,
        )>()
        .execute(|(input, mut orbit_cameras, mut transforms)| {
            if input.ui_wants_input {
                return;
            }

            for (idx, orbit) in orbit_cameras.iter_mut() {
                // Rotate on left-mouse drag
                if input.mouse_left {
                    let dx = input.cursor_delta[0] * orbit.rotate_sensitivity;
                    let dy = input.cursor_delta[1] * orbit.rotate_sensitivity;
                    orbit.rotate(-dx, dy);
                }

                // Zoom on scroll
                if input.scroll_delta[1].abs() > f32::EPSILON {
                    orbit.zoom(input.scroll_delta[1] * orbit.zoom_sensitivity);
                }

                // Update transform from orbital parameters
                if let Some(t) = transforms.get_mut(idx) {
                    *t = orbit.to_transform();
                }
            }
        });
        Ok(())
    }
}
