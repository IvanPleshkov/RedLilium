use redlilium_debug_drawer::DebugDrawer;

use crate::std::components::grid::GridConfig;

/// System that draws a debug grid using the [`DebugDrawer`].
///
/// Reads the [`GridConfig`] and [`DebugDrawer`] resources. If either
/// is missing, the system does nothing.
///
/// # Access
///
/// - Reads: `GridConfig` (resource), `DebugDrawer` (resource)
pub struct DrawGrid;

impl crate::System for DrawGrid {
    type Result = ();

    fn run<'a>(
        &'a self,
        ctx: &'a crate::SystemContext<'a>,
    ) -> Result<(), crate::system::SystemError> {
        ctx.lock::<(crate::Res<GridConfig>, crate::Res<DebugDrawer>)>()
            .execute(|(config, drawer)| {
                let mut draw_ctx = drawer.context();
                draw_ctx.draw_grid(
                    config.center,
                    config.cell_size,
                    config.half_count,
                    config.color,
                );
            });
        Ok(())
    }
}
