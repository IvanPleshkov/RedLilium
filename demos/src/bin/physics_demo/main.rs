mod demo;
mod renderer;
mod scenes_2d;
mod scenes_3d;
mod ui;

use redlilium_app::AppArgs;
use redlilium_core::profiling::create_profiled_allocator;

create_profiled_allocator!(GLOBAL_ALLOCATOR, 32);

fn main() {
    let mut args = redlilium_app::DefaultAppArgs::parse();
    args = args.with_title_str("RedLilium Physics Demo");
    redlilium_app::App::run(demo::PhysicsDemoApp::new(), args);
}
