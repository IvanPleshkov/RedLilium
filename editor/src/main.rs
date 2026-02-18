use redlilium_app::{App, AppArgs, AppContext, AppHandler, DefaultAppArgs, DrawContext};
use redlilium_ecs::{EcsRunner, Schedules, World};
use redlilium_graphics::FrameSchedule;

struct Editor {
    world: World,
    schedules: Schedules,
    runner: EcsRunner,
}

impl Editor {
    fn new() -> Self {
        Self {
            world: World::new(),
            schedules: Schedules::new(),
            runner: EcsRunner::single_thread(),
        }
    }
}

impl AppHandler for Editor {
    fn on_init(&mut self, _ctx: &mut AppContext) {
        log::info!("Editor initialized");
        self.schedules.run_startup(&mut self.world, &self.runner);
    }

    fn on_update(&mut self, ctx: &mut AppContext) -> bool {
        self.schedules
            .run_frame(&mut self.world, &self.runner, ctx.delta_time() as f64);
        true
    }

    fn on_draw(&mut self, ctx: DrawContext) -> FrameSchedule {
        ctx.finish(&[])
    }
}

fn main() {
    let args = DefaultAppArgs::parse().with_title_str("RedLilium Editor");
    App::run(Editor::new(), args);
}
