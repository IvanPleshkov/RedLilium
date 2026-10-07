//! Public scheduler contract for gameplay time scaling, independent of physics.
use redlilium_ecs::*;

const STEP: f64 = 1.0 / 128.0;

#[derive(Clone, Copy, Debug)]
struct Sample {
    delta: f64,
    fixed_delta: f64,
    frame_delta: f64,
    game_delta: f64,
}

#[derive(Default)]
struct Samples([Vec<Sample>; 4]);

struct Record<const PHASE: usize>;
impl<const PHASE: usize> System for Record<PHASE> {
    type Result = ();
    fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) -> Result<(), SystemError> {
        ctx.lock::<(Res<Time>, Res<GameTime>, ResMut<Samples>)>()
            .execute(|(time, game, mut samples)| {
                samples.0[PHASE].push(Sample {
                    delta: time.delta(),
                    fixed_delta: time.fixed_delta(),
                    frame_delta: time.frame_delta(),
                    game_delta: game.delta(),
                });
            });
        Ok(())
    }
}

struct SetScale(f64);
impl System for SetScale {
    type Result = ();
    fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) -> Result<(), SystemError> {
        ctx.lock::<(ResMut<GameTime>,)>()
            .execute(|(mut time,)| time.set_scale(self.0));
        Ok(())
    }
}

fn setup(scale: f64) -> (World, Schedules, EcsRunner) {
    let mut world = World::new();
    let mut game = GameTime::default();
    game.set_scale(scale);
    world.insert_resource(game);
    world.insert_resource(RealTime::default());
    world.insert_resource(Samples::default());
    let mut schedules = Schedules::new();
    schedules.set_fixed_timestep(STEP);
    schedules.get_mut::<PreUpdate>().add(Record::<0>);
    schedules.get_mut::<FixedUpdate>().add(Record::<1>);
    schedules.get_mut::<Update>().add(Record::<2>);
    schedules.get_mut::<PostUpdate>().add(Record::<3>);
    (world, schedules, EcsRunner::single_thread())
}

#[test]
fn scale_controls_fixed_frequency_without_changing_step_or_frame_schedules() {
    for fps in [32, 64, 128, 256] {
        for scale in [0.0, 0.25, 1.0, 2.0] {
            let (mut world, mut schedules, runner) = setup(scale);
            let dt = 1.0 / fps as f64;
            for _ in 0..fps {
                schedules.run_frame(&mut world, &runner, dt);
            }
            let samples = world.resource::<Samples>();
            assert_eq!(samples.0[1].len(), (128.0 * scale) as usize);
            for sample in &samples.0[1] {
                assert_eq!(sample.delta, STEP);
                assert_eq!(sample.fixed_delta, STEP);
                assert_eq!(sample.frame_delta, dt);
                assert_eq!(sample.game_delta, dt * scale);
            }
            for phase in [0, 2, 3] {
                assert_eq!(samples.0[phase].len(), fps);
                assert!(samples.0[phase].iter().all(|s| s.delta == dt));
            }
            assert_eq!(world.resource::<GameTime>().elapsed(), scale);
            assert_eq!(world.resource::<RealTime>().elapsed(), 1.0);
            assert_eq!(world.resource::<RealTime>().delta(), dt);
            assert_eq!(world.resource::<Time>().elapsed(), 1.0);
            assert_eq!(world.resource::<Time>().fixed_alpha(), 0.0);
        }
    }
}

#[test]
fn pause_preserves_fraction_and_resume_has_no_real_time_debt() {
    let (mut world, mut schedules, runner) = setup(0.25);
    schedules.run_frame(&mut world, &runner, STEP * 2.0);
    assert_eq!(world.resource::<Time>().fixed_alpha(), 0.5);
    assert!(world.resource::<Samples>().0[1].is_empty());

    world.resource_mut::<GameTime>().set_scale(0.0);
    for _ in 0..64 {
        schedules.run_frame(&mut world, &runner, 1.0 / 64.0);
        assert_eq!(world.resource::<Time>().fixed_alpha(), 0.5);
    }
    assert!(world.resource::<Samples>().0[1].is_empty());
    assert_eq!(world.resource::<GameTime>().elapsed(), STEP * 0.5);

    world.resource_mut::<GameTime>().set_scale(1.0);
    schedules.run_frame(&mut world, &runner, STEP);
    assert_eq!(world.resource::<Samples>().0[1].len(), 1);
    assert_eq!(world.resource::<Time>().fixed_alpha(), 0.5);
    assert_eq!(world.resource::<GameTime>().elapsed(), STEP * 1.5);
}

#[test]
fn fixed_scale_change_applies_next_frame_even_with_multiple_steps() {
    let (mut world, mut schedules, runner) = setup(1.0);
    // Freeze on the first simulated hit. The second step still belongs to
    // this frame's already sampled game delta.
    schedules.get_mut::<FixedUpdate>().add(SetScale(0.0));
    schedules.run_frame(&mut world, &runner, STEP * 2.0);
    assert_eq!(world.resource::<Samples>().0[1].len(), 2);
    assert_eq!(world.resource::<GameTime>().delta(), STEP * 2.0);
    schedules.run_frame(&mut world, &runner, STEP * 2.0);
    assert_eq!(world.resource::<Samples>().0[1].len(), 2);
    assert_eq!(world.resource::<GameTime>().delta(), 0.0);
}

#[test]
fn pre_update_scale_change_applies_this_frame() {
    let (mut world, mut schedules, runner) = setup(0.0);
    schedules.get_mut::<PreUpdate>().add(SetScale(0.25));
    schedules.run_frame(&mut world, &runner, STEP * 4.0);
    assert_eq!(world.resource::<Samples>().0[1].len(), 1);
    assert_eq!(world.resource::<GameTime>().delta(), STEP);
}

struct ResumeAfterRealSecond;
impl System for ResumeAfterRealSecond {
    type Result = ();
    fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) -> Result<(), SystemError> {
        ctx.lock::<(Res<RealTime>, ResMut<GameTime>)>()
            .execute(|(real, mut game)| {
                if real.elapsed() >= 1.0 {
                    game.set_scale(1.0);
                }
            });
        Ok(())
    }
}

#[test]
fn real_time_update_can_end_slow_motion_or_pause_without_catch_up() {
    for scale in [0.0, 0.25] {
        let (mut world, mut schedules, runner) = setup(scale);
        schedules.get_mut::<Update>().add(ResumeAfterRealSecond);
        for _ in 0..64 {
            schedules.run_frame(&mut world, &runner, 1.0 / 64.0);
        }
        let slow_steps = (128.0 * scale) as usize;
        assert_eq!(world.resource::<Samples>().0[1].len(), slow_steps);
        assert_eq!(world.resource::<GameTime>().elapsed(), scale);
        assert_eq!(world.resource::<GameTime>().scale(), 1.0);
        schedules.run_frame(&mut world, &runner, 1.0 / 64.0);
        assert_eq!(world.resource::<Samples>().0[1].len(), slow_steps + 2);
        assert_eq!(world.resource::<GameTime>().delta(), 1.0 / 64.0);
    }
}

#[test]
fn catch_up_limit_is_applied_to_game_time_and_drops_excess_debt() {
    let (mut world, mut schedules, runner) = setup(2.0);
    schedules.set_max_fixed_steps(4);
    schedules.run_frame(&mut world, &runner, STEP * 4.0);
    assert_eq!(world.resource::<Samples>().0[1].len(), 4);
    assert_eq!(world.resource::<GameTime>().delta(), STEP * 8.0);
    assert_eq!(world.resource::<Time>().fixed_alpha(), 0.0);
    world.resource_mut::<GameTime>().set_scale(1.0);
    schedules.run_frame(&mut world, &runner, STEP);
    assert_eq!(world.resource::<Samples>().0[1].len(), 5);
}

#[test]
fn interpolation_clock_is_scaled_even_without_fixed_systems() {
    let mut world = World::new();
    let mut game = GameTime::default();
    game.set_scale(0.25);
    world.insert_resource(game);
    let mut schedules = Schedules::new();
    schedules.set_fixed_timestep(STEP);
    schedules.run_frame(&mut world, &EcsRunner::single_thread(), STEP * 6.0);
    assert_eq!(world.resource::<Time>().fixed_alpha(), 0.5);
}
