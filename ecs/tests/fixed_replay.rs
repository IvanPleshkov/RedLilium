//! A fixed replay drives simulation without advancing the presentation timeline.
use redlilium_ecs::*;

const STEP: f64 = 1.0 / 128.0;
#[derive(Default)]
struct Samples {
    frames: usize,
    ticks: Vec<FixedTime>,
    flushed: usize,
}
struct Fixed;
impl System for Fixed {
    type Result = ();
    fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) -> Result<(), SystemError> {
        ctx.lock::<(Res<FixedTime>, Res<Time>, ResMut<Samples>)>()
            .execute(|(fixed, time, mut samples)| {
                assert_eq!(fixed.delta(), STEP);
                assert_eq!(time.delta(), STEP);
                assert_eq!(time.fixed_delta(), STEP);
                samples.ticks.push(*fixed);
            });
        ctx.commands(|world| world.resource_mut::<Samples>().flushed += 1);
        Ok(())
    }
}
struct Frame;
impl System for Frame {
    type Result = ();
    fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) -> Result<(), SystemError> {
        ctx.lock::<(ResMut<Samples>,)>()
            .execute(|(mut s,)| s.frames += 1);
        Ok(())
    }
}
fn setup() -> (World, Schedules, EcsRunner) {
    let mut w = World::new();
    w.insert_resource(Samples::default());
    w.insert_resource(GameTime::default());
    w.insert_resource(RealTime::default());
    w.add_event::<u32>();
    let mut s = Schedules::new();
    s.set_fixed_timestep(STEP);
    s.get_mut::<FixedUpdate>().add(Fixed);
    s.get_mut::<Update>().add(Frame);
    (w, s, EcsRunner::single_thread())
}
fn frame_clocks(w: &World) -> [f64; 8] {
    let t = w.resource::<Time>();
    let g = w.resource::<GameTime>();
    let r = w.resource::<RealTime>();
    [
        t.elapsed(),
        t.delta(),
        t.frame_delta(),
        t.fixed_alpha(),
        g.elapsed(),
        g.delta(),
        r.elapsed(),
        r.delta(),
    ]
}
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn replay_preserves_frame_time_events_and_residual_accumulator() {
    let (mut w, mut s, r) = setup();
    s.run_frame(&mut w, &r, STEP * 1.5);
    let checkpoint = s.capture_fixed_clock(&w).unwrap();
    s.run_frame(&mut w, &r, STEP * 2.0);
    let expected = w.resource::<Samples>().ticks[1..].to_vec();
    let presentation = frame_clocks(&w);
    w.resource_mut::<Events<u32>>().send(42);
    s.restore_fixed_clock(&mut w, &checkpoint).unwrap();
    assert_eq!(w.resource::<FixedTime>().tick(), 1);
    assert_eq!(frame_clocks(&w), presentation);
    // Explicit ticks also run while presentation/gameplay time is paused.
    w.resource_mut::<GameTime>().set_scale(0.0);
    w.resource_mut::<Samples>().ticks.clear();
    s.run_fixed_tick(&mut w, &r).unwrap();
    s.run_fixed_tick(&mut w, &r).unwrap();
    assert_eq!(w.resource::<Samples>().ticks, expected);
    assert_eq!(w.resource::<Samples>().flushed, 5);
    assert_eq!(w.resource::<Samples>().frames, 2);
    assert_eq!(frame_clocks(&w), presentation);
    assert_eq!(w.resource::<Events<u32>>().len(), 1);
    w.resource_mut::<GameTime>().set_scale(1.0);
    s.run_frame(&mut w, &r, STEP * 0.5);
    assert_eq!(w.resource::<FixedTime>().tick(), 4);
    assert_eq!(w.resource::<Time>().fixed_alpha(), 0.0);
}
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn clock_checkpoint_rejects_wrong_owner_world_and_timestep_before_mutation() {
    let (mut w, mut s, r) = setup();
    let before_first = s.capture_fixed_clock(&w).unwrap();
    s.run_fixed_tick(&mut w, &r).unwrap();
    let before = *w.resource::<FixedTime>();
    let (_, mut other, _) = setup();
    assert!(other.restore_fixed_clock(&mut w, &before_first).is_err());
    s.set_fixed_timestep(STEP * 2.0);
    assert!(s.restore_fixed_clock(&mut w, &before_first).is_err());
    assert_eq!(*w.resource::<FixedTime>(), before);
    let mut other_world = World::new();
    assert!(s.run_fixed_tick(&mut other_world, &r).is_err());
    assert!(
        s.restore_fixed_clock(&mut other_world, &before_first)
            .is_err()
    );
    assert!(!other_world.has_resource::<Time>());
    s.set_fixed_timestep(STEP);
    s.restore_fixed_clock(&mut w, &before_first).unwrap();
    assert_eq!(*w.resource::<FixedTime>(), FixedTime::default());
}
struct Fail;
impl System for Fail {
    type Result = ();
    fn run<'a>(&'a self, _: &'a SystemContext<'a>) -> Result<(), SystemError> {
        Err(SystemError::InvalidConfiguration {
            message: "expected failure".into(),
        })
    }
}
#[cfg_attr(not(target_arch = "wasm32"), test)]
fn attempted_tick_reports_errors_and_restores_effective_delta() {
    let (mut w, mut s, r) = setup();
    s.run_frame(&mut w, &r, STEP * 0.5);
    assert_eq!(w.resource::<FixedTime>().tick(), 0);
    let frame = frame_clocks(&w);
    s.get_mut::<FixedUpdate>().add(Fail);
    assert_eq!(s.run_fixed_tick(&mut w, &r).unwrap_err().len(), 1);
    assert_eq!(w.resource::<FixedTime>().tick(), 1);
    assert_eq!(w.resource::<Samples>().flushed, 1);
    assert_eq!(frame_clocks(&w), frame);
}

#[cfg(target_arch = "wasm32")]
pub fn run_wasm_contracts() {
    replay_preserves_frame_time_events_and_residual_accumulator();
    clock_checkpoint_rejects_wrong_owner_world_and_timestep_before_mutation();
    attempted_tick_reports_errors_and_restores_effective_delta();
}
