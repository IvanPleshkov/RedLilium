//! Public input contract, with no window, DOM, controller, or GPU.
use redlilium_core::input::KeyCode;
use redlilium_ecs::*;

fn connect(input: &mut WindowInput, id: GamepadId) {
    input.on_gamepad(GamepadEvent::Connected {
        id,
        name: "Test pad".into(),
        mapping: GamepadMapping::Standard,
    });
}
fn button(input: &mut WindowInput, id: GamepadId, pressed: bool) {
    input.on_gamepad(GamepadEvent::Button {
        id,
        button: GamepadButton::South as u32,
        value: f32::from(pressed),
        pressed,
    });
}
fn touch(
    input: &mut WindowInput,
    id: u64,
    phase: TouchPhase,
    position: [f64; 2],
    owner: InputOwner,
) {
    input.on_touch(
        TouchInput {
            id: TouchId(id),
            phase,
            position,
        },
        owner,
    );
}

struct Consumer {
    reader: InputEventReader,
    events: Vec<InputEvent>,
    steps: usize,
}
struct Consume;
impl System for Consume {
    type Result = ();
    fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) -> Result<(), SystemError> {
        ctx.lock::<(ResMut<Consumer>,)>()
            .execute(|(mut consumer,)| {
                let events = consumer.reader.drain();
                consumer.events.extend(events);
                consumer.steps += 1;
            });
        Ok(())
    }
}

#[test]
fn transitions_survive_zero_or_many_fixed_steps_and_independent_frame_readers() {
    for fps in [32, 64, 256] {
        for scale in [0.0, 0.25, 1.0] {
            let mut input = WindowInput::default();
            let mut frame_reader = input.subscribe();
            let reader = input.subscribe();
            let mut world = World::new();
            let input = world.insert_resource(input);
            world.insert_resource(Consumer {
                reader,
                events: vec![],
                steps: 0,
            });
            let mut time = GameTime::default();
            time.set_scale(scale);
            world.insert_resource(time);
            let mut schedules = Schedules::new();
            schedules.set_fixed_timestep(1.0 / 128.0);
            schedules.get_mut::<FixedUpdate>().add(Consume);
            let runner = EcsRunner::single_thread();
            connect(&mut input.write(), GamepadId(1));
            let mut frame_events = vec![];
            for _ in 0..fps {
                // A complete tap between physics ticks must remain observable.
                button(&mut input.write(), GamepadId(1), true);
                button(&mut input.write(), GamepadId(1), false);
                schedules.run_frame(&mut world, &runner, 1.0 / fps as f64);
                input.write().begin_frame();
                frame_events.extend(frame_reader.drain());
            }
            assert_eq!(world.resource::<Consumer>().steps, (128.0 * scale) as usize);
            world.resource_mut::<GameTime>().set_scale(1.0);
            schedules.run_frame(&mut world, &runner, 1.0 / 128.0);
            let consumer = world.resource::<Consumer>();
            assert_eq!(consumer.events, frame_events);
            assert_eq!(consumer.events.len(), 1 + 2 * fps);
            assert!(
                !input.read().gamepads()[&GamepadId(1)]
                    .button(GamepadButton::South)
                    .pressed
            );
        }
    }
}

#[test]
fn analog_values_two_devices_disconnect_and_reconnect() {
    let mut input = WindowInput::default();
    let mut reader = input.subscribe();
    connect(&mut input, GamepadId(1));
    connect(&mut input, GamepadId(2));
    for angle in 0..64 {
        let angle = angle as f32 * ::std::f32::consts::TAU / 64.0;
        for radius in [0.05, 0.25, 0.75, 1.0] {
            let xy = [angle.cos() * radius, angle.sin() * radius];
            for (axis, value) in xy.into_iter().enumerate() {
                input.on_gamepad(GamepadEvent::Axis {
                    id: GamepadId(1),
                    axis: axis as u32,
                    value,
                });
            }
            let pad = &input.gamepads()[&GamepadId(1)];
            assert_eq!(
                [pad.axis(GamepadAxis::LeftX), pad.axis(GamepadAxis::LeftY)],
                xy
            );
            assert_eq!(
                input.gamepads()[&GamepadId(2)].axis(GamepadAxis::LeftX),
                0.0
            );
        }
    }
    button(&mut input, GamepadId(1), true);
    reader.drain();
    input.on_gamepad(GamepadEvent::Disconnected { id: GamepadId(1) });
    assert!(!input.gamepads().contains_key(&GamepadId(1)));
    assert!(input.gamepads().contains_key(&GamepadId(2)));
    let events = reader.drain();
    assert!(events.iter().any(|e| matches!(
        e,
        InputEvent::Gamepad(GamepadEvent::Button { pressed: false, .. })
    )));
    assert_eq!(
        events.last(),
        Some(&InputEvent::Gamepad(GamepadEvent::Disconnected {
            id: GamepadId(1)
        }))
    );
    connect(&mut input, GamepadId(3));
    assert_eq!(
        input.gamepads()[&GamepadId(3)].axis(GamepadAxis::LeftX),
        0.0
    );
    // Late messages from the old connection cannot drive the new one.
    button(&mut input, GamepadId(1), true);
    assert!(
        !input.gamepads()[&GamepadId(3)]
            .button(GamepadButton::South)
            .pressed
    );
}

#[test]
fn touch_identity_owner_origin_and_cancellation_are_stable() {
    let mut input = WindowInput::default();
    input.set_viewport(800.0, 600.0, 2.0);
    let mut reader = input.subscribe();
    touch(
        &mut input,
        1,
        TouchPhase::Started,
        [100.0, 200.0],
        InputOwner::Game,
    );
    touch(
        &mut input,
        2,
        TouchPhase::Started,
        [400.0, 200.0],
        InputOwner::Ui,
    );
    touch(
        &mut input,
        2,
        TouchPhase::Moved,
        [200.0, 250.0],
        InputOwner::Game,
    );
    touch(
        &mut input,
        1,
        TouchPhase::Moved,
        [-50.0, 700.0],
        InputOwner::Ui,
    );
    assert_eq!(input.touches()[&TouchId(1)].origin, [100.0, 200.0]);
    assert_eq!(input.touches()[&TouchId(1)].position, [-50.0, 700.0]);
    assert_eq!(input.touches()[&TouchId(1)].owner, InputOwner::Game);
    assert_eq!(input.touches()[&TouchId(2)].owner, InputOwner::Ui);
    input.cancel_mouse(InputCancelReason::PointerLeft);
    assert_eq!(input.touches().len(), 2);
    touch(
        &mut input,
        2,
        TouchPhase::Ended,
        [200.0, 250.0],
        InputOwner::Game,
    );
    assert_eq!(input.touches().len(), 1);
    input.set_viewport(600.0, 800.0, 3.0);
    assert!(input.touches().is_empty());
    touch(
        &mut input,
        1,
        TouchPhase::Moved,
        [300.0, 100.0],
        InputOwner::Game,
    );
    assert!(input.touches().is_empty());
    let events = reader.drain();
    assert!(events.iter().any(|e| matches!(e, InputEvent::Touch { contact, phase: TouchPhase::Cancelled } if contact.id == TouchId(1))));
}

#[test]
fn focus_loss_neutralizes_every_device_and_requires_fresh_input() {
    let mut input = WindowInput::default();
    let mut reader = input.subscribe();
    connect(&mut input, GamepadId(1));
    button(&mut input, GamepadId(1), true);
    input.on_gamepad(GamepadEvent::Axis {
        id: GamepadId(1),
        axis: 0,
        value: 0.5,
    });
    input.on_key_pressed(KeyCode::W);
    input.on_mouse_button(0, true);
    touch(
        &mut input,
        1,
        TouchPhase::Started,
        [50.0, 50.0],
        InputOwner::Game,
    );
    reader.drain();
    input.on_focus_changed(false);
    assert!(!input.focused);
    assert!(!input.is_key_pressed(KeyCode::W));
    assert!(!input.mouse_left);
    assert!(input.touches().is_empty());
    assert_eq!(
        input.gamepads()[&GamepadId(1)].axis(GamepadAxis::LeftX),
        0.0
    );
    assert!(
        !input.gamepads()[&GamepadId(1)]
            .button(GamepadButton::South)
            .pressed
    );
    input.on_key_pressed(KeyCode::W);
    button(&mut input, GamepadId(1), true);
    input.on_mouse_button(0, true);
    assert!(!input.is_key_pressed(KeyCode::W));
    assert!(!input.mouse_left);
    input.on_focus_changed(true);
    assert!(!input.is_key_pressed(KeyCode::W));
    assert!(
        !input.gamepads()[&GamepadId(1)]
            .button(GamepadButton::South)
            .pressed
    );
    button(&mut input, GamepadId(1), true); // fresh poll resumes gamepads
    assert!(
        input.gamepads()[&GamepadId(1)]
            .button(GamepadButton::South)
            .pressed
    );
    assert!(reader.drain().iter().any(|e| matches!(
        e,
        InputEvent::Touch {
            phase: TouchPhase::Cancelled,
            ..
        }
    )));
}

#[test]
fn mouse_ownership_is_latched_and_raw_injection_is_sanitized() {
    let mut input = WindowInput::default();
    input.on_mouse_button_owned(0, true, InputOwner::Ui);
    input.ui_wants_input = false;
    assert!(input.mouse_left);
    assert!(!input.game_mouse_pressed(0));
    input.on_mouse_button_owned(0, true, InputOwner::Game);
    assert!(!input.game_mouse_pressed(0));
    input.on_mouse_button(0, false);
    input.on_mouse_button(0, true);
    assert!(input.game_mouse_pressed(0));
    connect(&mut input, GamepadId(1));
    for (value, expected) in [(2.0, 1.0), (-3.0, -1.0), (f32::NAN, 0.0)] {
        input.on_gamepad(GamepadEvent::Axis {
            id: GamepadId(1),
            axis: 0,
            value,
        });
        assert_eq!(
            input.gamepads()[&GamepadId(1)].axis(GamepadAxis::LeftX),
            expected
        );
    }
}
