# Runtime input

The stock runtime collects input before running schedules, independently of
`GameTime::scale()`. `WindowInput` is a public ECS resource containing keyboard,
mouse, touch and gamepad state. The same `on_*` methods used by the runtime are
available to headless tests; no window, GPU or platform backend is needed to
inject them. Games retain responsibility for action mapping, dead zones,
normalizing combined movement, and choosing the active player/controller.

## State and transitions

Read state for held actions. Use an `InputEventReader` for transitions such as
restart, jump, or the beginning/end of a drag:

```rust,ignore
// Once, when creating the consumer (before input arrives):
let reader = app.window_input().write().subscribe();
app.world_mut().insert_resource(MyInputConsumer { reader });

// In that consumer's Update OR FixedUpdate system:
for event in consumer.reader.drain() {
    // Each event is delivered once to this reader, in arrival order.
}
```

Each subscription is independent. An Update consumer cannot steal events from
a FixedUpdate consumer. Events survive frames with zero fixed steps, including
0× pause, and are not replayed in subsequent fixed steps after being drained.
Subscribe before input begins; a late subscription receives future events and
can initialize held state from `WindowInput`. Cloning `WindowInput` creates a
state snapshot, not another producer for existing subscriptions.

Unread events remain allocated until drained or the reader is dropped; there
is no silent overflow or two-frame expiration. Drop inactive consumers. If a
pause/menu should discard gameplay actions, drain that consumer during Update
or replace its subscription when resuming. Always use current state for held
thrust; a historical press event is not proof that a control remains held.
Only cursor/scroll frame deltas are cleared after schedules run, even when
drawing is skipped. Event queues and held state are not cleared by frame cleanup.

## Gamepads

`WindowInput::gamepads()` is keyed by `GamepadId`. An ID identifies one
connection, remains stable while connected, and changes on reconnection even
if a browser reuses its slot. It is not a persistent hardware serial number.
Disconnect emits neutral axis/button transitions before `Disconnected` and
removes that device. Other devices are unaffected.

For `GamepadMapping::Standard`, `pad.axis(GamepadAxis::LeftX)` and `LeftY`
are in `[-1, 1]`, with X positive right and Y positive down. Right stick uses
the same orientation. Intermediate magnitudes are retained; the engine applies
no gameplay dead zone, radial normalization, response curve or sensitivity.
`GamepadButton` names refer to physical positions, not printed A/B/X/Y labels.
Buttons expose both `pressed` and analog `value` in `[0, 1]`; triggers are
`LeftTrigger`/`RightTrigger`, shoulders are separate, and the D-pad is four
buttons. Standard numeric indices match the Web Gamepad standard layout.
Missing controls read as neutral.

Native desktop discovery uses gilrs, with its default dead-zone/jitter filters
disabled. Its mapped Y axes are converted to the engine's down-positive
orientation. Linux builds additionally require the libudev development package
and pkg-config. Backend initialization failure is logged and leaves gamepads
unavailable without preventing keyboard/mouse startup.

WASM polls `navigator.getGamepads()` every real frame. A browser may hide devices
until the user activates a controller; polling continues, so that initial
gesture discovers it. This API requires a supported browser in a secure context
(HTTPS or localhost) and permission when embedded. A nonstandard browser layout
is marked `Unmapped`: semantic accessors return neutral, while raw normalized
`axes`/`buttons` remain available for explicit game bindings. Raw indices are
device/backend-specific for unmapped devices.

Native event transitions are retained, including a press/release pair between
frames. Browsers expose snapshots: a tap entirely between browser samples may
be unobservable to any Gamepad API client. Once sampled, engine events are
retained independently of the physics frequency.

## Touch, coordinates and ownership

`WindowEvent::Touch` becomes `TouchInput` through the normal app/runtime path.
`TouchId` identifies a contact lifetime; platform IDs can be reused without
reusing engine IDs. `WindowInput::touches()` contains active contacts, and
`InputEvent::Touch` reports Started, Moved, Ended and Cancelled. A contact retains
its original position and its owner throughout the gesture.

Positions and origins use physical pixels relative to the viewport's top-left,
X right and Y down. Divide by `WindowInput::scale_factor` for logical pixels.
`window_width`/`window_height` describe the actual input viewport, even while a
GPU resize is pending. Positions may lie outside it during capture. Resize or
DPI changes cancel active gestures; a fresh down starts in the new coordinates.

At Started, the runtime hit-tests the contact position against egui's existing
UI geometry, at the current DPI. Ownership is available before FixedUpdate;
it does not depend on last frame's mouse-hover flag. A contact starting on UI
stays `InputOwner::Ui` even after crossing into the game. Gameplay contacts
stay `Game` when crossing UI and do not feed egui's pointer emulation. UI
touches produce egui touch events; only its first contact drives the UI pointer.
Lifting that finger never promotes an already-held second finger.

For a floating game stick, accept a Game-owned Started event only when no drag
is active, retain that ID and origin, and process only events for that ID until
Ended/Cancelled. Do not select a new contact by scanning the map every frame:
that would transfer control to a second finger. UI geometry is from the most
recent UI pass; when opening a modal or changing gameplay mode, call
`input.cancel_pointers(InputCancelReason::UiCaptured)` before consuming gameplay
input. Cancelled contacts ignore late moves/releases until a new Started event.

For mouse gameplay use `game_mouse_pressed(0)` (0 left, 1 right, 2 middle) or
the owner on MouseButton events. Ownership is latched at press. The legacy
`mouse_left`/`mouse_right`/`mouse_middle` flags remain raw state, and
`ui_wants_input` remains a coarse keyboard/mouse hint rather than gesture ownership.

The web host sets `touch-action: none` on its canvas, prevents default pointer
gestures, and captures touch pointers. Off-canvas movement/release therefore
continues the contact. `pointerup` ends the gesture normally; the browser's
following automatic `lostpointercapture` does not cancel it or erase its release.
Capture loss during an active gesture and OS `pointercancel` cancel without
synthesizing a click. This applies to mouse/pen as well as touch. DOM listeners
track gesture lifetimes and order supplemental cancellations after preceding
winit button/contact transitions, including when those transitions are buffered.
Button/contact input itself continues to come from winit. A canvas
cursor exit cancels mouse drags but preserves captured touch contacts. Native
touch uses winit's platform contact routing. Promoted touch mouse input is
suppressed by the browser/winit; on Windows the host's window subclass filters
the [OS-tagged compatibility mouse messages](https://learn.microsoft.com/en-us/windows/win32/tablet/system-events-and-mouse-messages).
Real mouse input can coexist with touch; there is no time-window heuristic.
The Windows bridge also forwards capture-loss and cancelled-pointer messages
that winit 0.30 does not expose, including synchronously sent notifications.

## Focus and cancellation

`WindowInput::focused` is effective input focus: losing window focus or becoming
hidden/occluded neutralizes keys, mouse buttons, gamepad axes/buttons and touch
contacts immediately. Readers receive release/cancel transitions plus
`Focused(false)`. Visibility changes on web use winit's document visibility
notifications and do not wait for a physics tick. egui also receives focus and
touch cancellation; cancellation does not synthesize a click.

Returning focus does not resurrect keyboard/pointer holds. Synthetic focus
presses and repeats of keys absent from current state are ignored; a fresh
press/contact is required. Gamepads resume from the next fresh platform sample,
so a physically held stick resumes after focus returns. While unfocused,
incoming held values remain neutral. `cancel_all(reason)` and
`cancel_pointers(reason)` provide explicit headless/game-side cancellation.

## Validation

`cargo test -p redlilium-ecs --test input_devices` exercises public injection,
analog circles, two devices, disconnect, focus, ownership, cancellation and
independent readers at 32/64/256 FPS and 0×/0.25×/1× game time. App unit tests
cover winit contact IDs and DOM/winit cancellation ordering (completed gestures,
active capture loss, mouse cancellation, multiple contacts and chorded buttons).
Graphics egui tests cover DPI hit testing, mouse clicks within/across frames,
late releases after cancellation, real taps and non-promotion of the second finger. macOS, Windows
and wasm32 runtime targets are compile-checked. These automated checks do not claim
physical controller/touchscreen or all-platform driver validation.
