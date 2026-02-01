# Milestone 3: graphics API integration

This file describes all requests to claude code related to the milestone.

## Request 1:
Let's do a huge work!
Read `docs\ARCHITECTURE.md`, `docs\DECISIONS.md` and the whole graphics crate in the workspace.
And let's implement this rendering abstraction using `wgpu` version "28.0.0". And using vulkan api and `ash` crate version '0.38.0'.
About wgpu you can read here: https://github.com/gfx-rs/wgpu.
About ash here: https://github.com/ash-rs/ash.
While working, enable please tests in `graphics\tests\gpu_tests.rs` one by one to check, that everything works.

## Request 2:
Let's do a huge work!
Read `docs\ARCHITECTURE.md`, `docs\DECISIONS.md` and the whole graphics crate in the workspace.
Graphics crate has already a backend using `wgpu` crate. Now it's time to implement the vulkan api backend using `ash`.
About ash here: https://github.com/ash-rs/ash.
While working, use tests in `graphics\tests\gpu_tests.rs` to check, that everything works.
Don't forget to provide a flag to enable debug layers and use debug layer in tests, no errors should be from validation layers.

## Request 3:
Read `docs\ARCHITECTURE.md`, `docs\DECISIONS.md` and the whole graphics crate in the workspace.
See the graphics crate and Instance structure.
I dont like `Instance::backend` function because I dont want to show inner graphics as a public API.
Please remove `Instance::backend`.
This change will break tests. Fix it using Frame scheduler design described in docs.
You can add some features in graphics crate if you think it's required to fix tests.

## Request 4:
Let's change integration tests in graphics crate.
I dont like that tests rely on `GraphicsDevice::execute_graph` because there is already a frame scheduler, who actually must perform graph executions.
Read `docs\ARCHITECTURE.md`, `docs\DECISIONS.md` to understand frame scheduler.
Please remove `GraphicsDevice::execute_graph` and change integration graphics tests so that tests use `FrameSchedule`.
Feel free to upgrade graphics api if something is missing to refactor tests.

## Request 5:
There is `GpuBackend` trait in graphics crate.
Because we have a fixed list of supported backends, we don't need actually a trait and dyn for it.
Let's refactor it. Change `GpuBackend` to the enum. All trait functions change to the enum functions.

## Request 6:
Graphics crate has integration tests.
Please add test with a basic WGSL shader. The test using this shader renders a quad in the half of the small render target texture.
Then, texture is read back to ram and you can use tools in integration tests to check pixel values.
Feel free to implement missing code in graphics backend.

## Request 7:
Please refactor `graphics/src/backend/wgpu_backend.rs` and split it into multiple files like in `graphics/src/backend/vulkan`.

## Request 8:
Please look at `execute_graph` function in the graphic integration tests.
It calls `execute_graph` from device and later uses scheduler.
It looks incorrect. `schedule` should do an graph execution.
Please remove `execute_graph` from device and from all backends.
Make scheduler works, it's should be possibe to execute gpu tasks using scheduler only.

## Request 9:
Please audit vulkan api integration in graphics crate.
`wgpu` crate (another backend) uses the coordinate systems of D3D and Metal. Depth ranges from [0, 1].
It differs from vulkan api coordinate system.
I preffer to use `wgpu` approach.
If you can, find how coordinate system is resolved in `wgpu` sources with vulkan.
Make the same for this project. If you cannot find, look at possibilities. Maybe there is some vulkan extension which allows to use D3D coordinate system.
Update also `docs\ARCHITECTURE.md` and `docs\DECISIONS.md`, the desicion about coordinate system is important and should be mentioned.

## Request 10:
Regarding `docs\ARCHITECTURE.md`,
RedLilium uses the **D3D/wgpu coordinate system convention** for consistency across backends:
| **Y-Axis (NDC)** | +Y points down |
| **Screen Origin** | Top-left corner |
| **Winding Order** | Counter-clockwise (CCW) front faces |
Please check if both vulkan api and wgpu backends in graphics crate follow this decision.

## Request 11:
Let's create a new integration test in graphics crate.
This test tries to create a window using `winit` crate.
It creates graphics, uses swapchain, draw 5 frames and closes.
Please use frame schedule and pipeline described in `docs\ARCHITECTURE.md`.
If possible to read back swapchain, feel free to read it and check using tools in graphics integration tests.
If window cannot be created or there is no any device compatible to the surface - finish test as passed.
Tesh should pass in this case because of CI running.

## Request 11:
Please explain this line comment and use actual swapchain even if it brakes readback in test

## Request 12:
Let's add to the instance in graphics crate in `new` a new agrument with parameters. parameters have a builder.
In parameters we need to provide a backend to use (in case of `wgpu` - also a backend for wgpu).
Also parameters can configure debug and validation layers.
Change integration tests and in window test use both backends (using `rstest`).

## Request 13:
Please fix test_window_swapchain_5_frames test with case::vulkan case.

## Request 14:
test_window_swapchain_5_frames test produces warning in the logs:
WARN  redlilium_graphics::backend::wgpu_impl::pass_encoding Pass 'frame_3' has surface attachment - swapchain rendering not fully implemented
Please fill the gaps to make this test fully functional
