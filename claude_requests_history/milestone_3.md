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
