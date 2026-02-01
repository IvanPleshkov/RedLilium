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
