## Milestone 2: Start Graphics

This file describes all requests to claude code related to the milestone.

Request:
Lets start a milestone 2. Milestone 2 is about graphics.
Let me describe the goals to achieve. The projects should have:
1. An abstract render graph. This render graph is required to define of high level all render operations and let the executor set proper low-level sync between passes and states of recources.
2. There are 3 backgrounds under abstract render graph.
It's a Vulkan API which to be expected to achieve all gpu powerfull. Also it allows to use high-end extensions and render graph is expected to be extendable.
It's a wgpu crate https://github.com/gfx-rs/wgpu of version `28.0.0` to support web and targets without vulkan api.
And it's a dummy API to use in tests where graphics is not required.
3. The render graph is optimized to use in multithreaded environments.
Right now it's too complicated to implement this milestone. Let's start from the plan. Please plan this feature and describe it in the `docs/ROADMAP.md` file.
Dont forget to update `docs/DECISIONS.md` file.
Also change the documentation of graphics crate and create kust empty basic structs ans traits without implementation.
