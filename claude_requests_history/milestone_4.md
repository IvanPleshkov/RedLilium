# Milestone 4: graphics API integration

This file describes all requests to claude code related to the milestone.

## Request 1:
Please audit the graphics crate, read also `docs\ARCHITECTURE.md`, `docs\DECISIONS.md`.
There is no mechanism for texture layout changing. Vulkan validator doesnt allow to skip it.
We need to implement the system for texture layout control and syncronization barriers placement in the graphics crate.
I don't want to do it manually, the graphics engine is on a higher level, where user should not think about it.
Instaed, I propose to implement new `TextureUsage`.
I want to have a `TextureUsage` as a plain graph with contains possible states of the texture.
Each texture has a `Arc<TextureUsage>` to share this struct between textures, where most of usages are the same.
We need also some special struct like `TextureLayoutController` (feel free to use another name), which is presented in `FramePipeline` or/and in `FrameSchedule`.
This struct helps to synchronize texture states and help to place barriers if they are needed.
Please try to implement this idea, dont forget to update `docs\ARCHITECTURE.md` and `docs\DECISIONS.md`.
Feel free to add unit tests.

Also we need an integration test, which creates two render textures and 2 quaq meshes.
We create one render graph, where quad is rendered to the render texture (pass 1).
Then we use render target from pass 1 as a texture for second quad and render in to the second texture (pass 2).
Next we read back the second render target (pass 3) and check using tools in integration tests that we have expected pixels at expected positions.
Keep attention that in such test we have image layout chagnes (between 1-2 and 2-3).
The graphics crate should automatically place required barriers and image layer transitions.
For example, the second render texture is known that it will be readed back because of their usage graph - we can change layout to the read back.
Please decide what to do, if usage of the graph provides non-unique transition layer (for instance, render target can be downloaded and sampled later),
maybe you need to upgrade material system for it.

Feel free to upgrade material system if material (or material instance) should have additional info to make barriers and texture layouts automatically.

## Request 2:
Please take a loot into `docs\ARCHITECTURE.md` and `docs\DECISIONS.md`.
There is an automatically placed barriers for texture usage in graphics crate.
Check if we need the same for buffers and do it if needed.

## Request 3:
Add to workspace a new crate app.
App is a library to create an application window with graphics from graphics crate.
App it a technical library without actual game code, no ecs here.
There is an `App` struct inside.
This struct is generic over the trait to handle window events and draw requests.
And over the trait to parse command line arguments.
Command line arguments trait also has a functions to use in app like graphics backend, windows mode and size etc.
Command line arguments trait has an additional function with `Option<u64>` how many should be processed before automatically exit.
It's helphul for AI agents and allow agents to check if validation errors exist.
Command line arguments for theese app params has default implementation.

Please read carefully `docs\ARCHITECTURE.md` and `docs\DECISIONS.md` to integrate graphics properly.
Use frame pipeline, frame schedule and resizing mechanism from graphics crate.

Next, we need a new demo with a perfect showcase of graphics crate.
As a result, I want a demo with a deferred pipeline, HDR, PBR with IBL, orbit camera (no ecs, it's a graphics showcase).
For image based lighting as a texture use this HDR texture, you can download dynamically in demo.
https://github.com/JoeyDeVries/LearnOpenGL/blob/master/resources/textures/hdr/newport_loft.hdr
Demo shows a grid of pbr lighted red spheres.

Add support of cubemap textures if graphics crate does not have it.

## Request 4:
Lets refactor pbr demo. I dont want to use harmonics yet. instead, I want to use environment map conversion into irradiance cubemap, like in learnopengl tutorials:
https://learnopengl.com/PBR/IBL/Diffuse-irradiance
https://learnopengl.com/PBR/IBL/Specular-IBL

## Request 5:
Lets refactor pbr demo. I dont want to calculate BRDF LUT, remove calculation and take it from
https://learnopengl.com/img/pbr/ibl_brdf_lut.png
App struct has a feature to run just N frames and exit. Run the demo with 10 frames and check that there is crash. If there is a crash, fix it

## Request 6:
Cool pbr demo! Can you also draw in background the cude texture (switch cubemap MIP manually using shift+number). by pressing number switch deferred channels to show that deferred works

## Request 7:
Please create a new demo in demos crate.
This demo should render a simple quad mesh, dont fill the whole screen, use just center place, background is cleared.
Demo downloads the texture from https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png
Please read `docs\ARCHITECTURE.md` and follow the architecture while implementing:
- all resources should be uploaded using render graphs
- use frame scheduler and frame pipeline
- use resizing tools from graphics crate
- use `App` from app crate
Please check that there is no errors in log, use please the tool in app that closes window after fixed frames (use 10).

## Request 8:
Please run textured quad demo with 10 frames before exit and with vulkan API backend (not wgpu).
Use validation layers and look into logs of application.
Fix vulkan api errors.
Probaply there is an issue with macos and moltenvk config, feel free to configure on local machine.

## Request 9:
Please run textured quad demo with 10 frames before exit and with vulkan API backend (not wgpu).
Use validation layers and look into logs of application.
On demo I see that there is no textured quad. I guess vulkan backend is missing something.
