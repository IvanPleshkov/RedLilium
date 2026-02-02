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


## TODO: the same for buffers as for textures
