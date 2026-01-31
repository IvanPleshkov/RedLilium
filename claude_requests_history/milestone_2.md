# Milestone 2: Start Graphics

This file describes all requests to claude code related to the milestone.

## Request 1:
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

## Request 2:
Before we start with actual implementation of the render graph, we need to prepare the scene.
I would like to use ECS from bevy https://github.com/bevyengine/bevy version `0.18.0`. Read carefully the dependency project to take a main points of custom ECS ideas from bevy.
Let's create a new crate in the project with common ecs components and systems. separate please components and systems to different folders to keep ECS sense.
As a basic components let it be components for transform, material, render mesh, collision.
Please also design component and system to design child components to have a local transform for a complicated prefabs.

## Request 3:
There are a basic ECS components in the ecs crate of the project and empty abstract rendering graph.
To implement an abstract rendering gragh it's required to glue ECS and rendering system.
One of the key concept of the renreding graph - it may be not unique in the process (please add this decision to docs/DECISIONS.md).
But it's okay handle one ecs world to multiple render graphs.
Each world can have their unique set of render graphs.
But backend is the same and sync patterns can be shared between ECS worlds.
Meanwhile, the process can contain multiple ECS worlds.
Please make all glue and update the demo project to handle the simple scene.
Skip camera and light components for now, camera and light will be implemnted later.

## Request 4:
There is a SceneRenderer to glue ECS and render graph. I dont like this approach because in the demo this structure is over the ECS and rendering.
Such implementation makes hard to support rendering to texture feature and multiple cameras.
Let's refactor this.
We need a camera component with enum of camera variants (Perspective, Orthographic, etc).
Also we need a system which actually handles a render graph for each camera.
I guess the system may be implemented in graphics crate and camera component is a basic component in ecs crate.
Camera component has a render target.
Camera system handles the rendering by a separate function. This system merges all render graphs from each camera. Camera with surface tagrets (may be multiple because of view) are applied last.
Rendering to texture feature has a priority to set explicitly which cameras should be rendered first.

## Request 5:
Let's refactor resources usage in graphics crate.
I dont like the approach, where render graph has a resource creation methods.
Let's create a graphics instance and graphics device. Instance can handle multiple devices.
Device can create resources. Resources are handled by `Arc`.

## Request 6:
Please change `Texture::device` to `Arc`. Do it also for `Buffer`, `Sampler`. Change also `GraphicsDevice::instance` to `Arc`.
The motivation is keeping device and instance alive when depend resource is alive.


## Request 7:
Let's refactor render graph in graphics crate. `TextureHandle` and `BufferHandle` are not needed anymore. Please use `Arc<Buffer>` and `Arc<Texture>` instead.

## Request 8:
Let's add material to the graphics crate.
I wish to have a `Material` and `MaterialInstance`. Let me describe both.
`Material` is a structure, created by devide and it holds the material related stuff like shader, bindings, etc.
`MaterialInstance` contains the reuired data to perform actual rendering like `Arc<Buffer>` etc. `MaterialInstance` contains `Arc` to related `Material`.
Also `MaterialInstance` bindings are described by separate structure to contain layout - we will make it complicated later and optimized to minimize render state switches (like camera uniform is definetly must be shared between scene objects draw calls). But keep in mind while designing that se wish to reduce unnecessary binding changes.

## Request 9:
In material system from graphics crate I wish to remove `BindingFrequency`.
I don't like this approach because I don't want to rely on high level rendering decisions in the base graphics crate.
I have another approach instead of `BindingFrequency`.
Please change `MaterialInstance::binding_groups` type to `Vec<Arc<BindingGroup>>`. The same for `MaterialDescriptor::binding_layouts`, the new type is `Vec<Arc<BindingLayout>>`.
Instead of sorting by binding frequency, we will implement later a resolver, which compares `Arc` pointers to group objects and reduce render state changes.

## Request 10:
Let's refactor render graph in graphics crate.
We added recently the material system to the graphics crate.
Let's remove `add_buffer` and `add_texture`. Now we can rely on material system.

## Request 11:
To render basic scene using graphics crate, there is swapchain abstraction required.
Please add swapchain. You must rely on the fact, that only `winit` crate is used to handle windowing and surface.

## Request 12:
In graphics crate there is a render graph.
It seems render graph passes do not support render targets.
I let you modify render graph to enable rendering targets.
Rendering to surface and to texture of course must be supported.

## Request 13:
Lets refactor render graph in graphics crate.
How it relies on the Hander with index.
Let's change is to using `Arc` like all other resources in graphics crate.

## Request 14:
Let's extend the render graph features in graphics crate.
Add please support of data transferring. Think first, is data transferring a pass or it should be a separate struct, explain your decision.

## Request 15:
Lets refactor render graph in graphics crate.
I wish to rename `RenderPass` into `Pass`.
Make please `Pass` as an enum with pass variants: graphics, transfer, compute.

## Request 16:
I'm not sure if `target.rs` file is placed correctly.
Please decide the proper place of this file or keep as is if you think that the current file place is a best option.

## Request 17:
Please take a loot to the render graph implementation. It seems usage of Arc is not optimimal because it produces a lot of allocations each frame.
Discuss please options how to fix it. Is it possible to reuse dead `Arc`s from prev frame?

## Request 18:
Each `Pass` in render graph from graphics crate has a vector of dependencies. It looks not efficient because it produces allocations for each pass.
Please refactror is and let render graph keep dependencies instead of pass.

## Request 19:
Please change `RenderGraph` functions: `add_graphics_pass`, `add_transfer_pass`, `add_compute_pass`.
Provide as arguments `GraphicsPass`, `TransferPass`, `ComputePass` instead of just a name with getters.
You can also remove getters because I expect, that getters are not required.

## Request 20:
I have a question about usage of RenderGraph in graphics project.
If I have for instance a depth prepass and shadow map, do I need two separate render graphs and sync between then?
Or the common practice is a collecting everything into a huge single render graph?
I guess multiple render graphs are better because small pieces are much easier to compile,
no any mutex is required because each graph is constructed separately.
If I'm right please provide a solution to handle a dependencied between multiple render graphs.
Is it a good idea to add additional variant to `Pass` enum? It's easy to use. On another hand, another render graph is not a render pass, isnt it?

### In addition to 20:
Before my decision, answer please my questions about frame schedule. Can frame sheduler sync shadow map and the light resolver?
It seems some barrier is required in between and it's not clear how to find such dependency.
Second question: is compiled graph executed immediately after push to frame scheule?
And next, I want to use modern graphics approach where we dont wait for swapchain and start working on the next frame before current is shown.
Is it possible with frame scheduler?

### In addition to 20:
I like the idea about frame schedule. Regarding to my question about sync between graphs, I would like to use an approach to analyse reads and writes.
I think it's not expensive if compiled graph has a map from resource to usages. But skip it for now.
Before we start, Also explain me more about immediate execution.
In your approach I guess there is a problem: where shadow render graph is ready to draw, is still waits for all other graphs.

### In addition to 20:
I would like to use streaming approach and start immediately. Let's rock! please use separate folder in graphics crate to implement render scheduler.

## Request 21:
Please take a look of render schelured in graphics crate.
Explain please how to cover window close event? Is it looks fine for gracefully end of the game loop?

### In addition to 21:
Yes, implement please the frame pipeline in the graphics crate in the separate folder.
Please make a lot of attention to documentation.
It seems there are a gap of render pipeline, render scheduler and render graph

## Request 22:
The file `graphics\src\pipeline\mod.rs` has a great description of rendering architecture of the graphics crate.
Can you please move the global vision of the rendering process to the follow list of targets (decide if its needed to be shorted for each target):
- `lib.rs` of the graphics crate
- `docs\ARCHITECTURE.md`
- `docs\DECISIONS.md`

## Request 23:
Read the graphics architecture in `docs\ARCHITECTURE.md`.
I think it's a good decision to create a frame pipeline as a method of device, frame scheduler as a method of pipeline.
But scheduler is created separately. WDYT?

## Request 24:
Read the graphics architecture in `docs\ARCHITECTURE.md`.
There is a solution how to stop gracefully.
Think please if such architecture is okay in case of window resize?
Which strategies are there about window resize and can current approach handle it?
It's not required to do a realtime resizing. But very long waiting for the resize which is common in toy engines, it not acceptable, we develop a professional high-end solution.

## Request 25:
Let's start to implement render graph compilation.
As a first step, move please the CompiledGraph to the separate folder in the graphics crate.
Then, create in this folder a compile function without implementation for now.

## Request 25:
Take a look to the compiled graph in the graphics crate.
There is a `Vec` with ordered list of execution.
Are you sure that render graph sould be compiled into the ordered command list to a single command buffer?
Reead also `docs\ARCHITECTURE.md` to understand the graphics architecture before decision.
I can say that the list of execution is not very optimal.
On another hand, there are multiple graphs and for effective parallel drawing maybe render scheduler works fine
and it's not required to make an graphics engine overcomplicated and pay CPU price for the advanced compilation.

## Request 26:
It's time to implement graph render compilation.
