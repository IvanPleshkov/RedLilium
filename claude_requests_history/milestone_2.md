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
