//! The [`App`] builder handed to [`Plugin::build`](crate::Plugin::build).

use std::sync::Arc;

use parking_lot::RwLock;

use redlilium_ecs::{
    AssetGpuFlush, AssetPump, Component, FlushUploads, ForwardRender, FrameRing, HotReload,
    MaterialInstanceLoad, MeshLoad, PostUpdate, Render, RenderSchedule, Resource, ScenePass,
    ScheduleLabel, Schedules, System, SystemsContainer, UpdateCameraMatrices,
    UpdateGlobalTransforms, WindowInput, World, register_rendering_components,
    register_std_components,
};

use crate::EngineContext;

/// Builder over the game world and its schedules.
///
/// Created by the runtime after the graphics device exists; the game's
/// [`Plugin`](crate::Plugin) configures it. Std + rendering components are
/// already registered and the engine's default systems are already installed:
///
/// - `PostUpdate`: `UpdateGlobalTransforms` → `UpdateCameraMatrices`, plus
///   the asset-loading systems (`HotReload`, `MeshLoad`, `MaterialInstanceLoad`,
///   `AssetPump`).
/// - `Render`: `FlushUploads` / `AssetGpuFlush` → `ForwardRender`.
///
/// Game systems slot into any schedule via [`add_system`](Self::add_system)
/// (or [`schedule_mut`](Self::schedule_mut) for dependency edges); full
/// access to the world is available through [`world_mut`](Self::world_mut).
pub struct App {
    world: World,
    schedules: Schedules,
    window_input: Arc<RwLock<WindowInput>>,
    aspect: f32,
}

impl App {
    pub(crate) fn new(engine: &EngineContext, aspect: f32) -> Self {
        let mut world = World::new();
        register_std_components(&mut world);
        register_rendering_components(&mut world);
        engine.inject_into(&mut world);

        // Per-world render plumbing: the frame graph slot, the forward pass
        // handle, and the per-draw dynamic-uniform ring.
        world.insert_resource(RenderSchedule::empty());
        world.insert_resource(ScenePass::default());
        let frame_ring = FrameRing::new(engine.device(), 1 << 20, "game_frame_ring")
            .expect("failed to create game frame ring");
        world.insert_resource(frame_ring);
        let window_input = world.insert_resource(WindowInput::default());

        let mut schedules = Schedules::new();
        {
            let post = schedules.get_mut::<PostUpdate>();
            post.add(UpdateGlobalTransforms);
            post.add(UpdateCameraMatrices);
            post.add_edge::<UpdateGlobalTransforms, UpdateCameraMatrices>()
                .expect("no cycle");
            post.add(HotReload);
            post.add(MeshLoad::default());
            post.add(MaterialInstanceLoad);
            post.add(AssetPump);
            post.add_edge::<MeshLoad, AssetPump>().expect("no cycle");
        }
        {
            let render = schedules.get_mut::<Render>();
            render.add(FlushUploads);
            render.add(AssetGpuFlush);
            render.add(ForwardRender);
            render
                .add_edge::<FlushUploads, ForwardRender>()
                .expect("no cycle");
            render
                .add_edge::<AssetGpuFlush, ForwardRender>()
                .expect("no cycle");
        }

        Self {
            world,
            schedules,
            window_input,
            aspect,
        }
    }

    /// The game world.
    pub fn world(&self) -> &World {
        &self.world
    }

    /// Mutable access to the game world (spawn the initial scene here).
    pub fn world_mut(&mut self) -> &mut World {
        &mut self.world
    }

    /// Mutable access to the schedules.
    pub fn schedules_mut(&mut self) -> &mut Schedules {
        &mut self.schedules
    }

    /// The systems container of one schedule — for dependency edges,
    /// conditions, and other advanced wiring.
    pub fn schedule_mut<L: ScheduleLabel>(&mut self) -> &mut SystemsContainer {
        self.schedules.get_mut::<L>()
    }

    /// Register a game component type (storage + name index, so the
    /// component participates in serialization and inspection).
    pub fn register_component<T: Component>(&mut self) -> &mut Self {
        self.world.register_inspector::<T>();
        self
    }

    /// Insert a resource into the world.
    pub fn insert_resource<T: Resource>(&mut self, value: T) -> &mut Self {
        self.world.insert_resource(value);
        self
    }

    /// Register an event type (see [`redlilium_ecs::Events`]).
    pub fn add_event<T: Send + Sync + 'static>(&mut self) -> &mut Self {
        self.world.add_event::<T>();
        self
    }

    /// Add a system to the schedule `L`.
    pub fn add_system<L: ScheduleLabel, S: System>(&mut self, system: S) -> &mut Self {
        self.schedules.get_mut::<L>().add(system);
        self
    }

    /// Build another plugin into this app.
    pub fn add_plugin(&mut self, plugin: &dyn crate::Plugin) -> &mut Self {
        plugin.build(self);
        self
    }

    /// Aspect ratio of the window at startup — for constructing the initial
    /// camera projection.
    pub fn initial_aspect(&self) -> f32 {
        self.aspect
    }

    /// Handle to the [`WindowInput`] resource, kept by the host to forward
    /// platform input events.
    pub(crate) fn window_input(&self) -> Arc<RwLock<WindowInput>> {
        self.window_input.clone()
    }

    /// Split borrow for the host frame loop.
    pub(crate) fn parts_mut(&mut self) -> (&mut World, &mut Schedules) {
        (&mut self.world, &mut self.schedules)
    }
}
