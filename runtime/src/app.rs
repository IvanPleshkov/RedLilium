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
    /// Cached handle to the world's `WindowInput` resource (the host writes
    /// platform input through it). This aliases the `Arc` stored in the world;
    /// a snapshot restore that re-inserted `WindowInput` would *replace* the
    /// world's `Arc` and desync this cache. Nothing registers `WindowInput` as
    /// a snapshot resource, so this is a latent trap, not a live bug — but keep
    /// it in mind before opting `WindowInput` (or any host-cached resource) into
    /// snapshots.
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

    /// First-boot construction: build a fresh [`App`], run the plugin's
    /// registration ([`build`](crate::Plugin::build)) and then populate the
    /// initial scene ([`spawn_scene`](crate::Plugin::spawn_scene)).
    ///
    /// This is the host-facing entry the runtime and the editor use to stand up
    /// a game from a loaded module; a warm reload uses [`reload`](Self::reload)
    /// instead, which restores the scene from a snapshot rather than spawning
    /// it. The caller drives startup/frames afterward (see the host loop).
    pub fn boot(engine: &EngineContext, plugin: &dyn crate::Plugin, aspect: f32) -> Self {
        let mut app = Self::new(engine, aspect);
        plugin.build(&mut app);
        plugin.spawn_scene(&mut app);
        app
    }

    /// Capture a name-keyed snapshot of the world for a warm-restart reload.
    ///
    /// Records every alive entity and every opted-in
    /// [`SnapshotResource`](redlilium_ecs::SnapshotResource); host/GPU managers
    /// injected by the [`EngineContext`] are excluded (they survive the reload
    /// as live `Arc`s, not through the snapshot). Take this **before** dropping
    /// the old world, then feed it to [`reload`](Self::reload).
    pub fn capture(
        &self,
    ) -> Result<redlilium_ecs::serialize::SerializedWorld, redlilium_ecs::serialize::SerializeError>
    {
        self.world.serialize_world()
    }

    /// Rebuild the app for a warm-restart reload (ADR-020): a fresh world with
    /// the plugin's registration re-run and the captured scene restored.
    ///
    /// Sequence: build a new [`App`] (std + rendering components, engine
    /// managers re-injected as the same shared `Arc`s), re-run
    /// [`Plugin::build`](crate::Plugin::build) to re-register the game's types,
    /// then restore `snapshot`. [`Plugin::spawn_scene`](crate::Plugin::spawn_scene)
    /// is deliberately **not** called — the snapshot is the scene — and startup
    /// is marked consumed so a later `run_startup` is a no-op.
    ///
    /// What does **not** carry across the reload, by design:
    ///
    /// - **`Startup` effects that aren't entity data.** A plugin's `Startup`
    ///   systems ran only on first boot. Anything they produced survives a
    ///   reload only if it's snapshot-captured (entities, or a
    ///   [`SnapshotResource`](redlilium_ecs::SnapshotResource)); otherwise
    ///   re-create it in [`build`](crate::Plugin::build).
    /// - **`Time`.** The reloaded world starts a fresh `Time` (elapsed resets to
    ///   0) — reload resets clock/undo/selection by design. Register `Time` as a
    ///   snapshot resource if a game needs continuity.
    ///
    /// Today every type registers under
    /// [`SourceId::HOST`](redlilium_ecs::SourceId); a real cross-dylib reload
    /// (#45 slice C) will stamp a fresh generation via
    /// [`World::with_registration_source`](redlilium_ecs::World::with_registration_source)
    /// so the downcast guard bites across generations.
    pub fn reload(
        engine: &EngineContext,
        plugin: &dyn crate::Plugin,
        aspect: f32,
        snapshot: &redlilium_ecs::serialize::SerializedWorld,
    ) -> Result<Self, redlilium_ecs::serialize::DeserializeError> {
        let mut app = Self::new(engine, aspect);
        plugin.build(&mut app);
        // The scene comes from the snapshot; `build` must not have spawned any
        // entities (contract: scene population lives in `spawn_scene`, skipped
        // on reload). A violation would stack the restored snapshot on top of a
        // freshly spawned scene — silent duplication that compounds per reload.
        debug_assert!(
            app.world.iter_entities().next().is_none(),
            "Plugin::build spawned entities; move scene population to spawn_scene \
             so a reload does not duplicate it on top of the restored snapshot"
        );
        app.world.deserialize_world_into(snapshot)?;
        // A reload takes its scene from the snapshot, never from Startup.
        app.schedules.mark_startup_done();
        Ok(app)
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{EngineContext, Plugin};
    use redlilium_ecs::{Component, MeshManager};
    use redlilium_graphics::GraphicsInstance;
    use redlilium_vfs::Vfs;

    /// A game component: registered in `build`, spawned in `spawn_scene`,
    /// serialized by its derive like any std component.
    #[derive(Clone, Component)]
    struct Blip {
        tag: u32,
    }

    /// The reload contract in miniature: `build` only registers, `spawn_scene`
    /// populates. A reload re-runs `build` and restores the snapshot, so the
    /// scene must survive without `spawn_scene` running again.
    struct BlipPlugin;
    impl Plugin for BlipPlugin {
        fn build(&self, app: &mut App) {
            app.register_component::<Blip>();
        }
        fn spawn_scene(&self, app: &mut App) {
            let world = app.world_mut();
            let e = world.spawn();
            world.insert(e, Blip { tag: 7 }).unwrap();
        }
    }

    fn test_engine() -> EngineContext {
        let instance = GraphicsInstance::new().expect("graphics instance");
        let device = instance.create_device().expect("graphics device");
        EngineContext::with_vfs(device, Vfs::new())
    }

    /// Warm-restart reload (ADR-020 slice A, no dylib): capture → drop world →
    /// rebuild via the same static plugin → the scene is restored from the
    /// snapshot (not re-spawned) and the persistent `EngineContext` managers
    /// survive as the *same* shared allocation.
    #[test]
    fn warm_reload_preserves_scene_and_engine_context() {
        let engine = test_engine();
        let plugin = BlipPlugin;

        // First boot: registration + initial-scene spawn.
        let mut app = App::new(&engine, 1.0);
        plugin.build(&mut app);
        plugin.spawn_scene(&mut app);
        {
            let blips = app.world().read_all::<Blip>().unwrap();
            assert_eq!(blips.iter().count(), 1, "one Blip before reload");
        }

        // The shared GPU manager the live world sees, before the reload.
        let mesh_before = app.world().resource_shared::<MeshManager>();

        // Capture the scene, then drop the old world entirely.
        let snapshot = app.capture().unwrap();
        drop(app);

        // Reload: fresh world, `build` re-run, scene restored from the snapshot.
        // `spawn_scene` is NOT called — a second spawn would duplicate entities.
        let app = App::reload(&engine, &plugin, 1.0, &snapshot).unwrap();

        // Scene survived, name-keyed: exactly one Blip, still tag 7.
        let tags: Vec<u32> = {
            let blips = app.world().read_all::<Blip>().unwrap();
            blips.iter().map(|(_, b)| b.tag).collect()
        };
        assert_eq!(tags, vec![7], "scene restored once, unchanged");

        // EngineContext survived: the reloaded world shares the same manager
        // allocation, not a freshly built one.
        let mesh_after = app.world().resource_shared::<MeshManager>();
        assert!(
            Arc::ptr_eq(&mesh_before, &mesh_after),
            "MeshManager Arc identity preserved across reload"
        );
    }
}
