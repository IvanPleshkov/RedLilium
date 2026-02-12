# Plugin System

## What Is It?

A plugin system provides a modular extension mechanism for an ECS framework. Plugins encapsulate related functionality — components, systems, resources, and configuration — into reusable units that can be added to an application with a single call. They enable ecosystem growth and code organization.

```rust
// Bevy-style plugin (not available in RedLilium)
pub struct PhysicsPlugin {
    pub gravity: Vec3,
    pub timestep: f64,
}

impl Plugin for PhysicsPlugin {
    fn build(&self, app: &mut App) {
        app.insert_resource(PhysicsConfig {
            gravity: self.gravity,
            timestep: self.timestep,
        })
        .add_systems(Update, (
            step_physics,
            sync_transforms.after(step_physics),
        ))
        .register_type::<RigidBody>()
        .register_type::<Collider>();
    }
}

// Usage
App::new()
    .add_plugins(DefaultPlugins)
    .add_plugins(PhysicsPlugin { gravity: Vec3::NEG_Y * 9.81, timestep: 1.0 / 60.0 })
    .run();
```

### Key Properties

- **Encapsulation**: Plugin bundles all related registrations (components, systems, resources, events).
- **Configuration**: Plugins accept parameters for customization.
- **Dependencies**: Plugins can declare dependencies on other plugins.
- **Plugin groups**: Combine multiple plugins into a single group (e.g., `DefaultPlugins`).
- **Conditional plugins**: Enable/disable features at build time.
- **Third-party ecosystem**: Community can publish reusable plugins as crates.

### Use Cases

- **Engine features**: Rendering, audio, input, networking as separate plugins.
- **Gameplay systems**: AI, inventory, dialogue, quest systems.
- **Editor tools**: Inspector, profiler, debug overlay as optional plugins.
- **Testing**: Mock plugins that replace production systems for testing.

## Current Approach in RedLilium

RedLilium uses manual registration functions instead of a plugin system:

```rust
// Current approach — manual setup
let mut world = World::new();
register_std_components(&mut world);  // Registers Transform, Camera, etc.

let mut systems = SystemsContainer::new();
systems.add(UpdateGlobalTransforms);
systems.add(UpdateCameraMatrices);
systems.add_edge::<UpdateGlobalTransforms, UpdateCameraMatrices>().unwrap();

// "Physics plugin" — manual
#[cfg(feature = "physics-3d")]
{
    systems.add(PhysicsSystem);
    systems.add_edge::<PhysicsSystem, UpdateGlobalTransforms>().unwrap();
}

world.insert_resource(PhysicsConfig::default());
```

This works for a single application but doesn't support:
- Reusable plugin crates with self-contained setup.
- Plugin dependency resolution.
- Plugin groups.
- Third-party ecosystem patterns.

## ECS Libraries That Support This

| Library | Implementation |
|---------|---------------|
| **Bevy** | `Plugin` trait, `PluginGroup`, `App::add_plugins()`, `DefaultPlugins`/`MinimalPlugins`, dependency checking, plugin build/finish/cleanup lifecycle |
| **flecs** | Modules: `ECS_MODULE(world, MyModule)`, auto-import dependencies, module-level component/system registration |
| **Unity DOTS** | Package Manager for feature modules, `ICustomBootstrap` for custom world setup, assembly definitions |
| **EnTT** | No built-in plugin system |
| **hecs** | No built-in plugin system |
| **Legion** | No built-in plugin system |
| **Shipyard** | Workloads serve as plugin-like modules |
