# Reflection and Inspector

The ECS provides runtime reflection for components via the `Component` trait and an integrated egui inspector UI for editing component values at runtime.

## Component Trait

The `Component` trait extends `Send + Sync + 'static` with reflection capabilities:

```rust
pub trait Component: Send + Sync + 'static {
    /// Static name string (e.g., "Transform")
    const NAME: &'static str;

    /// Instance method returning the name
    fn component_name(&self) -> &'static str { Self::NAME }

    /// Render inspector UI for this component's fields
    fn inspect_ui(&mut self, ui: &mut egui::Ui);
}
```

## Derive Macro

Use `#[derive(Component)]` to auto-implement the trait:

```rust
use redlilium_ecs::Component;

#[derive(Component)]
struct Health {
    current: f32,
    max: f32,
}

// Automatically generates:
// - NAME = "Health"
// - inspect_ui() that renders sliders for f32 fields
```

The derive macro uses the `Inspect` wrapper to generate appropriate UI widgets for each field type.

## Manual Implementation

For custom inspector layouts:

```rust
impl Component for CustomType {
    const NAME: &'static str = "CustomType";

    fn inspect_ui(&mut self, ui: &mut egui::Ui) {
        ui.horizontal(|ui| {
            ui.label("Value:");
            ui.add(egui::Slider::new(&mut self.value, 0.0..=100.0));
        });
    }
}
```

## Inspector Registration

Components are registered at three levels:

### Storage Only

```rust
world.register_component::<InternalFlag>();
// Component can be inserted/queried but not visible in inspector
```

### Inspector Visible

```rust
world.register_inspector::<Camera>();
// Visible in inspector (can view and edit fields)
// Cannot be added via "Add Component" button (no Default)
```

### Inspector with Add Support

```rust
world.register_inspector_default::<Transform>();
// Visible in inspector AND can be added via button (uses Default::default())
```

## Inspector Operations

The World provides type-erased inspector operations:

```rust
// List components an entity has (inspector-registered only)
let components = world.inspectable_components_of(entity);
// e.g., ["Transform", "Camera", "Visibility"]

// List components that can be added (have Default)
let addable = world.addable_components_of(entity);
// e.g., ["PointLight", "SpotLight"] — components not yet on this entity

// Render inspector UI for a component by name
world.inspect_by_name(entity, "Transform", &mut ui);

// Remove a component by name
world.remove_by_name(entity, "Health");

// Add a default component by name
world.insert_default_by_name(entity, "Visibility");
```

## Inspect Trait (Internal)

The `Inspect` wrapper provides compile-time dispatch for UI rendering:

```rust
// For types that support egui widgets (f32, u32, bool, Vec3, etc.)
Inspect(&mut value).show("field_name", ui);

// Fallback for types without widget support
// Shows a read-only label with Debug representation
InspectFallback::show(&mut Inspect(&mut arc_data), "data", ui);
```

Supported types get interactive widgets:
- `f32`, `f64` → Drag value
- `u8`, `u16`, `u32`, `u64`, `i32`, etc. → Drag value
- `bool` → Checkbox
- `String` → Text edit
- `Vec3`, `Vec4`, `Quat`, `Mat4` → Multi-field editors

Unsupported types fall back to a read-only display.

## Inspector UI Modules (Feature-Gated)

The `inspector` feature enables full UI modules:

```rust
// In Cargo.toml:
// redlilium-ecs = { features = ["inspector"] }
```

Modules available with the feature:
- `ui::component_inspector` — Per-entity component editor panel
- `ui::world_inspector` — World-wide entity browser

## InspectorEntry (Internal)

Each registered component has a type-erased `InspectorEntry` stored in the World:

```rust
struct InspectorEntry {
    has_fn: fn(&World, Entity) -> bool,           // Check component presence
    inspect_fn: fn(&mut World, Entity, &mut Ui) -> bool,  // Render UI
    remove_fn: fn(&mut World, Entity) -> bool,     // Remove component
    insert_default_fn: Option<fn(&mut World, Entity)>,     // Add default
}
```

This enables the inspector to work with any component without knowing its concrete type.

## Pod Components

Components implementing `bytemuck::Pod + Zeroable` get additional capabilities:

```rust
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
struct Transform {
    translation: Vec3,
    rotation: Quat,
    scale: Vec3,
}
```

Pod components support:
- Byte-level serialization (`bytemuck::bytes_of`)
- GPU buffer upload (direct memcpy)
- Snapshot/rollback (copy bytes)

## Public API

### Component Trait

| Item | Description |
|------|-------------|
| `Component::NAME` | Static name string |
| `component_name()` | Instance name method |
| `inspect_ui(&mut self, ui)` | Render inspector UI |

### World Inspector Methods

| Method | Description |
|--------|-------------|
| `register_inspector::<T>()` | Register with inspector (view only) |
| `register_inspector_default::<T>()` | Register with inspector + add support |
| `inspectable_components_of(entity)` | List visible component names |
| `addable_components_of(entity)` | List addable component names |
| `inspect_by_name(entity, name, ui)` | Render UI by name |
| `remove_by_name(entity, name)` | Remove by name |
| `insert_default_by_name(entity, name)` | Add default by name |
