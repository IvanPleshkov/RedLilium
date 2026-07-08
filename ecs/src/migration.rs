//! Component migration system for handling schema changes across reloads.
//!
//! When a component type's schema changes, old serialized data may no longer
//! deserialize directly. This module provides a migration framework to transform
//! old-schema `Value`s into new-schema ones, or gracefully degrade by stripping
//! incompatible components.
//!
//! # Registration
//!
//! At plugin build time, register migrations via [`World::register_component_migration`]:
//! ```ignore
//! app.world.register_component_migration::<OldPosition>(
//!     |value, from_schema, to_schema| {
//!         if from_schema == 1 && to_schema == 2 {
//!             // transform value from v1 to v2
//!             Some(transformed_value)
//!         } else {
//!             None
//!         }
//!     }
//! );
//! ```
//!
//! # Execution
//!
//! During snapshot restore, if deserialization fails for a component:
//! 1. Check if a migration is registered for that type
//! 2. If found, apply it to the serialized value and retry deserialize
//! 3. If no migration, log a warning and skip (per-component degradation)

use std::collections::HashMap;

use crate::serialize::Value;

/// Describes a value-based migration for a component type.
///
/// A migration transforms a serialized `Value` from an old schema version
/// to a new one, or returns `None` to indicate no applicable migration.
/// The migration never touches old code or vtables — it's pure data.
///
/// The callback receives:
/// - `value: &Value` — the serialized component data
/// - `from_schema_version: u32` — the version recorded in the snapshot
/// - `to_schema_version: u32` — the version of the current component registration
///
/// Returns `Some(transformed)` if a migration was applied, `None` otherwise.
pub type ComponentMigrationFn =
    Box<dyn Fn(&Value, u32, u32) -> Option<Value> + Send + Sync + 'static>;

/// Registry of component migrations, keyed by component type name.
///
/// Migrations are looked up by name during deserialization; this allows
/// a snapshot to carry only type names (not `TypeId`s) and still find
/// the right migration in the current world.
#[derive(Default)]
pub struct MigrationRegistry {
    migrations: HashMap<&'static str, ComponentMigrationFn>,
}

impl MigrationRegistry {
    /// Creates an empty migration registry.
    pub fn new() -> Self {
        Self {
            migrations: HashMap::new(),
        }
    }

    /// Registers a migration function for a component type by name.
    pub fn register(&mut self, type_name: &'static str, migration: ComponentMigrationFn) {
        self.migrations.insert(type_name, migration);
    }

    /// Looks up a migration for a component type by name.
    pub fn get(&self, type_name: &str) -> Option<&ComponentMigrationFn> {
        self.migrations.get(type_name)
    }

    /// Attempts to apply a migration to a serialized value.
    ///
    /// Returns `Some(transformed)` if a migration exists and produces a value,
    /// `None` otherwise (no migration found, or migration declined).
    pub fn try_migrate(
        &self,
        type_name: &str,
        value: &Value,
        from_schema: u32,
        to_schema: u32,
    ) -> Option<Value> {
        self.migrations
            .get(type_name)
            .and_then(|f| f(value, from_schema, to_schema))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_registry_is_empty() {
        let registry = MigrationRegistry::new();
        assert!(registry.get("SomeComponent").is_none());
    }

    #[test]
    fn register_and_lookup_migration() {
        let mut registry = MigrationRegistry::new();
        let migration: ComponentMigrationFn = Box::new(|_value: &Value, from: u32, to: u32| {
            if from == 1 && to == 2 {
                Some(Value::Null)
            } else {
                None
            }
        });
        registry.register("TestComponent", migration);

        // Migration exists and matches versions
        let result = registry.try_migrate("TestComponent", &Value::Null, 1, 2);
        assert!(result.is_some());

        // Wrong versions → None
        let result = registry.try_migrate("TestComponent", &Value::Null, 2, 3);
        assert!(result.is_none());

        // Unknown component → None
        let result = registry.try_migrate("UnknownComponent", &Value::Null, 1, 2);
        assert!(result.is_none());
    }

    #[test]
    fn migration_can_decline_migration() {
        let mut registry = MigrationRegistry::new();
        let migration: ComponentMigrationFn = Box::new(|_value: &Value, _from: u32, _to: u32| None);
        registry.register("TestComponent", migration);

        // Even though migration exists, it returns None
        let result = registry.try_migrate("TestComponent", &Value::Null, 1, 2);
        assert!(result.is_none());
    }
}
