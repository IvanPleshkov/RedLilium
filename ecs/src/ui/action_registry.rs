//! Name → constructor registry over [`EditAction`]s (docs/REMOTE.md, "любое
//! действие AbstractEditor доступно через канал").
//!
//! Each entry builds an action from wire parameters (a natural-RON map), so a
//! remote client can invoke it by name: the constructed action goes through
//! the same `ActionQueue` → history path as the UI's. Registered as a world
//! resource; shells and games can [`register`](ActionRegistry::register)
//! their own actions next to the built-ins.
//!
//! Constructors resolve entity specs (`"index@spawn_tick"`) against the live
//! world at dispatch time; the action itself re-validates liveness at apply
//! time (a frame later).

use std::collections::BTreeMap;

use redlilium_core::abstract_editor::EditAction;

use super::component_inspector::{
    AddComponentAction, ImportComponentAction, RemoveComponentAction,
};
use super::world_inspector::{DeleteEntityAction, ReparentAction, SpawnEntityAction, SpawnReport};
use crate::serialize::{SerializedComponent, find_entity, natural_to_value};
use crate::{Entity, Parent, World};

/// A constructed action plus the feedback slot the caller polls after apply.
pub struct RegisteredAction {
    pub action: Box<dyn EditAction<World>>,
    /// Spawn-style actions report the entities they created here (the ids
    /// don't exist until the action applies).
    pub spawned: Option<SpawnReport>,
}

impl RegisteredAction {
    fn plain(action: impl EditAction<World> + 'static) -> Self {
        Self {
            action: Box::new(action),
            spawned: None,
        }
    }
}

type Constructor =
    Box<dyn Fn(&World, &ron::Value) -> Result<RegisteredAction, String> + Send + Sync>;

struct Entry {
    usage: &'static str,
    construct: Constructor,
}

/// The action registry (a world resource).
#[derive(Default)]
pub struct ActionRegistry {
    entries: BTreeMap<String, Entry>,
}

impl ActionRegistry {
    pub fn new() -> Self {
        Self::default()
    }

    /// The registry with the standard scene-authoring actions.
    pub fn with_builtins() -> Self {
        let mut r = Self::new();
        r.register(
            "spawn_entity",
            "(name?: \"…\", parent?: \"idx@tick\") — new entity with Transform/Visibility; response carries its id",
            |world, params| {
                let report = SpawnReport::default();
                let action =
                    SpawnEntityAction::new(opt_str(params, "name"), opt_entity(world, params, "parent")?)
                        .with_report(report.clone());
                Ok(RegisteredAction {
                    action: Box::new(action),
                    spawned: Some(report),
                })
            },
        );
        r.register(
            "delete_entity",
            "(entity: \"idx@tick\") — despawn the subtree; undo restores it",
            |world, params| {
                let entity = req_entity(world, params, "entity")?;
                let parent = world.get::<Parent>(entity).map(|p| p.0);
                Ok(RegisteredAction::plain(DeleteEntityAction::new(
                    entity, parent,
                )))
            },
        );
        r.register(
            "reparent",
            "(entity: \"idx@tick\", parent?: \"idx@tick\") — attach under parent, or make root if absent",
            |world, params| {
                let entity = req_entity(world, params, "entity")?;
                let old = world.get::<Parent>(entity).map(|p| p.0);
                let new = opt_entity(world, params, "parent")?;
                Ok(RegisteredAction::plain(ReparentAction::new(entity, old, new)))
            },
        );
        r.register(
            "add_component",
            "(entity: \"idx@tick\", component: \"Name\") — add the default-valued component",
            |world, params| {
                Ok(RegisteredAction::plain(AddComponentAction {
                    entity: req_entity(world, params, "entity")?,
                    name: req_str(params, "component")?,
                }))
            },
        );
        r.register(
            "remove_component",
            "(entity: \"idx@tick\", component: \"Name\") — remove; undo restores the value",
            |world, params| {
                Ok(RegisteredAction::plain(RemoveComponentAction::new(
                    req_entity(world, params, "entity")?,
                    req_str(params, "component")?,
                )))
            },
        );
        r.register(
            "set_component",
            "(entity: \"idx@tick\", component: \"Transform\", data: (…)) — set the whole component from natural RON",
            |world, params| {
                let entity = req_entity(world, params, "entity")?;
                let component = req_str(params, "component")?;
                let data = field(params, "data").ok_or("missing 'data'")?;
                let serialized = SerializedComponent {
                    type_name: component,
                    data: natural_to_value(data)?,
                };
                Ok(RegisteredAction::plain(ImportComponentAction::new(
                    entity, serialized,
                )))
            },
        );
        r.register(
            "select",
            "(entities: [\"idx@tick\", …]) — set the selection (not recorded in history)",
            |world, params| {
                let entities = match field(params, "entities") {
                    Some(ron::Value::Seq(items)) => items
                        .iter()
                        .map(|v| match v {
                            ron::Value::String(s) => {
                                find_entity(world, s).ok_or_else(|| format!("unknown entity '{s}'"))
                            }
                            _ => Err("entity specs must be strings".into()),
                        })
                        .collect::<Result<Vec<Entity>, String>>()?,
                    _ => return Err("missing 'entities' list".into()),
                };
                Ok(RegisteredAction::plain(super::SelectAction::set(entities)))
            },
        );
        r
    }

    /// Register `name`. `usage` is the one-line parameter reference shown by
    /// introspection — keep it exact, remote agents build calls from it.
    pub fn register(
        &mut self,
        name: impl Into<String>,
        usage: &'static str,
        construct: impl Fn(&World, &ron::Value) -> Result<RegisteredAction, String>
        + Send
        + Sync
        + 'static,
    ) {
        self.entries.insert(
            name.into(),
            Entry {
                usage,
                construct: Box::new(construct),
            },
        );
    }

    /// Build the named action from wire parameters.
    pub fn construct(
        &self,
        name: &str,
        world: &World,
        params: &ron::Value,
    ) -> Result<RegisteredAction, String> {
        let entry = self
            .entries
            .get(name)
            .ok_or_else(|| format!("unknown action '{name}'"))?;
        (entry.construct)(world, params)
    }

    /// `(name, usage)` for every registered action, sorted by name.
    pub fn list(&self) -> Vec<(&str, &str)> {
        self.entries
            .iter()
            .map(|(name, e)| (name.as_str(), e.usage))
            .collect()
    }
}

// --- wire-parameter helpers -------------------------------------------------

/// Look up `key` in a natural-RON parameter map.
fn field<'a>(params: &'a ron::Value, key: &str) -> Option<&'a ron::Value> {
    match params {
        ron::Value::Map(map) => map
            .iter()
            .find(|(k, _)| **k == ron::Value::String(key.into()))
            .map(|(_, v)| v),
        _ => None,
    }
}

fn opt_str(params: &ron::Value, key: &str) -> Option<String> {
    match field(params, key) {
        Some(ron::Value::String(s)) => Some(s.clone()),
        _ => None,
    }
}

fn req_str(params: &ron::Value, key: &str) -> Result<String, String> {
    opt_str(params, key).ok_or_else(|| format!("missing '{key}'"))
}

/// An optional entity parameter: absent is `None`, present-but-unresolvable
/// is an error (a typo must not silently degrade into "no parent").
fn opt_entity(world: &World, params: &ron::Value, key: &str) -> Result<Option<Entity>, String> {
    match opt_str(params, key) {
        Some(spec) => find_entity(world, &spec)
            .map(Some)
            .ok_or_else(|| format!("unknown entity '{spec}'")),
        None => Ok(None),
    }
}

fn req_entity(world: &World, params: &ron::Value, key: &str) -> Result<Entity, String> {
    opt_entity(world, params, key)?.ok_or_else(|| format!("missing '{key}'"))
}
