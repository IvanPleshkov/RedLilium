//! Conversion between the engine's [`Value`] tree and "natural" RON values —
//! the human/model-facing form used by the remote protocol (`docs/REMOTE.md`).
//!
//! A serialized component internally is a tagged tree
//! (`Map([("translation", List([F32(1.0), …]))])`); the natural form is the
//! RON one would write by hand (`(translation: (1.0, …))` — parsed as a RON
//! map). The conversion is loss-tolerant by design: natural numbers come back
//! as `I64`/`F64` and the serde layer coerces them into the field's real type.
//!
//! Entity references have no natural RON shape, so they round-trip as a map
//! with the reserved `__entity` key holding the `"index@spawn_tick"` form.

use super::value::Value;

/// The reserved key marking an entity reference in natural form.
pub const ENTITY_KEY: &str = "__entity";

/// Convert an engine value tree to a natural RON value.
pub fn value_to_natural(value: &Value) -> ron::Value {
    match value {
        Value::Null => ron::Value::Unit,
        Value::Bool(v) => ron::Value::Bool(*v),
        Value::I64(v) => ron::Value::Number(ron::Number::from(*v)),
        Value::U64(v) => ron::Value::Number(ron::Number::from(*v as i64)),
        Value::F32(v) => ron::Value::Number(ron::Number::from(*v as f64)),
        Value::F64(v) => ron::Value::Number(ron::Number::from(*v)),
        Value::String(v) => ron::Value::String(v.clone()),
        Value::Bytes(v) => ron::Value::Seq(
            v.iter()
                .map(|b| ron::Value::Number(ron::Number::from(*b as i64)))
                .collect(),
        ),
        Value::List(items) => ron::Value::Seq(items.iter().map(value_to_natural).collect()),
        Value::Map(pairs) => {
            let mut map = ron::Map::new();
            for (k, v) in pairs {
                map.insert(ron::Value::String(k.clone()), value_to_natural(v));
            }
            ron::Value::Map(map)
        }
        Value::Entity { index, spawn_tick } => {
            let mut map = ron::Map::new();
            map.insert(
                ron::Value::String(ENTITY_KEY.to_owned()),
                ron::Value::String(format!("{index}@{spawn_tick}")),
            );
            ron::Value::Map(map)
        }
        other => {
            // Shared prefab-only variants (ArcValue/ArcRef) have no remote
            // form; surface them as an explicit marker rather than panicking.
            ron::Value::String(format!("<unsupported: {other:?}>"))
        }
    }
}

/// Convert a natural RON value back to an engine value tree.
pub fn natural_to_value(value: &ron::Value) -> Result<Value, String> {
    Ok(match value {
        ron::Value::Unit => Value::Null,
        ron::Value::Bool(v) => Value::Bool(*v),
        ron::Value::Char(c) => Value::String(c.to_string()),
        ron::Value::Number(n) => match n {
            ron::Number::Integer(i) => Value::I64(*i),
            other => Value::F64(other.into_f64()),
        },
        ron::Value::String(v) => Value::String(v.clone()),
        ron::Value::Option(opt) => match opt {
            Some(inner) => natural_to_value(inner)?,
            None => Value::Null,
        },
        ron::Value::Seq(items) => Value::List(
            items
                .iter()
                .map(natural_to_value)
                .collect::<Result<_, _>>()?,
        ),
        ron::Value::Map(map) => {
            // The reserved entity form: (__entity: "index@tick").
            if map.len() == 1
                && let Some((ron::Value::String(key), ron::Value::String(spec))) = map.iter().next()
                && key == ENTITY_KEY
            {
                let (index, tick) = parse_entity_spec(spec)?;
                return Ok(Value::Entity {
                    index,
                    spawn_tick: tick,
                });
            }
            let mut pairs = Vec::with_capacity(map.len());
            for (k, v) in map.iter() {
                let ron::Value::String(key) = k else {
                    return Err(format!("map key must be a string, got {k:?}"));
                };
                pairs.push((key.clone(), natural_to_value(v)?));
            }
            Value::Map(pairs)
        }
    })
}

/// Resolve an `"index@spawn_tick"` entity spec against a live world.
pub fn find_entity(world: &crate::World, spec: &str) -> Option<crate::Entity> {
    let (index, tick) = parse_entity_spec(spec).ok()?;
    world
        .iter_entities()
        .find(|e| e.index() == index && e.spawn_tick() == tick)
}

/// Parse the `"index@spawn_tick"` entity spec.
pub fn parse_entity_spec(spec: &str) -> Result<(u32, u64), String> {
    let (index, tick) = spec
        .split_once('@')
        .ok_or_else(|| format!("entity spec '{spec}' is not 'index@tick'"))?;
    Ok((
        index
            .parse()
            .map_err(|e| format!("entity index in '{spec}': {e}"))?,
        tick.parse()
            .map_err(|e| format!("entity tick in '{spec}': {e}"))?,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A component-shaped tree survives the natural round trip; numbers relax
    /// to the natural forms (F32 → F64, U64 → I64) which the serde layer
    /// coerces back into the field types on deserialization.
    #[test]
    fn natural_roundtrip() {
        let tree = Value::Map(vec![
            (
                "translation".into(),
                Value::List(vec![Value::F32(1.5), Value::F32(0.0), Value::F32(-2.0)]),
            ),
            ("visible".into(), Value::Bool(true)),
            ("name".into(), Value::String("Hero".into())),
            (
                "parent".into(),
                Value::Entity {
                    index: 12,
                    spawn_tick: 3,
                },
            ),
        ]);
        let natural = value_to_natural(&tree);
        let text = ron::to_string(&natural).unwrap();
        let parsed: ron::Value = ron::from_str(&text).unwrap();
        let back = natural_to_value(&parsed).unwrap();

        let Value::Map(pairs) = back else {
            panic!("expected map")
        };
        // ron::Map is ordered by key; deserialization reads fields by name,
        // so order is irrelevant — look entries up by key here too.
        let get = |name: &str| pairs.iter().find(|(k, _)| k == name).unwrap().1.clone();
        assert_eq!(
            get("translation"),
            Value::List(vec![Value::F64(1.5), Value::F64(0.0), Value::F64(-2.0)])
        );
        assert_eq!(get("visible"), Value::Bool(true));
        assert_eq!(get("name"), Value::String("Hero".into()));
        assert_eq!(
            get("parent"),
            Value::Entity {
                index: 12,
                spawn_tick: 3
            }
        );
    }

    /// Hand-written natural RON (what a remote client sends) converts into a
    /// value tree the component deserializer accepts.
    #[test]
    fn hand_written_natural_parses() {
        let parsed: ron::Value =
            ron::from_str("(translation: (5.0, 0.7, -2.0), scale: (1, 1, 1))").unwrap();
        let tree = natural_to_value(&parsed).unwrap();
        let Value::Map(pairs) = tree else {
            panic!("expected map")
        };
        let get = |name: &str| pairs.iter().find(|(k, _)| k == name).unwrap().1.clone();
        assert!(matches!(get("translation"), Value::List(_)));
        assert_eq!(
            get("scale"),
            Value::List(vec![Value::I64(1), Value::I64(1), Value::I64(1)])
        );
    }
}
