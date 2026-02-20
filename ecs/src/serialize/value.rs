//! Format-agnostic intermediate value representation.
//!
//! The [`Value`] enum captures component field data including ECS-specific
//! constructs like entity references and Arc deduplication markers.
//!
//! Use [`to_value`] and [`from_value`] to convert between arbitrary serde
//! types and `Value`.

use std::fmt;

use serde::de::{self, DeserializeSeed, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Serialize};

use super::error::{DeserializeError, SerializeError};

/// Format-agnostic value representation for component fields.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Value {
    Null,
    Bool(bool),
    I64(i64),
    U64(u64),
    F32(f32),
    F64(f64),
    String(String),
    Bytes(Vec<u8>),
    List(Vec<Value>),
    Map(Vec<(String, Value)>),
    /// An entity reference (serialized for later remapping).
    Entity {
        index: u32,
        spawn_tick: u64,
    },
    /// First occurrence of a shared Arc value (inline data + dedup ID).
    ArcValue {
        id: u32,
        inner: Box<Value>,
    },
    /// Back-reference to a previously serialized Arc (dedup ID).
    ArcRef(u32),
}

// ---------------------------------------------------------------------------
// to_value: T -> Value  (via custom serde::Serializer)
// ---------------------------------------------------------------------------

/// Convert any `T: Serialize` into a [`Value`].
pub fn to_value<T: Serialize>(value: &T) -> Result<Value, SerializeError> {
    value
        .serialize(ValueSerializer)
        .map_err(|e| SerializeError::FieldError {
            field: String::new(),
            message: e.to_string(),
        })
}

/// Convert a [`Value`] back into any `T: DeserializeOwned`.
pub fn from_value<T: de::DeserializeOwned>(value: Value) -> Result<T, DeserializeError> {
    T::deserialize(ValueDeserializer(value)).map_err(|e| DeserializeError::FormatError(e.0))
}

// ---------------------------------------------------------------------------
// ValueSerializer
// ---------------------------------------------------------------------------

struct ValueSerializer;

#[derive(Debug)]
struct ValueError(String);

impl fmt::Display for ValueError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::error::Error for ValueError {}

impl serde::ser::Error for ValueError {
    fn custom<T: fmt::Display>(msg: T) -> Self {
        ValueError(msg.to_string())
    }
}

impl serde::Serializer for ValueSerializer {
    type Ok = Value;
    type Error = ValueError;
    type SerializeSeq = ValueSerializeSeq;
    type SerializeTuple = ValueSerializeSeq;
    type SerializeTupleStruct = ValueSerializeSeq;
    type SerializeTupleVariant = ValueSerializeTupleVariant;
    type SerializeMap = ValueSerializeMap;
    type SerializeStruct = ValueSerializeMap;
    type SerializeStructVariant = ValueSerializeStructVariant;

    fn serialize_bool(self, v: bool) -> Result<Value, ValueError> {
        Ok(Value::Bool(v))
    }

    fn serialize_i8(self, v: i8) -> Result<Value, ValueError> {
        Ok(Value::I64(v as i64))
    }
    fn serialize_i16(self, v: i16) -> Result<Value, ValueError> {
        Ok(Value::I64(v as i64))
    }
    fn serialize_i32(self, v: i32) -> Result<Value, ValueError> {
        Ok(Value::I64(v as i64))
    }
    fn serialize_i64(self, v: i64) -> Result<Value, ValueError> {
        Ok(Value::I64(v))
    }

    fn serialize_u8(self, v: u8) -> Result<Value, ValueError> {
        Ok(Value::U64(v as u64))
    }
    fn serialize_u16(self, v: u16) -> Result<Value, ValueError> {
        Ok(Value::U64(v as u64))
    }
    fn serialize_u32(self, v: u32) -> Result<Value, ValueError> {
        Ok(Value::U64(v as u64))
    }
    fn serialize_u64(self, v: u64) -> Result<Value, ValueError> {
        Ok(Value::U64(v))
    }

    fn serialize_f32(self, v: f32) -> Result<Value, ValueError> {
        Ok(Value::F32(v))
    }
    fn serialize_f64(self, v: f64) -> Result<Value, ValueError> {
        Ok(Value::F64(v))
    }

    fn serialize_char(self, v: char) -> Result<Value, ValueError> {
        Ok(Value::String(v.to_string()))
    }
    fn serialize_str(self, v: &str) -> Result<Value, ValueError> {
        Ok(Value::String(v.to_owned()))
    }
    fn serialize_bytes(self, v: &[u8]) -> Result<Value, ValueError> {
        Ok(Value::Bytes(v.to_vec()))
    }

    fn serialize_none(self) -> Result<Value, ValueError> {
        Ok(Value::Null)
    }
    fn serialize_some<T: ?Sized + Serialize>(self, value: &T) -> Result<Value, ValueError> {
        value.serialize(self)
    }

    fn serialize_unit(self) -> Result<Value, ValueError> {
        Ok(Value::Null)
    }
    fn serialize_unit_struct(self, _name: &'static str) -> Result<Value, ValueError> {
        Ok(Value::Null)
    }
    fn serialize_unit_variant(
        self,
        _name: &'static str,
        _variant_index: u32,
        variant: &'static str,
    ) -> Result<Value, ValueError> {
        Ok(Value::String(variant.to_owned()))
    }

    fn serialize_newtype_struct<T: ?Sized + Serialize>(
        self,
        _name: &'static str,
        value: &T,
    ) -> Result<Value, ValueError> {
        value.serialize(self)
    }

    fn serialize_newtype_variant<T: ?Sized + Serialize>(
        self,
        _name: &'static str,
        _variant_index: u32,
        variant: &'static str,
        value: &T,
    ) -> Result<Value, ValueError> {
        let inner = value.serialize(ValueSerializer)?;
        Ok(Value::Map(vec![(variant.to_owned(), inner)]))
    }

    fn serialize_seq(self, len: Option<usize>) -> Result<ValueSerializeSeq, ValueError> {
        Ok(ValueSerializeSeq {
            items: Vec::with_capacity(len.unwrap_or(0)),
        })
    }

    fn serialize_tuple(self, len: usize) -> Result<ValueSerializeSeq, ValueError> {
        self.serialize_seq(Some(len))
    }

    fn serialize_tuple_struct(
        self,
        _name: &'static str,
        len: usize,
    ) -> Result<ValueSerializeSeq, ValueError> {
        self.serialize_seq(Some(len))
    }

    fn serialize_tuple_variant(
        self,
        _name: &'static str,
        _variant_index: u32,
        variant: &'static str,
        len: usize,
    ) -> Result<ValueSerializeTupleVariant, ValueError> {
        Ok(ValueSerializeTupleVariant {
            variant: variant.to_owned(),
            items: Vec::with_capacity(len),
        })
    }

    fn serialize_map(self, len: Option<usize>) -> Result<ValueSerializeMap, ValueError> {
        Ok(ValueSerializeMap {
            entries: Vec::with_capacity(len.unwrap_or(0)),
            current_key: None,
        })
    }

    fn serialize_struct(
        self,
        _name: &'static str,
        len: usize,
    ) -> Result<ValueSerializeMap, ValueError> {
        self.serialize_map(Some(len))
    }

    fn serialize_struct_variant(
        self,
        _name: &'static str,
        _variant_index: u32,
        variant: &'static str,
        len: usize,
    ) -> Result<ValueSerializeStructVariant, ValueError> {
        Ok(ValueSerializeStructVariant {
            variant: variant.to_owned(),
            entries: Vec::with_capacity(len),
        })
    }
}

struct ValueSerializeSeq {
    items: Vec<Value>,
}

impl serde::ser::SerializeSeq for ValueSerializeSeq {
    type Ok = Value;
    type Error = ValueError;

    fn serialize_element<T: ?Sized + Serialize>(&mut self, value: &T) -> Result<(), ValueError> {
        self.items.push(value.serialize(ValueSerializer)?);
        Ok(())
    }

    fn end(self) -> Result<Value, ValueError> {
        Ok(Value::List(self.items))
    }
}

impl serde::ser::SerializeTuple for ValueSerializeSeq {
    type Ok = Value;
    type Error = ValueError;

    fn serialize_element<T: ?Sized + Serialize>(&mut self, value: &T) -> Result<(), ValueError> {
        serde::ser::SerializeSeq::serialize_element(self, value)
    }

    fn end(self) -> Result<Value, ValueError> {
        serde::ser::SerializeSeq::end(self)
    }
}

impl serde::ser::SerializeTupleStruct for ValueSerializeSeq {
    type Ok = Value;
    type Error = ValueError;

    fn serialize_field<T: ?Sized + Serialize>(&mut self, value: &T) -> Result<(), ValueError> {
        serde::ser::SerializeSeq::serialize_element(self, value)
    }

    fn end(self) -> Result<Value, ValueError> {
        serde::ser::SerializeSeq::end(self)
    }
}

struct ValueSerializeTupleVariant {
    variant: String,
    items: Vec<Value>,
}

impl serde::ser::SerializeTupleVariant for ValueSerializeTupleVariant {
    type Ok = Value;
    type Error = ValueError;

    fn serialize_field<T: ?Sized + Serialize>(&mut self, value: &T) -> Result<(), ValueError> {
        self.items.push(value.serialize(ValueSerializer)?);
        Ok(())
    }

    fn end(self) -> Result<Value, ValueError> {
        Ok(Value::Map(vec![(self.variant, Value::List(self.items))]))
    }
}

struct ValueSerializeMap {
    entries: Vec<(String, Value)>,
    current_key: Option<String>,
}

impl serde::ser::SerializeMap for ValueSerializeMap {
    type Ok = Value;
    type Error = ValueError;

    fn serialize_key<T: ?Sized + Serialize>(&mut self, key: &T) -> Result<(), ValueError> {
        let key_val = key.serialize(ValueSerializer)?;
        let key_str = match key_val {
            Value::String(s) => s,
            other => format!("{other:?}"),
        };
        self.current_key = Some(key_str);
        Ok(())
    }

    fn serialize_value<T: ?Sized + Serialize>(&mut self, value: &T) -> Result<(), ValueError> {
        let key = self
            .current_key
            .take()
            .ok_or_else(|| ValueError("serialize_value called before serialize_key".into()))?;
        self.entries.push((key, value.serialize(ValueSerializer)?));
        Ok(())
    }

    fn end(self) -> Result<Value, ValueError> {
        Ok(Value::Map(self.entries))
    }
}

impl serde::ser::SerializeStruct for ValueSerializeMap {
    type Ok = Value;
    type Error = ValueError;

    fn serialize_field<T: ?Sized + Serialize>(
        &mut self,
        key: &'static str,
        value: &T,
    ) -> Result<(), ValueError> {
        self.entries
            .push((key.to_owned(), value.serialize(ValueSerializer)?));
        Ok(())
    }

    fn end(self) -> Result<Value, ValueError> {
        Ok(Value::Map(self.entries))
    }
}

struct ValueSerializeStructVariant {
    variant: String,
    entries: Vec<(String, Value)>,
}

impl serde::ser::SerializeStructVariant for ValueSerializeStructVariant {
    type Ok = Value;
    type Error = ValueError;

    fn serialize_field<T: ?Sized + Serialize>(
        &mut self,
        key: &'static str,
        value: &T,
    ) -> Result<(), ValueError> {
        self.entries
            .push((key.to_owned(), value.serialize(ValueSerializer)?));
        Ok(())
    }

    fn end(self) -> Result<Value, ValueError> {
        Ok(Value::Map(vec![(self.variant, Value::Map(self.entries))]))
    }
}

// ---------------------------------------------------------------------------
// ValueDeserializer: Value -> T
// ---------------------------------------------------------------------------

struct ValueDeserializer(Value);

impl<'de> serde::Deserializer<'de> for ValueDeserializer {
    type Error = ValueError;

    fn deserialize_any<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, ValueError> {
        match self.0 {
            Value::Null => visitor.visit_unit(),
            Value::Bool(v) => visitor.visit_bool(v),
            Value::I64(v) => visitor.visit_i64(v),
            Value::U64(v) => visitor.visit_u64(v),
            Value::F32(v) => visitor.visit_f32(v),
            Value::F64(v) => visitor.visit_f64(v),
            Value::String(v) => visitor.visit_string(v),
            Value::Bytes(v) => visitor.visit_byte_buf(v),
            Value::List(v) => visitor.visit_seq(ValueSeqAccess {
                iter: v.into_iter(),
            }),
            Value::Map(v) => visitor.visit_map(ValueMapAccess {
                iter: v.into_iter(),
                pending_value: None,
            }),
            Value::Entity { .. } | Value::ArcValue { .. } | Value::ArcRef(_) => Err(ValueError(
                "ECS-specific Value variants must be handled by the context, not serde".into(),
            )),
        }
    }

    fn deserialize_bool<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, ValueError> {
        match self.0 {
            Value::Bool(v) => visitor.visit_bool(v),
            _ => self.deserialize_any(visitor),
        }
    }

    fn deserialize_i8<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, ValueError> {
        self.deserialize_i64(visitor)
    }
    fn deserialize_i16<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, ValueError> {
        self.deserialize_i64(visitor)
    }
    fn deserialize_i32<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, ValueError> {
        self.deserialize_i64(visitor)
    }
    fn deserialize_i64<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, ValueError> {
        match self.0 {
            Value::I64(v) => visitor.visit_i64(v),
            Value::U64(v) => visitor.visit_i64(v as i64),
            _ => self.deserialize_any(visitor),
        }
    }

    fn deserialize_u8<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, ValueError> {
        self.deserialize_u64(visitor)
    }
    fn deserialize_u16<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, ValueError> {
        self.deserialize_u64(visitor)
    }
    fn deserialize_u32<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, ValueError> {
        self.deserialize_u64(visitor)
    }
    fn deserialize_u64<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, ValueError> {
        match self.0 {
            Value::U64(v) => visitor.visit_u64(v),
            Value::I64(v) => visitor.visit_u64(v as u64),
            _ => self.deserialize_any(visitor),
        }
    }

    fn deserialize_f32<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, ValueError> {
        match self.0 {
            Value::F32(v) => visitor.visit_f32(v),
            Value::F64(v) => visitor.visit_f32(v as f32),
            _ => self.deserialize_any(visitor),
        }
    }
    fn deserialize_f64<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, ValueError> {
        match self.0 {
            Value::F64(v) => visitor.visit_f64(v),
            Value::F32(v) => visitor.visit_f64(v as f64),
            _ => self.deserialize_any(visitor),
        }
    }

    fn deserialize_char<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, ValueError> {
        self.deserialize_str(visitor)
    }
    fn deserialize_str<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, ValueError> {
        self.deserialize_string(visitor)
    }
    fn deserialize_string<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, ValueError> {
        match self.0 {
            Value::String(v) => visitor.visit_string(v),
            _ => self.deserialize_any(visitor),
        }
    }

    fn deserialize_bytes<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, ValueError> {
        self.deserialize_byte_buf(visitor)
    }
    fn deserialize_byte_buf<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, ValueError> {
        match self.0 {
            Value::Bytes(v) => visitor.visit_byte_buf(v),
            _ => self.deserialize_any(visitor),
        }
    }

    fn deserialize_option<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, ValueError> {
        match self.0 {
            Value::Null => visitor.visit_none(),
            other => visitor.visit_some(ValueDeserializer(other)),
        }
    }

    fn deserialize_unit<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, ValueError> {
        visitor.visit_unit()
    }
    fn deserialize_unit_struct<V: Visitor<'de>>(
        self,
        _name: &'static str,
        visitor: V,
    ) -> Result<V::Value, ValueError> {
        visitor.visit_unit()
    }

    fn deserialize_newtype_struct<V: Visitor<'de>>(
        self,
        _name: &'static str,
        visitor: V,
    ) -> Result<V::Value, ValueError> {
        visitor.visit_newtype_struct(self)
    }

    fn deserialize_seq<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, ValueError> {
        match self.0 {
            Value::List(v) => visitor.visit_seq(ValueSeqAccess {
                iter: v.into_iter(),
            }),
            _ => Err(ValueError("expected list".into())),
        }
    }

    fn deserialize_tuple<V: Visitor<'de>>(
        self,
        _len: usize,
        visitor: V,
    ) -> Result<V::Value, ValueError> {
        self.deserialize_seq(visitor)
    }
    fn deserialize_tuple_struct<V: Visitor<'de>>(
        self,
        _name: &'static str,
        _len: usize,
        visitor: V,
    ) -> Result<V::Value, ValueError> {
        self.deserialize_seq(visitor)
    }

    fn deserialize_map<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, ValueError> {
        match self.0 {
            Value::Map(v) => visitor.visit_map(ValueMapAccess {
                iter: v.into_iter(),
                pending_value: None,
            }),
            _ => Err(ValueError("expected map".into())),
        }
    }

    fn deserialize_struct<V: Visitor<'de>>(
        self,
        _name: &'static str,
        _fields: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, ValueError> {
        self.deserialize_map(visitor)
    }

    fn deserialize_enum<V: Visitor<'de>>(
        self,
        _name: &'static str,
        _variants: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, ValueError> {
        match self.0 {
            Value::String(s) => visitor.visit_enum(ValueEnumAccess::Unit(s)),
            Value::Map(entries) => {
                if entries.len() != 1 {
                    return Err(ValueError("enum map must have exactly one entry".into()));
                }
                let (variant, value) = entries.into_iter().next().unwrap();
                visitor.visit_enum(ValueEnumAccess::Newtype(variant, value))
            }
            _ => Err(ValueError("expected string or map for enum".into())),
        }
    }

    fn deserialize_identifier<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, ValueError> {
        self.deserialize_string(visitor)
    }

    fn deserialize_ignored_any<V: Visitor<'de>>(self, visitor: V) -> Result<V::Value, ValueError> {
        visitor.visit_unit()
    }
}

impl de::Error for ValueError {
    fn custom<T: fmt::Display>(msg: T) -> Self {
        ValueError(msg.to_string())
    }
}

// --- SeqAccess ---

struct ValueSeqAccess {
    iter: std::vec::IntoIter<Value>,
}

impl<'de> SeqAccess<'de> for ValueSeqAccess {
    type Error = ValueError;

    fn next_element_seed<T: DeserializeSeed<'de>>(
        &mut self,
        seed: T,
    ) -> Result<Option<T::Value>, ValueError> {
        match self.iter.next() {
            Some(v) => seed.deserialize(ValueDeserializer(v)).map(Some),
            None => Ok(None),
        }
    }
}

// --- MapAccess ---

struct ValueMapAccess {
    iter: std::vec::IntoIter<(String, Value)>,
    pending_value: Option<Value>,
}

impl<'de> MapAccess<'de> for ValueMapAccess {
    type Error = ValueError;

    fn next_key_seed<K: DeserializeSeed<'de>>(
        &mut self,
        seed: K,
    ) -> Result<Option<K::Value>, ValueError> {
        match self.iter.next() {
            Some((k, v)) => {
                self.pending_value = Some(v);
                seed.deserialize(ValueDeserializer(Value::String(k)))
                    .map(Some)
            }
            None => Ok(None),
        }
    }

    fn next_value_seed<V: DeserializeSeed<'de>>(
        &mut self,
        seed: V,
    ) -> Result<V::Value, ValueError> {
        let value = self
            .pending_value
            .take()
            .ok_or_else(|| ValueError("next_value_seed called before next_key_seed".into()))?;
        seed.deserialize(ValueDeserializer(value))
    }
}

// --- EnumAccess ---

enum ValueEnumAccess {
    Unit(String),
    Newtype(String, Value),
}

impl<'de> de::EnumAccess<'de> for ValueEnumAccess {
    type Error = ValueError;
    type Variant = ValueVariantAccess;

    fn variant_seed<V: DeserializeSeed<'de>>(
        self,
        seed: V,
    ) -> Result<(V::Value, Self::Variant), ValueError> {
        match self {
            ValueEnumAccess::Unit(s) => {
                let val = seed.deserialize(ValueDeserializer(Value::String(s)))?;
                Ok((val, ValueVariantAccess::Unit))
            }
            ValueEnumAccess::Newtype(s, value) => {
                let val = seed.deserialize(ValueDeserializer(Value::String(s)))?;
                Ok((val, ValueVariantAccess::Newtype(value)))
            }
        }
    }
}

enum ValueVariantAccess {
    Unit,
    Newtype(Value),
}

impl<'de> de::VariantAccess<'de> for ValueVariantAccess {
    type Error = ValueError;

    fn unit_variant(self) -> Result<(), ValueError> {
        Ok(())
    }

    fn newtype_variant_seed<T: DeserializeSeed<'de>>(
        self,
        seed: T,
    ) -> Result<T::Value, ValueError> {
        match self {
            ValueVariantAccess::Newtype(v) => seed.deserialize(ValueDeserializer(v)),
            ValueVariantAccess::Unit => Err(ValueError("expected newtype variant".into())),
        }
    }

    fn tuple_variant<V: Visitor<'de>>(
        self,
        _len: usize,
        visitor: V,
    ) -> Result<V::Value, ValueError> {
        match self {
            ValueVariantAccess::Newtype(v) => {
                serde::Deserializer::deserialize_seq(ValueDeserializer(v), visitor)
            }
            ValueVariantAccess::Unit => Err(ValueError("expected tuple variant".into())),
        }
    }

    fn struct_variant<V: Visitor<'de>>(
        self,
        _fields: &'static [&'static str],
        visitor: V,
    ) -> Result<V::Value, ValueError> {
        match self {
            ValueVariantAccess::Newtype(v) => {
                serde::Deserializer::deserialize_map(ValueDeserializer(v), visitor)
            }
            ValueVariantAccess::Unit => Err(ValueError("expected struct variant".into())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_primitives() {
        assert!(from_value::<bool>(to_value(&true).unwrap()).unwrap());
        assert_eq!(from_value::<i32>(to_value(&42i32).unwrap()).unwrap(), 42);
        assert_eq!(from_value::<u64>(to_value(&99u64).unwrap()).unwrap(), 99);
        assert_eq!(from_value::<f32>(to_value(&1.5f32).unwrap()).unwrap(), 1.5);
        assert_eq!(from_value::<f64>(to_value(&1.5f64).unwrap()).unwrap(), 1.5);
        assert_eq!(
            from_value::<String>(to_value(&"hello").unwrap()).unwrap(),
            "hello"
        );
    }

    #[test]
    fn roundtrip_vec() {
        let v = vec![1u32, 2, 3];
        let val = to_value(&v).unwrap();
        let restored: Vec<u32> = from_value(val).unwrap();
        assert_eq!(restored, vec![1, 2, 3]);
    }

    #[test]
    fn roundtrip_option() {
        let some: Option<i32> = Some(42);
        let none: Option<i32> = None;
        assert_eq!(
            from_value::<Option<i32>>(to_value(&some).unwrap()).unwrap(),
            Some(42)
        );
        assert_eq!(
            from_value::<Option<i32>>(to_value(&none).unwrap()).unwrap(),
            None
        );
    }

    #[test]
    fn roundtrip_struct() {
        #[derive(Debug, PartialEq, Serialize, Deserialize)]
        struct Point {
            x: f32,
            y: f32,
        }
        let p = Point { x: 1.0, y: 2.0 };
        let val = to_value(&p).unwrap();
        let restored: Point = from_value(val).unwrap();
        assert_eq!(restored, p);
    }

    #[test]
    fn roundtrip_enum() {
        #[derive(Debug, PartialEq, Serialize, Deserialize)]
        enum Color {
            Red,
            Green,
            Custom { r: u8, g: u8, b: u8 },
        }
        assert_eq!(
            from_value::<Color>(to_value(&Color::Red).unwrap()).unwrap(),
            Color::Red
        );
        assert_eq!(
            from_value::<Color>(to_value(&Color::Custom { r: 1, g: 2, b: 3 }).unwrap()).unwrap(),
            Color::Custom { r: 1, g: 2, b: 3 }
        );
    }

    #[test]
    fn roundtrip_nested() {
        #[derive(Debug, PartialEq, Serialize, Deserialize)]
        struct Inner {
            value: f64,
        }
        #[derive(Debug, PartialEq, Serialize, Deserialize)]
        struct Outer {
            name: String,
            inner: Inner,
            items: Vec<i32>,
        }
        let o = Outer {
            name: "test".to_string(),
            inner: Inner { value: 99.9 },
            items: vec![1, 2, 3],
        };
        let val = to_value(&o).unwrap();
        let restored: Outer = from_value(val).unwrap();
        assert_eq!(restored, o);
    }
}
