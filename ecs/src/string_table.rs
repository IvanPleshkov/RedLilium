//! Interned string storage for Pod-compatible string references.
//!
//! Components that need string data store a [`StringId`] (a `u32` wrapper)
//! instead of `String`. The actual string content lives in a [`StringTable`]
//! stored as a World resource.

use std::collections::HashMap;

/// A Pod-compatible reference to an interned string.
///
/// Stores the index into a [`StringTable`]. Use `StringTable::get()` to
/// resolve the ID back to a string slice.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, bytemuck::Pod, bytemuck::Zeroable)]
#[repr(C)]
pub struct StringId(pub u32);

impl StringId {
    /// The empty string ID. Always maps to `""` in any `StringTable`.
    pub const EMPTY: Self = Self(0);
}

impl std::fmt::Display for StringId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "StringId({})", self.0)
    }
}

/// A deduplicating string storage.
///
/// Assigns each unique string a [`StringId`]. Index 0 is always the empty
/// string. Intended to be stored as a World resource so that Pod components
/// can reference strings by ID.
pub struct StringTable {
    strings: Vec<String>,
    lookup: HashMap<String, StringId>,
}

impl StringTable {
    /// Create a new table with the empty string pre-inserted at index 0.
    pub fn new() -> Self {
        let mut table = Self {
            strings: Vec::new(),
            lookup: HashMap::new(),
        };
        table.strings.push(String::new());
        table.lookup.insert(String::new(), StringId::EMPTY);
        table
    }

    /// Intern a string, returning its [`StringId`].
    ///
    /// If the string was already interned, returns the existing ID.
    pub fn intern(&mut self, s: &str) -> StringId {
        if let Some(&id) = self.lookup.get(s) {
            return id;
        }
        let id = StringId(self.strings.len() as u32);
        self.strings.push(s.to_string());
        self.lookup.insert(s.to_string(), id);
        id
    }

    /// Resolve a [`StringId`] to its string slice.
    ///
    /// # Panics
    ///
    /// Panics if the ID is out of range.
    pub fn get(&self, id: StringId) -> &str {
        &self.strings[id.0 as usize]
    }

    /// Try to resolve a [`StringId`], returning `None` if invalid.
    pub fn try_get(&self, id: StringId) -> Option<&str> {
        self.strings.get(id.0 as usize).map(|s| s.as_str())
    }

    /// Number of interned strings (including the empty string).
    pub fn len(&self) -> usize {
        self.strings.len()
    }

    /// Whether the table contains only the empty string.
    pub fn is_empty(&self) -> bool {
        self.strings.len() <= 1
    }
}

impl Default for StringTable {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_string_at_index_zero() {
        let table = StringTable::new();
        assert_eq!(table.get(StringId::EMPTY), "");
        assert_eq!(table.len(), 1);
    }

    #[test]
    fn intern_and_resolve() {
        let mut table = StringTable::new();
        let id = table.intern("hello");
        assert_eq!(table.get(id), "hello");
        assert_ne!(id, StringId::EMPTY);
    }

    #[test]
    fn deduplication() {
        let mut table = StringTable::new();
        let a = table.intern("world");
        let b = table.intern("world");
        assert_eq!(a, b);
        assert_eq!(table.len(), 2); // "" + "world"
    }

    #[test]
    fn empty_string_deduplicates() {
        let mut table = StringTable::new();
        let id = table.intern("");
        assert_eq!(id, StringId::EMPTY);
    }

    #[test]
    fn try_get_invalid() {
        let table = StringTable::new();
        assert!(table.try_get(StringId(999)).is_none());
    }

    #[test]
    fn multiple_strings() {
        let mut table = StringTable::new();
        let ids: Vec<_> = ["alpha", "beta", "gamma"]
            .iter()
            .map(|s| table.intern(s))
            .collect();

        assert_eq!(table.get(ids[0]), "alpha");
        assert_eq!(table.get(ids[1]), "beta");
        assert_eq!(table.get(ids[2]), "gamma");
        assert_eq!(table.len(), 4); // "" + 3
    }

    #[test]
    fn string_id_is_pod() {
        // Verify StringId can be used with bytemuck
        let id = StringId(42);
        let bytes = bytemuck::bytes_of(&id);
        assert_eq!(bytes.len(), 4);
        let restored: &StringId = bytemuck::from_bytes(bytes);
        assert_eq!(*restored, id);
    }
}
