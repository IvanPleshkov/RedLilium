use std::any::TypeId;
use std::collections::HashSet;

/// Describes the component and resource access a system requires.
///
/// Used by the scheduler to detect conflicts between systems
/// and determine which systems can run in parallel.
///
/// Two accesses conflict if one writes a type that the other reads or writes.
#[derive(Debug, Clone, Default)]
pub struct Access {
    /// Component types read by this system.
    reads: HashSet<TypeId>,
    /// Component types written by this system.
    writes: HashSet<TypeId>,
    /// Resource types read by this system.
    resource_reads: HashSet<TypeId>,
    /// Resource types written by this system.
    resource_writes: HashSet<TypeId>,
}

impl Access {
    /// Creates an empty access descriptor.
    pub fn new() -> Self {
        Self::default()
    }

    /// Declares read access to component type T.
    pub fn add_read<T: 'static>(&mut self) {
        self.reads.insert(TypeId::of::<T>());
    }

    /// Declares write access to component type T.
    pub fn add_write<T: 'static>(&mut self) {
        self.writes.insert(TypeId::of::<T>());
    }

    /// Declares read access to resource type T.
    pub fn add_resource_read<T: 'static>(&mut self) {
        self.resource_reads.insert(TypeId::of::<T>());
    }

    /// Declares write access to resource type T.
    pub fn add_resource_write<T: 'static>(&mut self) {
        self.resource_writes.insert(TypeId::of::<T>());
    }

    /// Returns whether this access conflicts with another.
    ///
    /// Two accesses conflict if one writes a type that the other reads or writes.
    pub fn conflicts_with(&self, other: &Access) -> bool {
        // Check component conflicts
        if self
            .writes
            .iter()
            .any(|t| other.reads.contains(t) || other.writes.contains(t))
        {
            return true;
        }
        if other
            .writes
            .iter()
            .any(|t| self.reads.contains(t) || self.writes.contains(t))
        {
            return true;
        }
        // Check resource conflicts
        if self
            .resource_writes
            .iter()
            .any(|t| other.resource_reads.contains(t) || other.resource_writes.contains(t))
        {
            return true;
        }
        if other
            .resource_writes
            .iter()
            .any(|t| self.resource_reads.contains(t) || self.resource_writes.contains(t))
        {
            return true;
        }
        false
    }

    /// Returns whether this access is read-only (no writes at all).
    pub fn is_read_only(&self) -> bool {
        self.writes.is_empty() && self.resource_writes.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct CompA;
    struct CompB;
    struct ResA;

    #[test]
    fn no_conflict_disjoint_reads() {
        let mut a = Access::new();
        a.add_read::<CompA>();
        let mut b = Access::new();
        b.add_read::<CompB>();
        assert!(!a.conflicts_with(&b));
    }

    #[test]
    fn no_conflict_same_reads() {
        let mut a = Access::new();
        a.add_read::<CompA>();
        let mut b = Access::new();
        b.add_read::<CompA>();
        assert!(!a.conflicts_with(&b));
    }

    #[test]
    fn conflict_read_write_same_type() {
        let mut a = Access::new();
        a.add_read::<CompA>();
        let mut b = Access::new();
        b.add_write::<CompA>();
        assert!(a.conflicts_with(&b));
    }

    #[test]
    fn conflict_write_write_same_type() {
        let mut a = Access::new();
        a.add_write::<CompA>();
        let mut b = Access::new();
        b.add_write::<CompA>();
        assert!(a.conflicts_with(&b));
    }

    #[test]
    fn no_conflict_different_write_types() {
        let mut a = Access::new();
        a.add_write::<CompA>();
        let mut b = Access::new();
        b.add_write::<CompB>();
        assert!(!a.conflicts_with(&b));
    }

    #[test]
    fn resource_conflict() {
        let mut a = Access::new();
        a.add_resource_read::<ResA>();
        let mut b = Access::new();
        b.add_resource_write::<ResA>();
        assert!(a.conflicts_with(&b));
    }

    #[test]
    fn no_resource_conflict_both_read() {
        let mut a = Access::new();
        a.add_resource_read::<ResA>();
        let mut b = Access::new();
        b.add_resource_read::<ResA>();
        assert!(!a.conflicts_with(&b));
    }

    #[test]
    fn is_read_only() {
        let mut a = Access::new();
        a.add_read::<CompA>();
        a.add_resource_read::<ResA>();
        assert!(a.is_read_only());

        a.add_write::<CompB>();
        assert!(!a.is_read_only());
    }

    #[test]
    fn conflict_is_symmetric() {
        let mut a = Access::new();
        a.add_read::<CompA>();
        let mut b = Access::new();
        b.add_write::<CompA>();
        assert!(a.conflicts_with(&b));
        assert!(b.conflicts_with(&a));
    }

    #[test]
    fn empty_accesses_no_conflict() {
        let a = Access::new();
        let b = Access::new();
        assert!(!a.conflicts_with(&b));
    }
}
