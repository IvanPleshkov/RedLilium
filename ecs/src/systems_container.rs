use std::any::TypeId;
use std::collections::HashMap;
use std::fmt;

use crate::system::{DynSystem, System};

/// A registered system with its metadata.
pub(crate) struct StoredSystem {
    /// The system instance, type-erased.
    pub system: Box<dyn DynSystem>,
    /// Human-readable type name for debug/error messages.
    pub type_name: &'static str,
}

/// An explicit ordering edge between two systems.
///
/// `from` must complete before `to` can start.
pub struct Edge {
    /// TypeId of the system that runs first.
    pub from: TypeId,
    /// TypeId of the system that runs after.
    pub to: TypeId,
}

impl Edge {
    /// Creates an edge: `Before` must complete before `After` starts.
    pub fn new<Before: 'static, After: 'static>() -> Self {
        Self {
            from: TypeId::of::<Before>(),
            to: TypeId::of::<After>(),
        }
    }
}

/// Error returned when adding edges would create a dependency cycle.
#[derive(Debug)]
pub struct CycleError {
    /// Type names of systems involved in the cycle.
    pub involved: Vec<String>,
}

impl fmt::Display for CycleError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "Dependency cycle detected among: {}",
            self.involved.join(", ")
        )
    }
}

impl std::error::Error for CycleError {}

/// Container for registered systems with explicit dependency ordering.
///
/// Systems are registered with [`add()`](SystemsContainer::add). Ordering
/// constraints are added with [`add_edge()`](SystemsContainer::add_edge) or
/// [`add_edges()`](SystemsContainer::add_edges). Cycle detection uses Kahn's
/// topological sort algorithm.
///
/// The runner uses this container (immutably) to determine execution order
/// and access system instances.
///
/// # Example
///
/// ```ignore
/// let mut container = SystemsContainer::new();
/// container.add(UpdateGlobalTransforms);
/// container.add(UpdateCameraMatrices);
/// container.add_edge::<UpdateGlobalTransforms, UpdateCameraMatrices>().unwrap();
/// ```
pub struct SystemsContainer {
    /// Registered systems in insertion order.
    systems: Vec<StoredSystem>,
    /// Forward adjacency list: edges[i] = indices of systems that depend on i.
    edges: Vec<Vec<usize>>,
    /// Cached in-degree per system (number of dependencies).
    in_degrees: Vec<usize>,
    /// Map from TypeId to index in `systems`.
    id_to_idx: HashMap<TypeId, usize>,
}

impl SystemsContainer {
    /// Creates a new empty systems container.
    pub fn new() -> Self {
        Self {
            systems: Vec::new(),
            edges: Vec::new(),
            in_degrees: Vec::new(),
            id_to_idx: HashMap::new(),
        }
    }

    /// Registers a system.
    ///
    /// # Panics
    ///
    /// Panics if a system of the same type is already registered.
    pub fn add<S: System>(&mut self, system: S) {
        let type_id = TypeId::of::<S>();
        let type_name = std::any::type_name::<S>();

        assert!(
            !self.id_to_idx.contains_key(&type_id),
            "System `{type_name}` is already registered"
        );

        let idx = self.systems.len();
        self.id_to_idx.insert(type_id, idx);
        self.systems.push(StoredSystem {
            system: Box::new(system),
            type_name,
        });
        self.edges.push(Vec::new());
        self.in_degrees.push(0);
    }

    /// Adds a single ordering edge: `Before` must complete before `After` starts.
    ///
    /// Returns `Err(CycleError)` if the edge would create a cycle.
    ///
    /// # Panics
    ///
    /// Panics if either system type is not registered.
    pub fn add_edge<Before: 'static, After: 'static>(&mut self) -> Result<(), CycleError> {
        self.add_edges(&[Edge::new::<Before, After>()])
    }

    /// Adds multiple ordering edges in a batch.
    ///
    /// All edges are validated together — if any would create a cycle,
    /// none are applied and `Err(CycleError)` is returned.
    ///
    /// # Panics
    ///
    /// Panics if any system TypeId in the edges is not registered.
    pub fn add_edges(&mut self, new_edges: &[Edge]) -> Result<(), CycleError> {
        if new_edges.is_empty() {
            return Ok(());
        }

        // Resolve edges to indices and validate registration
        let resolved: Vec<(usize, usize)> = new_edges
            .iter()
            .map(|edge| {
                let from = *self.id_to_idx.get(&edge.from).unwrap_or_else(|| {
                    panic!("System with TypeId {:?} is not registered", edge.from)
                });
                let to = *self.id_to_idx.get(&edge.to).unwrap_or_else(|| {
                    panic!("System with TypeId {:?} is not registered", edge.to)
                });
                (from, to)
            })
            .collect();

        // Clone current state for validation
        let mut test_edges = self.edges.clone();
        let mut test_in_degrees = self.in_degrees.clone();

        for &(from, to) in &resolved {
            // Skip duplicate edges
            if !test_edges[from].contains(&to) {
                test_edges[from].push(to);
                test_in_degrees[to] += 1;
            }
        }

        // Validate with Kahn's topological sort
        if let Some(cycle_indices) = find_cycle(&test_edges, &test_in_degrees) {
            let involved = cycle_indices
                .iter()
                .map(|&idx| self.systems[idx].type_name.to_string())
                .collect();
            return Err(CycleError { involved });
        }

        // Commit changes
        self.edges = test_edges;
        self.in_degrees = test_in_degrees;
        Ok(())
    }

    /// Returns the number of registered systems.
    pub fn system_count(&self) -> usize {
        self.systems.len()
    }

    /// Returns a reference to the system at the given index.
    pub(crate) fn get_system(&self, idx: usize) -> &dyn DynSystem {
        &*self.systems[idx].system
    }

    /// Returns the type name of the system at the given index.
    pub fn get_type_name(&self, idx: usize) -> &'static str {
        self.systems[idx].type_name
    }

    /// Returns the indices of systems with no dependencies (in-degree 0).
    ///
    /// These are the systems that can start immediately.
    pub fn ready_indices(&self) -> Vec<usize> {
        self.in_degrees
            .iter()
            .enumerate()
            .filter(|&(_, &deg)| deg == 0)
            .map(|(idx, _)| idx)
            .collect()
    }

    /// Returns the indices of systems that depend on the given system.
    ///
    /// When system `idx` completes, these systems should have their
    /// remaining dependency count decremented.
    pub fn dependents_of(&self, idx: usize) -> &[usize] {
        &self.edges[idx]
    }

    /// Returns the in-degree (number of dependencies) for each system.
    ///
    /// Used by the runner to initialize remaining dependency counters.
    pub fn in_degrees(&self) -> &[usize] {
        &self.in_degrees
    }
}

impl Default for SystemsContainer {
    fn default() -> Self {
        Self::new()
    }
}

/// Runs Kahn's topological sort. Returns `Some(cycle_members)` if a cycle exists,
/// or `None` if the graph is acyclic.
fn find_cycle(edges: &[Vec<usize>], in_degrees: &[usize]) -> Option<Vec<usize>> {
    let n = in_degrees.len();
    let mut remaining = in_degrees.to_vec();
    let mut queue = std::collections::VecDeque::new();
    let mut visited_count = 0;

    for (i, &deg) in remaining.iter().enumerate() {
        if deg == 0 {
            queue.push_back(i);
        }
    }

    while let Some(node) = queue.pop_front() {
        visited_count += 1;
        for &dependent in &edges[node] {
            remaining[dependent] -= 1;
            if remaining[dependent] == 0 {
                queue.push_back(dependent);
            }
        }
    }

    if visited_count == n {
        None
    } else {
        // Nodes with remaining > 0 are in cycles
        let cycle_members: Vec<usize> = remaining
            .iter()
            .enumerate()
            .filter(|&(_, &r)| r > 0)
            .map(|(i, _)| i)
            .collect();
        Some(cycle_members)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::system_context::SystemContext;

    struct SystemA;
    impl System for SystemA {
        async fn run<'a>(&'a self, _ctx: &'a SystemContext<'a>) {}
    }

    struct SystemB;
    impl System for SystemB {
        async fn run<'a>(&'a self, _ctx: &'a SystemContext<'a>) {}
    }

    struct SystemC;
    impl System for SystemC {
        async fn run<'a>(&'a self, _ctx: &'a SystemContext<'a>) {}
    }

    #[test]
    fn add_systems() {
        let mut container = SystemsContainer::new();
        container.add(SystemA);
        container.add(SystemB);
        assert_eq!(container.system_count(), 2);
    }

    #[test]
    #[should_panic(expected = "already registered")]
    fn duplicate_system_panics() {
        let mut container = SystemsContainer::new();
        container.add(SystemA);
        container.add(SystemA);
    }

    #[test]
    fn add_edge_creates_dependency() {
        let mut container = SystemsContainer::new();
        container.add(SystemA);
        container.add(SystemB);
        container.add_edge::<SystemA, SystemB>().unwrap();

        // A has no deps, B depends on A
        let ready = container.ready_indices();
        assert_eq!(ready, vec![0]); // Only SystemA (index 0)

        assert_eq!(container.dependents_of(0), &[1]); // A → B
        assert!(container.dependents_of(1).is_empty()); // B → nothing
    }

    #[test]
    fn all_ready_when_no_edges() {
        let mut container = SystemsContainer::new();
        container.add(SystemA);
        container.add(SystemB);
        container.add(SystemC);

        let ready = container.ready_indices();
        assert_eq!(ready, vec![0, 1, 2]);
    }

    #[test]
    fn chain_dependencies() {
        let mut container = SystemsContainer::new();
        container.add(SystemA);
        container.add(SystemB);
        container.add(SystemC);
        container.add_edge::<SystemA, SystemB>().unwrap();
        container.add_edge::<SystemB, SystemC>().unwrap();

        let ready = container.ready_indices();
        assert_eq!(ready, vec![0]); // Only A is ready

        assert_eq!(container.in_degrees(), &[0, 1, 1]);
    }

    #[test]
    fn diamond_dependencies() {
        let mut container = SystemsContainer::new();
        container.add(SystemA);
        container.add(SystemB);
        container.add(SystemC);

        // A → B, A → C
        container
            .add_edges(&[
                Edge::new::<SystemA, SystemB>(),
                Edge::new::<SystemA, SystemC>(),
            ])
            .unwrap();

        let ready = container.ready_indices();
        assert_eq!(ready, vec![0]); // Only A

        assert_eq!(container.dependents_of(0).len(), 2); // A → B, C
    }

    #[test]
    fn cycle_detection_simple() {
        let mut container = SystemsContainer::new();
        container.add(SystemA);
        container.add(SystemB);
        container.add_edge::<SystemA, SystemB>().unwrap();

        let result = container.add_edge::<SystemB, SystemA>();
        assert!(result.is_err());

        let err = result.unwrap_err();
        assert_eq!(err.involved.len(), 2);
    }

    #[test]
    fn cycle_detection_three_node() {
        let mut container = SystemsContainer::new();
        container.add(SystemA);
        container.add(SystemB);
        container.add(SystemC);
        container.add_edge::<SystemA, SystemB>().unwrap();
        container.add_edge::<SystemB, SystemC>().unwrap();

        let result = container.add_edge::<SystemC, SystemA>();
        assert!(result.is_err());
    }

    #[test]
    fn cycle_does_not_modify_state() {
        let mut container = SystemsContainer::new();
        container.add(SystemA);
        container.add(SystemB);
        container.add_edge::<SystemA, SystemB>().unwrap();

        let original_in_degrees = container.in_degrees().to_vec();

        // Attempt to add cycle — should fail
        let _ = container.add_edge::<SystemB, SystemA>();

        // State should be unchanged
        assert_eq!(container.in_degrees(), &original_in_degrees);
    }

    #[test]
    fn batch_edges_all_or_nothing() {
        let mut container = SystemsContainer::new();
        container.add(SystemA);
        container.add(SystemB);
        container.add(SystemC);
        container.add_edge::<SystemA, SystemB>().unwrap();

        // Batch that includes a cycle: A→B already exists, add B→C and C→A
        let result = container.add_edges(&[
            Edge::new::<SystemB, SystemC>(),
            Edge::new::<SystemC, SystemA>(),
        ]);
        assert!(result.is_err());

        // B→C should NOT have been applied either
        assert!(container.dependents_of(1).is_empty()); // B has no dependents
    }

    #[test]
    fn duplicate_edge_is_idempotent() {
        let mut container = SystemsContainer::new();
        container.add(SystemA);
        container.add(SystemB);
        container.add_edge::<SystemA, SystemB>().unwrap();
        container.add_edge::<SystemA, SystemB>().unwrap(); // duplicate

        assert_eq!(container.in_degrees(), &[0, 1]);
        assert_eq!(container.dependents_of(0).len(), 1);
    }

    #[test]
    fn empty_edges_batch_ok() {
        let mut container = SystemsContainer::new();
        container.add(SystemA);
        assert!(container.add_edges(&[]).is_ok());
    }

    #[test]
    fn cycle_error_display() {
        let err = CycleError {
            involved: vec!["SystemA".to_string(), "SystemB".to_string()],
        };
        let msg = format!("{err}");
        assert!(msg.contains("SystemA"));
        assert!(msg.contains("SystemB"));
    }

    #[test]
    fn get_system_and_type_name() {
        let mut container = SystemsContainer::new();
        container.add(SystemA);

        assert!(container.get_type_name(0).contains("SystemA"));
    }
}
