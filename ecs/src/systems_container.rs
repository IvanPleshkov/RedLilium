use std::any::{Any, TypeId};
use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::{Arc, RwLock};

use crate::condition::{ConditionMode, ConditionResult, condition_checker};

/// Function pointer type for condition checking.
type ConditionCheckerFn = fn(&(dyn Any + Send + Sync)) -> bool;
use crate::function_system::IntoSystem;
use crate::system::{
    DynExclusiveSystem, DynSystem, ExclusiveFunctionSystem, ExclusiveSystem, System,
};

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

/// A registered system entry — either regular or exclusive.
pub(crate) enum SystemEntry {
    /// A regular system that borrows `&World` via `SystemContext`.
    Regular(Arc<RwLock<dyn DynSystem>>),
    /// An exclusive system that receives `&mut World` directly.
    Exclusive(Arc<RwLock<dyn DynExclusiveSystem>>),
}

/// Container for registered systems with explicit dependency ordering.
///
/// Systems are registered with [`add()`](SystemsContainer::add) (regular)
/// or [`add_exclusive()`](SystemsContainer::add_exclusive) (exclusive).
/// Ordering constraints are added with
/// [`add_edge()`](SystemsContainer::add_edge) or
/// [`add_edges()`](SystemsContainer::add_edges). Cycle detection uses Kahn's
/// topological sort algorithm.
///
/// Edges work between any combination of regular and exclusive systems.
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
    systems: Vec<SystemEntry>,
    /// Cached type names to avoid locking just for diagnostics.
    names: Vec<&'static str>,
    /// Forward adjacency list: edges[i] = indices of systems that depend on i.
    edges: Vec<Vec<usize>>,
    /// Cached in-degree per system (number of dependencies).
    in_degrees: Vec<usize>,
    /// Map from TypeId to index in `systems`.
    id_to_idx: HashMap<TypeId, usize>,
    /// Reverse map from index to TypeId.
    idx_to_id: Vec<TypeId>,
    /// Pre-computed topological order for single-threaded execution.
    single_thread_order: Vec<usize>,
    /// For each system index, the set of ancestor system TypeIds whose
    /// results are accessible (i.e. guaranteed to have completed before
    /// this system starts, based on the dependency graph).
    accessible_results: Vec<HashSet<TypeId>>,
    /// Per-system condition checker. `Some` if this system was registered
    /// via [`add_condition`](SystemsContainer::add_condition).
    condition_checkers: Vec<Option<ConditionCheckerFn>>,
    /// For each system, the indices of condition systems that gate it.
    /// Populated automatically by [`add_edges`](SystemsContainer::add_edges)
    /// when the source system has a condition checker.
    condition_edges: Vec<Vec<usize>>,
    /// For each system, how its conditions are combined.
    condition_modes: Vec<ConditionMode>,
}

impl SystemsContainer {
    /// Creates a new empty systems container.
    pub fn new() -> Self {
        Self {
            systems: Vec::new(),
            names: Vec::new(),
            edges: Vec::new(),
            in_degrees: Vec::new(),
            id_to_idx: HashMap::new(),
            idx_to_id: Vec::new(),
            single_thread_order: Vec::new(),
            accessible_results: Vec::new(),
            condition_checkers: Vec::new(),
            condition_edges: Vec::new(),
            condition_modes: Vec::new(),
        }
    }

    /// Registers a system, wrapping it in `Arc<RwLock<S>>`.
    ///
    /// Returns the typed `Arc` handle so the caller can keep a clone
    /// for external access (e.g. inspector, editor). The container
    /// stores a coerced `Arc<RwLock<dyn DynSystem>>` that shares the
    /// same underlying data.
    ///
    /// # Panics
    ///
    /// Panics if a system of the same type is already registered.
    pub fn add<S: System>(&mut self, system: S) -> Arc<RwLock<S>> {
        let type_id = TypeId::of::<S>();

        assert!(
            !self.id_to_idx.contains_key(&type_id),
            "System `{}` is already registered",
            std::any::type_name::<S>()
        );

        let arc = Arc::new(RwLock::new(system));
        let idx = self.systems.len();
        self.id_to_idx.insert(type_id, idx);
        self.idx_to_id.push(type_id);
        self.systems.push(SystemEntry::Regular(arc.clone()));
        self.names.push(std::any::type_name::<S>());
        self.edges.push(Vec::new());
        self.in_degrees.push(0);
        self.accessible_results.push(HashSet::new());
        self.condition_checkers.push(None);
        self.condition_edges.push(Vec::new());
        self.condition_modes.push(ConditionMode::All);
        self.rebuild_order();
        arc
    }

    /// Registers a condition system.
    ///
    /// Like [`add()`](Self::add), but marks the system as a condition.
    /// When edges are added from this system to other systems, those
    /// edges are automatically treated as condition edges — the target
    /// system will only run if this condition returns
    /// [`Condition::True`](crate::Condition::True).
    ///
    /// The compile-time bound `S::Result: ConditionResult` ensures only
    /// systems returning [`Condition<T>`](crate::Condition) can be registered
    /// as conditions.
    ///
    /// # Panics
    ///
    /// Panics if a system of the same type is already registered.
    pub fn add_condition<S: System>(&mut self, system: S) -> Arc<RwLock<S>>
    where
        S::Result: ConditionResult,
    {
        let arc = self.add(system);
        let idx = self.id_to_idx[&TypeId::of::<S>()];
        self.condition_checkers[idx] = Some(condition_checker::<S::Result>);
        arc
    }

    /// Registers a per-entity function as a system.
    ///
    /// The function is called once per matching entity during each tick.
    /// Locking and inner-join iteration are handled automatically.
    /// This is the simplest way to add logic that processes entities.
    ///
    /// # Example
    ///
    /// ```ignore
    /// fn movement((pos, vel): (&mut Position, &Velocity)) {
    ///     pos.x += vel.x;
    /// }
    ///
    /// let mut container = SystemsContainer::new();
    /// container.add_fn::<(Write<Position>, Read<Velocity>), _>(movement);
    /// ```
    pub fn add_fn<A, F>(
        &mut self,
        func: F,
    ) -> Arc<RwLock<crate::function_system::ForEachSystem<F, A>>>
    where
        A: crate::function_system::ForEachAccess + Send + Sync + 'static,
        F: for<'a> Fn(A::EachItem<'a>) + Send + Sync + 'static,
    {
        self.add(crate::function_system::for_each::<A, F>(func))
    }

    /// Registers a function that receives raw component storages as a system.
    ///
    /// Unlike [`add_fn`](Self::add_fn), this gives you the full storage
    /// objects (`Ref<T>`, `RefMut<T>`) so you can iterate and join manually.
    /// Use this when you need custom iteration patterns.
    ///
    /// # Example
    ///
    /// ```ignore
    /// fn gravity((mut velocities,): (RefMut<Velocity>,)) {
    ///     for (_, vel) in velocities.iter_mut() {
    ///         vel.y -= 9.81;
    ///     }
    /// }
    ///
    /// let mut container = SystemsContainer::new();
    /// container.add_fn_raw::<(Write<Velocity>,), _>(gravity);
    /// ```
    pub fn add_fn_raw<Marker, F: IntoSystem<Marker>>(&mut self, func: F) -> Arc<RwLock<F::System>> {
        self.add(func.into_system())
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
        match topological_sort(&test_edges, &test_in_degrees) {
            Ok(order) => {
                self.edges = test_edges;
                self.in_degrees = test_in_degrees;
                self.single_thread_order = order;
                self.rebuild_accessible_results();

                // Auto-detect condition edges: if the source system was
                // registered via add_condition(), mark the edge as conditional.
                for &(from, to) in &resolved {
                    if self.condition_checkers[from].is_some()
                        && !self.condition_edges[to].contains(&from)
                    {
                        self.condition_edges[to].push(from);
                    }
                }

                Ok(())
            }
            Err(cycle_indices) => {
                let involved = cycle_indices
                    .iter()
                    .map(|&idx| self.names[idx].to_string())
                    .collect();
                Err(CycleError { involved })
            }
        }
    }

    /// Returns the number of registered systems.
    pub fn system_count(&self) -> usize {
        self.systems.len()
    }

    /// Registers an exclusive system, wrapping it in `Arc<RwLock<S>>`.
    ///
    /// Returns the typed `Arc` handle. Exclusive systems receive `&mut World`
    /// directly and act as barriers in the scheduler.
    ///
    /// # Panics
    ///
    /// Panics if a system of the same type is already registered.
    pub fn add_exclusive<S: ExclusiveSystem>(&mut self, system: S) -> Arc<RwLock<S>> {
        let type_id = TypeId::of::<S>();

        assert!(
            !self.id_to_idx.contains_key(&type_id),
            "System `{}` is already registered",
            std::any::type_name::<S>()
        );

        let arc = Arc::new(RwLock::new(system));
        let idx = self.systems.len();
        self.id_to_idx.insert(type_id, idx);
        self.idx_to_id.push(type_id);
        self.systems.push(SystemEntry::Exclusive(arc.clone()));
        self.names.push(std::any::type_name::<S>());
        self.edges.push(Vec::new());
        self.in_degrees.push(0);
        self.accessible_results.push(HashSet::new());
        self.condition_checkers.push(None);
        self.condition_edges.push(Vec::new());
        self.condition_modes.push(ConditionMode::All);
        self.rebuild_order();
        arc
    }

    /// Registers a closure as an exclusive system.
    ///
    /// # Example
    ///
    /// ```ignore
    /// container.add_exclusive_fn(|world: &mut World| {
    ///     world.insert_resource(42u32);
    /// });
    /// ```
    pub fn add_exclusive_fn<F>(&mut self, func: F) -> Arc<RwLock<ExclusiveFunctionSystem<F>>>
    where
        F: FnMut(&mut crate::world::World) + Send + Sync + 'static,
    {
        self.add_exclusive(ExclusiveFunctionSystem::new(func))
    }

    /// Returns whether the system at the given index is exclusive.
    pub(crate) fn is_exclusive(&self, idx: usize) -> bool {
        matches!(&self.systems[idx], SystemEntry::Exclusive(_))
    }

    /// Returns the `Arc<RwLock<dyn DynSystem>>` for the regular system at the given index.
    ///
    /// The runner read-locks this to call `run_boxed()`.
    ///
    /// # Panics
    ///
    /// Panics if the system at `idx` is exclusive.
    pub(crate) fn get_system(&self, idx: usize) -> &Arc<RwLock<dyn DynSystem>> {
        match &self.systems[idx] {
            SystemEntry::Regular(sys) => sys,
            SystemEntry::Exclusive(_) => {
                panic!(
                    "system `{}` at index {} is exclusive, not regular",
                    self.names[idx], idx
                )
            }
        }
    }

    /// Returns the `Arc<RwLock<dyn DynExclusiveSystem>>` for the exclusive system
    /// at the given index.
    ///
    /// The runner write-locks this to call `run_boxed()`.
    ///
    /// # Panics
    ///
    /// Panics if the system at `idx` is regular.
    pub(crate) fn get_exclusive_system(&self, idx: usize) -> &Arc<RwLock<dyn DynExclusiveSystem>> {
        match &self.systems[idx] {
            SystemEntry::Exclusive(sys) => sys,
            SystemEntry::Regular(_) => {
                panic!(
                    "system `{}` at index {} is regular, not exclusive",
                    self.names[idx], idx
                )
            }
        }
    }

    /// Returns the type name of the system at the given index.
    pub fn get_type_name(&self, idx: usize) -> &'static str {
        self.names[idx]
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

    /// Returns a pre-computed topological order for single-threaded execution.
    ///
    /// Iterating this slice and running each system sequentially guarantees
    /// all dependency constraints are satisfied.
    pub fn single_thread_order(&self) -> &[usize] {
        &self.single_thread_order
    }

    /// Returns the set of ancestor system `TypeId`s whose results are
    /// accessible to the system at the given index.
    ///
    /// Only systems that are guaranteed to complete before this system
    /// (via transitive dependency edges) are included.
    pub(crate) fn accessible_results(&self, idx: usize) -> &HashSet<TypeId> {
        &self.accessible_results[idx]
    }

    /// Returns the system index for a given `TypeId`, if registered.
    pub(crate) fn type_id_to_idx(&self) -> &HashMap<TypeId, usize> {
        &self.id_to_idx
    }

    /// Returns the `TypeId` for the system at the given index.
    pub(crate) fn idx_to_type_id(&self, idx: usize) -> TypeId {
        self.idx_to_id[idx]
    }

    /// Sets the condition evaluation mode for system `T`.
    ///
    /// By default, all conditions must pass ([`ConditionMode::All`]).
    /// Use this to switch to [`ConditionMode::Any`] where at least
    /// one passing condition is sufficient.
    ///
    /// # Panics
    ///
    /// Panics if `T` is not registered.
    pub fn set_condition_mode<T: 'static>(&mut self, mode: ConditionMode) {
        let idx = *self
            .id_to_idx
            .get(&TypeId::of::<T>())
            .unwrap_or_else(|| panic!("System `{}` is not registered", std::any::type_name::<T>()));
        self.condition_modes[idx] = mode;
    }

    /// Evaluates the run conditions for the system at `idx`.
    ///
    /// Returns `true` if the system should run, `false` if it should be
    /// skipped. Systems with no condition edges always return `true`.
    pub(crate) fn check_conditions(
        &self,
        idx: usize,
        results: &crate::system_results_store::SystemResultsStore,
    ) -> bool {
        let conditions = &self.condition_edges[idx];
        if conditions.is_empty() {
            return true;
        }
        match self.condition_modes[idx] {
            ConditionMode::All => conditions.iter().all(|&cond_idx| {
                let checker =
                    self.condition_checkers[cond_idx].expect("condition system missing checker");
                results.get_raw(cond_idx).is_some_and(checker)
            }),
            ConditionMode::Any => conditions.iter().any(|&cond_idx| {
                let checker =
                    self.condition_checkers[cond_idx].expect("condition system missing checker");
                results.get_raw(cond_idx).is_some_and(checker)
            }),
        }
    }

    /// Rebuilds `single_thread_order` and `accessible_results` from the current graph.
    fn rebuild_order(&mut self) {
        // Graph is always acyclic (validated by add_edges), so unwrap is safe.
        self.single_thread_order =
            topological_sort(&self.edges, &self.in_degrees).expect("graph should be acyclic");
        self.rebuild_accessible_results();
    }

    /// Computes the transitive ancestor set for each system.
    ///
    /// For each system S in topological order:
    ///   accessible[S] = union of {P} ∪ accessible[P] for each direct predecessor P
    fn rebuild_accessible_results(&mut self) {
        let n = self.systems.len();

        // Build reverse edges: reverse_edges[i] = direct predecessors of i
        let mut reverse_edges: Vec<Vec<usize>> = vec![vec![]; n];
        for (from, deps) in self.edges.iter().enumerate() {
            for &to in deps {
                reverse_edges[to].push(from);
            }
        }

        let mut accessible: Vec<HashSet<TypeId>> = vec![HashSet::new(); n];

        for &idx in &self.single_thread_order {
            let mut set = HashSet::new();
            for &pred_idx in &reverse_edges[idx] {
                set.insert(self.idx_to_id[pred_idx]);
                set.extend(&accessible[pred_idx]);
            }
            accessible[idx] = set;
        }

        self.accessible_results = accessible;
    }
}

impl Default for SystemsContainer {
    fn default() -> Self {
        Self::new()
    }
}

/// Runs Kahn's topological sort. Returns the sorted order, or the cycle
/// members if a cycle exists.
fn topological_sort(edges: &[Vec<usize>], in_degrees: &[usize]) -> Result<Vec<usize>, Vec<usize>> {
    let n = in_degrees.len();
    let mut remaining = in_degrees.to_vec();
    let mut queue = std::collections::VecDeque::new();
    let mut order = Vec::with_capacity(n);

    for (i, &deg) in remaining.iter().enumerate() {
        if deg == 0 {
            queue.push_back(i);
        }
    }

    while let Some(node) = queue.pop_front() {
        order.push(node);
        for &dependent in &edges[node] {
            remaining[dependent] -= 1;
            if remaining[dependent] == 0 {
                queue.push_back(dependent);
            }
        }
    }

    if order.len() == n {
        Ok(order)
    } else {
        let cycle_members = remaining
            .iter()
            .enumerate()
            .filter(|&(_, &r)| r > 0)
            .map(|(i, _)| i)
            .collect();
        Err(cycle_members)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::system_context::SystemContext;

    struct SystemA;
    impl System for SystemA {
        type Result = ();
        fn run<'a>(
            &'a self,
            _ctx: &'a SystemContext<'a>,
        ) -> Result<(), crate::system::SystemError> {
            Ok(())
        }
    }

    struct SystemB;
    impl System for SystemB {
        type Result = ();
        fn run<'a>(
            &'a self,
            _ctx: &'a SystemContext<'a>,
        ) -> Result<(), crate::system::SystemError> {
            Ok(())
        }
    }

    struct SystemC;
    impl System for SystemC {
        type Result = ();
        fn run<'a>(
            &'a self,
            _ctx: &'a SystemContext<'a>,
        ) -> Result<(), crate::system::SystemError> {
            Ok(())
        }
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

    // ---- Exclusive system tests ----

    struct ExclusiveA;
    impl ExclusiveSystem for ExclusiveA {
        type Result = ();
        fn run(
            &mut self,
            _world: &mut crate::world::World,
        ) -> Result<(), crate::system::SystemError> {
            Ok(())
        }
    }

    struct ExclusiveB;
    impl ExclusiveSystem for ExclusiveB {
        type Result = ();
        fn run(
            &mut self,
            _world: &mut crate::world::World,
        ) -> Result<(), crate::system::SystemError> {
            Ok(())
        }
    }

    #[test]
    fn add_exclusive_system() {
        let mut container = SystemsContainer::new();
        container.add_exclusive(ExclusiveA);
        assert_eq!(container.system_count(), 1);
        assert!(container.is_exclusive(0));
    }

    #[test]
    fn add_exclusive_fn_system() {
        let mut container = SystemsContainer::new();
        container.add_exclusive_fn(|_world: &mut crate::world::World| {});
        assert_eq!(container.system_count(), 1);
        assert!(container.is_exclusive(0));
    }

    #[test]
    fn is_exclusive_false_for_regular() {
        let mut container = SystemsContainer::new();
        container.add(SystemA);
        assert!(!container.is_exclusive(0));
    }

    #[test]
    #[should_panic(expected = "already registered")]
    fn duplicate_exclusive_panics() {
        let mut container = SystemsContainer::new();
        container.add_exclusive(ExclusiveA);
        container.add_exclusive(ExclusiveA);
    }

    #[test]
    fn edge_regular_to_exclusive() {
        let mut container = SystemsContainer::new();
        container.add(SystemA);
        container.add_exclusive(ExclusiveA);
        container.add_edge::<SystemA, ExclusiveA>().unwrap();

        assert_eq!(container.ready_indices(), vec![0]);
        assert_eq!(container.dependents_of(0), &[1]);
    }

    #[test]
    fn edge_exclusive_to_regular() {
        let mut container = SystemsContainer::new();
        container.add_exclusive(ExclusiveA);
        container.add(SystemA);
        container.add_edge::<ExclusiveA, SystemA>().unwrap();

        assert_eq!(container.ready_indices(), vec![0]);
        assert_eq!(container.dependents_of(0), &[1]);
    }

    #[test]
    fn edge_exclusive_to_exclusive() {
        let mut container = SystemsContainer::new();
        container.add_exclusive(ExclusiveA);
        container.add_exclusive(ExclusiveB);
        container.add_edge::<ExclusiveA, ExclusiveB>().unwrap();

        assert_eq!(container.ready_indices(), vec![0]);
    }

    #[test]
    fn mixed_dependency_chain() {
        let mut container = SystemsContainer::new();
        container.add(SystemA);
        container.add_exclusive(ExclusiveA);
        container.add(SystemB);
        container.add_edge::<SystemA, ExclusiveA>().unwrap();
        container.add_edge::<ExclusiveA, SystemB>().unwrap();

        assert_eq!(container.in_degrees(), &[0, 1, 1]);
        assert_eq!(container.ready_indices(), vec![0]);
    }

    #[test]
    #[should_panic(expected = "is exclusive, not regular")]
    fn get_system_panics_for_exclusive() {
        let mut container = SystemsContainer::new();
        container.add_exclusive(ExclusiveA);
        container.get_system(0);
    }

    #[test]
    #[should_panic(expected = "is regular, not exclusive")]
    fn get_exclusive_system_panics_for_regular() {
        let mut container = SystemsContainer::new();
        container.add(SystemA);
        container.get_exclusive_system(0);
    }

    // ---- Condition system tests ----

    use crate::condition::{Condition, ConditionMode};

    struct CondTrue;
    impl System for CondTrue {
        type Result = Condition<()>;
        fn run<'a>(
            &'a self,
            _ctx: &'a SystemContext<'a>,
        ) -> Result<Condition<()>, crate::system::SystemError> {
            Ok(Condition::True(()))
        }
    }

    struct CondFalse;
    impl System for CondFalse {
        type Result = Condition<()>;
        fn run<'a>(
            &'a self,
            _ctx: &'a SystemContext<'a>,
        ) -> Result<Condition<()>, crate::system::SystemError> {
            Ok(Condition::False)
        }
    }

    #[test]
    fn add_condition_creates_dependency() {
        let mut container = SystemsContainer::new();
        container.add_condition(CondTrue);
        container.add(SystemA);
        container.add_edge::<CondTrue, SystemA>().unwrap();

        // Condition edge auto-detected
        assert_eq!(container.condition_edges[1], vec![0]);
        // Also a regular dependency
        assert_eq!(container.dependents_of(0), &[1]);
    }

    #[test]
    fn add_edge_non_condition_no_condition_edge() {
        let mut container = SystemsContainer::new();
        container.add(SystemA);
        container.add(SystemB);
        container.add_edge::<SystemA, SystemB>().unwrap();

        // No condition edge created
        assert!(container.condition_edges[1].is_empty());
    }

    #[test]
    fn duplicate_condition_edge_is_idempotent() {
        let mut container = SystemsContainer::new();
        container.add_condition(CondTrue);
        container.add(SystemA);
        container.add_edge::<CondTrue, SystemA>().unwrap();
        container.add_edge::<CondTrue, SystemA>().unwrap(); // duplicate

        assert_eq!(container.condition_edges[1].len(), 1);
    }

    #[test]
    fn set_condition_mode_works() {
        let mut container = SystemsContainer::new();
        container.add(SystemA);
        assert_eq!(container.condition_modes[0], ConditionMode::All);

        container.set_condition_mode::<SystemA>(ConditionMode::Any);
        assert_eq!(container.condition_modes[0], ConditionMode::Any);
    }

    #[test]
    #[should_panic(expected = "is not registered")]
    fn set_condition_mode_panics_unregistered() {
        let mut container = SystemsContainer::new();
        container.set_condition_mode::<SystemA>(ConditionMode::Any);
    }

    #[test]
    fn check_conditions_no_conditions_returns_true() {
        use crate::system_results_store::SystemResultsStore;
        use std::collections::HashMap;

        let mut container = SystemsContainer::new();
        container.add(SystemA);

        let store = SystemResultsStore::new(1, HashMap::new());
        assert!(container.check_conditions(0, &store));
    }

    #[test]
    fn check_conditions_all_mode() {
        use crate::system_results_store::SystemResultsStore;

        let mut container = SystemsContainer::new();
        container.add_condition(CondTrue);
        container.add_condition(CondFalse);
        container.add(SystemA);
        container.add_edge::<CondTrue, SystemA>().unwrap();
        container.add_edge::<CondFalse, SystemA>().unwrap();

        let store = SystemResultsStore::new(3, container.type_id_to_idx().clone());
        store.store(0, Box::new(Condition::<()>::True(())));
        store.store(1, Box::new(Condition::<()>::False));

        // All mode: one false → overall false
        assert!(!container.check_conditions(2, &store));
    }

    #[test]
    fn check_conditions_any_mode() {
        use crate::system_results_store::SystemResultsStore;

        let mut container = SystemsContainer::new();
        container.add_condition(CondTrue);
        container.add_condition(CondFalse);
        container.add(SystemA);
        container.add_edge::<CondTrue, SystemA>().unwrap();
        container.add_edge::<CondFalse, SystemA>().unwrap();
        container.set_condition_mode::<SystemA>(ConditionMode::Any);

        let store = SystemResultsStore::new(3, container.type_id_to_idx().clone());
        store.store(0, Box::new(Condition::<()>::True(())));
        store.store(1, Box::new(Condition::<()>::False));

        // Any mode: one true → overall true
        assert!(container.check_conditions(2, &store));
    }
}
