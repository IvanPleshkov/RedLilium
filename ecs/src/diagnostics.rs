//! Runtime diagnostics for ECS system scheduling.
//!
//! Provides opt-in collection of ambiguity detection and timing data
//! without compile-time feature flags. Pass [`RunDiagnostics`] to
//! [`run_with()`](crate::EcsRunner::run_with) to control what is collected.

use std::any::TypeId;
use std::fmt;
use std::sync::Mutex;
use std::time::Duration;

use crate::access_set::{AccessInfo, normalize_access_infos};
use crate::system::SystemError;
use crate::systems_container::SystemsContainer;
use crate::world::World;

// ---------------------------------------------------------------------------
// Public configuration & result types
// ---------------------------------------------------------------------------

/// Configuration for what diagnostic data to collect during a run.
///
/// All fields default to `false` (no collection, zero overhead).
///
/// # Example
///
/// ```ignore
/// let result = runner.run_with(&mut world, &systems, &RunDiagnostics {
///     detect_ambiguities: true,
///     collect_timings: true,
/// });
/// println!("{}", result.report);
/// ```
#[derive(Debug, Clone, Default)]
pub struct RunDiagnostics {
    /// Record per-system access patterns and detect ambiguities.
    pub detect_ambiguities: bool,
    /// Record per-system wall-clock timing and CPU utilization.
    pub collect_timings: bool,
}

/// Results from a single ECS run.
pub struct RunResult {
    /// Errors from system execution (panics).
    pub errors: Vec<SystemError>,
    /// Diagnostic report. Fields are populated based on [`RunDiagnostics`] settings.
    pub report: RunReport,
}

/// Diagnostic report from a single ECS run.
///
/// Fields are `None` when the corresponding diagnostic was not requested.
#[derive(Debug, Default)]
pub struct RunReport {
    /// Detected ambiguities between unordered systems.
    /// `None` if `detect_ambiguities` was not set.
    pub ambiguities: Option<Vec<AmbiguityInfo>>,
    /// Timing data. `None` if `collect_timings` was not set.
    pub timings: Option<TimingReport>,
}

/// Timing data from a single ECS run.
#[derive(Debug)]
pub struct TimingReport {
    /// Wall-clock time for the entire run.
    pub wall_time: Duration,
    /// Sum of all system execution durations across all threads.
    pub total_cpu_time: Duration,
    /// Number of worker threads used (1 for single-threaded runner).
    pub num_threads: usize,
    /// Per-system timing in execution order.
    pub systems: Vec<SystemTiming>,
}

/// Timing for a single system execution.
#[derive(Debug)]
pub struct SystemTiming {
    /// System type name.
    pub name: &'static str,
    /// Execution duration (wall-clock).
    pub duration: Duration,
}

/// A detected ambiguity between two unordered systems that access
/// overlapping components/resources with at least one write.
#[derive(Debug)]
pub struct AmbiguityInfo {
    /// Name of the first system.
    pub system_a: &'static str,
    /// Name of the second system.
    pub system_b: &'static str,
    /// The conflicting component/resource accesses.
    pub conflicts: Vec<AccessConflict>,
}

/// A single conflicting access between two systems.
#[derive(Debug)]
pub struct AccessConflict {
    /// TypeId of the component/resource.
    pub type_id: TypeId,
    /// Resolved type name (from the world's component registry, or
    /// `"<resource>"` for unregistered types).
    pub type_name: &'static str,
    /// Whether system A writes this type.
    pub a_writes: bool,
    /// Whether system B writes this type.
    pub b_writes: bool,
}

// ---------------------------------------------------------------------------
// Display impls
// ---------------------------------------------------------------------------

impl fmt::Display for RunReport {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Some(ambiguities) = &self.ambiguities {
            if ambiguities.is_empty() {
                writeln!(f, "No ambiguities detected.")?;
            } else {
                writeln!(f, "Ambiguities ({}):", ambiguities.len())?;
                for a in ambiguities {
                    writeln!(f, "  {a}")?;
                }
            }
        }
        if let Some(timings) = &self.timings {
            writeln!(
                f,
                "Timing: {:.2?} wall, {:.2?} CPU across {} thread(s)",
                timings.wall_time, timings.total_cpu_time, timings.num_threads,
            )?;
            if timings.num_threads > 1 && timings.wall_time.as_nanos() > 0 {
                let available = timings.wall_time.as_secs_f64() * timings.num_threads as f64;
                let used = timings.total_cpu_time.as_secs_f64();
                writeln!(f, "  CPU utilization: {:.1}%", (used / available) * 100.0)?;
            }
            for st in &timings.systems {
                writeln!(f, "  {}: {:.2?}", st.name, st.duration)?;
            }
        }
        Ok(())
    }
}

impl fmt::Display for AmbiguityInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} <-> {}: ", self.system_a, self.system_b)?;
        for (i, c) in self.conflicts.iter().enumerate() {
            if i > 0 {
                write!(f, ", ")?;
            }
            let a_mode = if c.a_writes { "write" } else { "read" };
            let b_mode = if c.b_writes { "write" } else { "read" };
            write!(f, "{} ({}/{})", c.type_name, a_mode, b_mode)?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Internal: access recording
// ---------------------------------------------------------------------------

/// Records component/resource accesses per system during a run.
///
/// Created by the runner when `detect_ambiguities` is enabled.
/// Passed to [`SystemContext`](crate::SystemContext) via builder method.
pub(crate) struct AccessRecorder {
    records: Vec<Mutex<Vec<AccessInfo>>>,
}

impl AccessRecorder {
    /// Creates a recorder with one slot per system.
    pub fn new(system_count: usize) -> Self {
        Self {
            records: (0..system_count).map(|_| Mutex::new(Vec::new())).collect(),
        }
    }

    /// Records access infos for the given system index.
    ///
    /// Called from `SystemContext::record_access()` during lock/query.
    pub fn record(&self, system_idx: usize, infos: &[AccessInfo]) {
        self.records[system_idx]
            .lock()
            .unwrap()
            .extend_from_slice(infos);
    }

    /// Consumes the recorder and returns per-system access records.
    pub fn into_records(self) -> Vec<Vec<AccessInfo>> {
        self.records
            .into_iter()
            .map(|m| m.into_inner().unwrap())
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Internal: ambiguity analysis
// ---------------------------------------------------------------------------

/// Analyzes recorded accesses for ambiguities.
///
/// Two systems are ambiguous if:
/// 1. They access the same component/resource (at least one writing)
/// 2. Neither is a transitive ancestor/descendant of the other
/// 3. Neither is an exclusive system (which acts as a barrier)
pub(crate) fn analyze_ambiguities(
    records: Vec<Vec<AccessInfo>>,
    systems: &SystemsContainer,
    world: &World,
) -> Vec<AmbiguityInfo> {
    let n = records.len();
    let mut ambiguities = Vec::new();

    // Normalize each system's recorded access (merge duplicates, upgrade writes)
    let normalized: Vec<Vec<AccessInfo>> = records
        .into_iter()
        .map(|r| normalize_access_infos(&r))
        .collect();

    for i in 0..n {
        if systems.is_exclusive(i) || normalized[i].is_empty() {
            continue;
        }
        for j in (i + 1)..n {
            if systems.is_exclusive(j) || normalized[j].is_empty() {
                continue;
            }

            // Check if there's an ordering constraint between i and j
            let id_i = systems.idx_to_type_id(i);
            let id_j = systems.idx_to_type_id(j);

            if systems.accessible_results(j).contains(&id_i)
                || systems.accessible_results(i).contains(&id_j)
            {
                continue; // ordered — not ambiguous
            }

            // Find conflicting accesses
            let conflicts = find_conflicts(&normalized[i], &normalized[j], world);
            if !conflicts.is_empty() {
                ambiguities.push(AmbiguityInfo {
                    system_a: systems.get_type_name(i),
                    system_b: systems.get_type_name(j),
                    conflicts,
                });
            }
        }
    }

    ambiguities
}

fn find_conflicts(a: &[AccessInfo], b: &[AccessInfo], world: &World) -> Vec<AccessConflict> {
    let mut conflicts = Vec::new();
    for ai in a {
        for bi in b {
            if ai.type_id == bi.type_id && (ai.is_write || bi.is_write) {
                let type_name = world
                    .component_type_name(ai.type_id)
                    .unwrap_or("<resource>");
                conflicts.push(AccessConflict {
                    type_id: ai.type_id,
                    type_name,
                    a_writes: ai.is_write,
                    b_writes: bi.is_write,
                });
            }
        }
    }
    conflicts
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::access_set::{Read, Write};
    use crate::system::System;
    use crate::system_context::SystemContext;

    struct Position {
        _x: f32,
    }
    struct Velocity {
        _x: f32,
    }

    struct SystemA;
    impl System for SystemA {
        type Result = ();
        fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) -> Result<(), crate::system::SystemError> {
            ctx.lock::<(Write<Position>, Read<Velocity>)>()
                .execute(|_| {});
            Ok(())
        }
    }

    struct SystemB;
    impl System for SystemB {
        type Result = ();
        fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) -> Result<(), crate::system::SystemError> {
            ctx.lock::<(Read<Position>, Write<Velocity>)>()
                .execute(|_| {});
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
    fn detects_ambiguity_between_unordered_systems() {
        use crate::runner::EcsRunnerSingleThread;
        use crate::systems_container::SystemsContainer;

        let mut world = World::new();
        world.register_component::<Position>();
        world.register_component::<Velocity>();
        let e = world.spawn();
        world.insert(e, Position { _x: 0.0 }).unwrap();
        world.insert(e, Velocity { _x: 0.0 }).unwrap();

        let mut container = SystemsContainer::new();
        container.add(SystemA);
        container.add(SystemB);
        // No edges — systems are unordered

        let runner = EcsRunnerSingleThread::new();
        let result = runner.run_with(
            &mut world,
            &container,
            &RunDiagnostics {
                detect_ambiguities: true,
                ..Default::default()
            },
        );

        let ambiguities = result.report.ambiguities.unwrap();
        assert_eq!(ambiguities.len(), 1);
        assert_eq!(ambiguities[0].conflicts.len(), 2); // Position and Velocity
    }

    #[test]
    fn no_ambiguity_when_ordered() {
        use crate::runner::EcsRunnerSingleThread;
        use crate::systems_container::SystemsContainer;

        let mut world = World::new();
        world.register_component::<Position>();
        world.register_component::<Velocity>();
        let e = world.spawn();
        world.insert(e, Position { _x: 0.0 }).unwrap();
        world.insert(e, Velocity { _x: 0.0 }).unwrap();

        let mut container = SystemsContainer::new();
        container.add(SystemA);
        container.add(SystemB);
        container.add_edge::<SystemA, SystemB>().unwrap();

        let runner = EcsRunnerSingleThread::new();
        let result = runner.run_with(
            &mut world,
            &container,
            &RunDiagnostics {
                detect_ambiguities: true,
                ..Default::default()
            },
        );

        let ambiguities = result.report.ambiguities.unwrap();
        assert!(ambiguities.is_empty());
    }

    #[test]
    fn no_ambiguity_with_disjoint_access() {
        use crate::runner::EcsRunnerSingleThread;
        use crate::systems_container::SystemsContainer;

        let mut world = World::new();
        world.register_component::<Position>();
        world.register_component::<Velocity>();
        let e = world.spawn();
        world.insert(e, Position { _x: 0.0 }).unwrap();
        world.insert(e, Velocity { _x: 0.0 }).unwrap();

        let mut container = SystemsContainer::new();
        container.add(SystemA);
        container.add(SystemC); // SystemC accesses nothing
        // No edges — but no overlap

        let runner = EcsRunnerSingleThread::new();
        let result = runner.run_with(
            &mut world,
            &container,
            &RunDiagnostics {
                detect_ambiguities: true,
                ..Default::default()
            },
        );

        let ambiguities = result.report.ambiguities.unwrap();
        assert!(ambiguities.is_empty());
    }

    #[test]
    fn timing_report_collected() {
        use crate::runner::EcsRunnerSingleThread;
        use crate::systems_container::SystemsContainer;

        let mut world = World::new();
        let mut container = SystemsContainer::new();
        container.add(SystemC);

        let runner = EcsRunnerSingleThread::new();
        let result = runner.run_with(
            &mut world,
            &container,
            &RunDiagnostics {
                collect_timings: true,
                ..Default::default()
            },
        );

        let timings = result.report.timings.unwrap();
        assert_eq!(timings.num_threads, 1);
        assert_eq!(timings.systems.len(), 1);
    }

    #[test]
    fn no_diagnostics_returns_empty_report() {
        use crate::runner::EcsRunnerSingleThread;
        use crate::systems_container::SystemsContainer;

        let mut world = World::new();
        let mut container = SystemsContainer::new();
        container.add(SystemC);

        let runner = EcsRunnerSingleThread::new();
        let result = runner.run_with(&mut world, &container, &RunDiagnostics::default());

        assert!(result.report.ambiguities.is_none());
        assert!(result.report.timings.is_none());
    }

    #[test]
    fn display_report() {
        let report = RunReport {
            ambiguities: Some(vec![AmbiguityInfo {
                system_a: "SystemA",
                system_b: "SystemB",
                conflicts: vec![AccessConflict {
                    type_id: TypeId::of::<Position>(),
                    type_name: "Position",
                    a_writes: true,
                    b_writes: false,
                }],
            }]),
            timings: None,
        };
        let s = format!("{report}");
        assert!(s.contains("SystemA <-> SystemB"));
        assert!(s.contains("Position (write/read)"));
    }

    #[test]
    fn access_recorder_records_and_returns() {
        let recorder = AccessRecorder::new(2);
        let info = AccessInfo {
            type_id: TypeId::of::<Position>(),
            is_write: true,
        };
        recorder.record(0, &[info]);
        recorder.record(1, &[info]);

        let records = recorder.into_records();
        assert_eq!(records.len(), 2);
        assert_eq!(records[0].len(), 1);
        assert_eq!(records[1].len(), 1);
    }
}
