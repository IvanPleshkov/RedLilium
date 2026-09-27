#![cfg(not(target_arch = "wasm32"))]
#[path = "fixtures/system_panics/scenarios.rs"]
mod scenarios;
use redlilium_ecs::{EcsRunner, SystemError, SystemsContainer, World};
use std::sync::{Arc, atomic::Ordering};

#[test]
fn panic_boundaries_match_across_runners_and_system_kinds() {
    for multi in [false, true] {
        for kind in 0..3 {
            for failure in 0..3 {
                let runner = if multi {
                    EcsRunner::multi_thread(1)
                } else {
                    EcsRunner::single_thread()
                };
                let mut world = World::new();
                let mut systems = SystemsContainer::new();
                let stats = Arc::new(scenarios::Stats::default());
                scenarios::install(&mut systems, stats.clone(), kind, failure);
                assert!(runner.run(&mut world, &systems).is_empty());
                let errors = runner.run(&mut world, &systems);
                let expected =
                    ["run failure", "reuse failure", "result drop failure"][failure as usize];
                assert!(
                    matches!(errors.as_slice(), [SystemError::Panicked { message, .. }] if message == expected),
                    "{errors:?}"
                );
                assert_eq!(
                    stats.runs.load(Ordering::SeqCst),
                    if failure == 0 { 2 } else { 1 }
                );
                assert_eq!(stats.downstream.load(Ordering::SeqCst), 2);
                assert!(runner.run(&mut world, &systems).is_empty());
                assert_eq!(
                    stats.runs.load(Ordering::SeqCst),
                    if failure == 0 { 3 } else { 2 }
                );
                assert_eq!(stats.downstream.load(Ordering::SeqCst), 3);
            }
        }
    }
}
