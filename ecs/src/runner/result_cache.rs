use std::any::Any;
use std::collections::HashMap;
use std::sync::Weak;

use crate::system::{SystemError, SystemsContainer};
use crate::world::World;

type Results = Vec<Option<Box<dyn Any + Send + Sync>>>;

struct CachedResults {
    owner: Weak<()>,
    results: Results,
}

/// Runner-owned results. The weak marker detects a missing lifecycle cleanup;
/// it neither owns the schedule nor automatically destroys its cached values.
#[derive(Default)]
pub(super) struct ResultCache {
    schedules: HashMap<u64, CachedResults>,
}

impl ResultCache {
    /// Scan only when a new schedule is admitted. Keep even an empty entry so
    /// repeated runs (including empty schedules) avoid another orphan scan.
    pub(super) fn validate(
        &mut self,
        systems: &SystemsContainer,
        world: &World,
    ) -> Result<(), SystemError> {
        let id = systems.container_id();
        if self.schedules.contains_key(&id) {
            return systems.bind_world(world);
        }
        let mut container_ids: Vec<_> = self
            .schedules
            .iter()
            .filter_map(|(&id, cached)| {
                (cached.owner.strong_count() == 0 && cached.results.iter().any(Option::is_some))
                    .then_some(id)
            })
            .collect();
        if !container_ids.is_empty() {
            container_ids.sort_unstable();
            return Err(SystemError::OrphanedScheduleResults { container_ids });
        }
        // A rejected orphan check must not bind a previously unused container.
        // Likewise, a world mismatch must not register it in this runner's cache.
        systems.bind_world(world)?;
        self.schedules.insert(
            id,
            CachedResults {
                owner: systems.lifetime_marker(),
                results: Vec::new(),
            },
        );
        Ok(())
    }

    pub(super) fn take(&mut self, systems: &SystemsContainer) -> Results {
        self.schedules
            .get_mut(&systems.container_id())
            .map(|cached| std::mem::take(&mut cached.results))
            .unwrap_or_default()
    }

    pub(super) fn store(&mut self, systems: &SystemsContainer, results: Results) {
        self.schedules
            .entry(systems.container_id())
            .or_insert_with(|| CachedResults {
                owner: systems.lifetime_marker(),
                results: Vec::new(),
            })
            .results = results;
    }
}
