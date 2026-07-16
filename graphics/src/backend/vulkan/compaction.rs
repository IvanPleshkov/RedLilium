//! Compacted-size queries for transparent BLAS compaction (#110 phase 3,
//! ADR-032).
//!
//! One [`vk::QueryPool`] of type `ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR`
//! per frame slot, mirroring the timestamp pools. When a BLAS built with
//! `ALLOW_COMPACTION` is encoded, a `vkCmdWriteAccelerationStructuresProperties`
//! records its compacted size into the current slot's pool (after a build→read
//! barrier); the result is read back with `vkGetQueryPoolResults` (no WAIT bit)
//! when that slot retires in `advance_frame` — the slot fence already
//! guarantees availability, the same safety argument as the timestamp and
//! staging-belt readbacks.
//!
//! Each query is **reset individually on the same command buffer** immediately
//! before it is written (`vkCmdResetQueryPool(pool, query, 1)`), so the reset
//! and the write always share a queue — a single per-slot pool is therefore
//! safe even when some AS builds run on the async-compute queue (#110 phase 3 +
//! async builds). This is why, unlike the timestamp pools, no per-queue split
//! or once-per-slot whole-pool reset is needed.
//!
//! The manager reaches each BLAS's compaction state only through a
//! [`Weak`] handle, so a BLAS the user dropped between build and readback
//! simply falls out (its own `Drop` frees its resources); the strong owner is
//! always the [`Blas`](crate::resources::Blas).

use std::sync::{Arc, Weak};

use ash::vk;

use crate::resources::{BlasCompaction, CompactionPhase};

use super::MAX_FRAMES_IN_FLIGHT;

/// Compacted-size queries per frame slot. A frame that builds more compactable
/// BLASes than this clamps — the extras simply stay uncompacted (never an
/// assert), which is always safe.
const QUERIES_PER_POOL: u32 = 64;

/// One (query index, compaction) awaiting readback.
struct PendingQuery {
    query: u32,
    compaction: Weak<BlasCompaction>,
}

/// A query pool for one frame slot plus its per-cycle bookkeeping.
///
/// `next` is the next free query index this slot-cycle; each reserved query is
/// reset on its own command buffer just before being written. Readback and
/// abort reset `next` to 0 and clear `pending` for the slot's next use.
struct PoolSlot {
    pool: vk::QueryPool,
    next: u32,
    pending: Vec<PendingQuery>,
}

/// Per-slot compacted-size query pools — one per Vulkan backend, created only
/// when the device supports ray queries (`accel_loader` is `Some`).
pub struct CompactionQueryManager {
    slots: Vec<PoolSlot>,
}

impl CompactionQueryManager {
    /// Create one query pool per frame slot. Returns `None` (compaction size
    /// queries disabled) if any pool fails to create.
    pub fn new(device: &ash::Device) -> Option<Self> {
        let mut slots = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);
        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            let info = vk::QueryPoolCreateInfo::default()
                .query_type(vk::QueryType::ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR)
                .query_count(QUERIES_PER_POOL);
            match unsafe { device.create_query_pool(&info, None) } {
                Ok(pool) => slots.push(PoolSlot {
                    pool,
                    next: 0,
                    pending: Vec::new(),
                }),
                Err(e) => {
                    log::warn!(
                        "compacted-size query pool creation failed: {e:?}; \
                         BLAS compaction disabled"
                    );
                    for s in &slots {
                        unsafe { device.destroy_query_pool(s.pool, None) };
                    }
                    return None;
                }
            }
        }
        Some(Self { slots })
    }

    /// Reserve a query for a compactable BLAS build in `slot`, resetting just
    /// that query on `cmd` so the reset shares the write's queue. Returns the
    /// `(pool, query)` the caller writes the compacted-size property into, or
    /// `None` if the pool is full this slot (the BLAS stays uncompacted).
    pub fn reserve(
        &mut self,
        core: &ash::Device,
        slot: usize,
        cmd: vk::CommandBuffer,
        compaction: &Arc<BlasCompaction>,
    ) -> Option<(vk::QueryPool, u32)> {
        let ps = &mut self.slots[slot];
        if ps.next >= QUERIES_PER_POOL {
            return None;
        }
        let query = ps.next;
        ps.next += 1;
        // Per-query reset on the writing command buffer (see module docs): the
        // reset and the property write are then always on the same queue.
        unsafe { core.cmd_reset_query_pool(cmd, ps.pool, query, 1) };
        ps.pending.push(PendingQuery {
            query,
            compaction: Arc::downgrade(compaction),
        });
        Some((ps.pool, query))
    }

    /// Read back `slot`'s compacted sizes and deliver them to the BLASes, then
    /// arm the pool for reuse. Called from `advance_frame` for the retiring
    /// slot (its fence has signaled, so the queries are available — no WAIT
    /// bit, same argument as the timestamp readback).
    pub fn read_slot(&mut self, core: &ash::Device, slot: usize) {
        let ps = &mut self.slots[slot];
        if ps.next > 0 && !ps.pending.is_empty() {
            let mut results = vec![0u64; ps.next as usize];
            let res = unsafe {
                core.get_query_pool_results(ps.pool, 0, &mut results, vk::QueryResultFlags::TYPE_64)
            };
            if matches!(res, Ok(()) | Err(vk::Result::NOT_READY)) {
                for pending in &ps.pending {
                    if let Some(compaction) = pending.compaction.upgrade() {
                        let size = results[pending.query as usize];
                        // A zero size means the query was not written (a clamped
                        // or errored submit); leave the BLAS awaiting.
                        if size > 0 {
                            compaction.deliver_size(size);
                        }
                    }
                }
            } else if let Err(e) = res {
                log::debug!("compacted-size query readback failed for slot {slot}: {e:?}");
            }
        }
        ps.next = 0;
        ps.pending.clear();
    }

    /// Abandon `slot`'s reserved queries after a recording error (the command
    /// buffer that carried the reset + writes was never submitted). Rolls each
    /// awaiting BLAS back to [`CompactionPhase::NeedsQuery`] so a later build
    /// retries, and re-arms the pool so its next use resets it.
    pub fn abort_slot(&mut self, slot: usize) {
        let ps = &mut self.slots[slot];
        for pending in &ps.pending {
            if let Some(compaction) = pending.compaction.upgrade()
                && compaction.phase() == CompactionPhase::AwaitingSize
            {
                compaction.set_phase(CompactionPhase::NeedsQuery);
            }
        }
        ps.next = 0;
        ps.pending.clear();
    }

    /// Destroy every query pool. Call before the logical device is destroyed.
    pub fn destroy(&mut self, core: &ash::Device) {
        for s in &self.slots {
            unsafe { core.destroy_query_pool(s.pool, None) };
        }
    }
}
