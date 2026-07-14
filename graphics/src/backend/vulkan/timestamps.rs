//! Per-pass GPU timestamp queries for the Vulkan backend (#95).
//!
//! One [`vk::QueryPool`] per (queue, frame-slot), mirroring the per-slot
//! command pools. Timestamps are written with `vkCmdWriteTimestamp2` (sync2,
//! #94) around every pass and around the whole submit; the pool is reset with a
//! command-buffer `vkCmdResetQueryPool` (no host-reset feature dependency)
//! before its first write in a slot. Results are read back with
//! `vkGetQueryPoolResults` (no WAIT bit) when the slot retires in
//! `advance_frame` — the slot fence already guarantees completion, the same
//! safety argument as the staging belt.
//!
//! Durations only (end − begin within one queue). Cross-queue timestamps are
//! not comparable without `VK_EXT_calibrated_timestamps` (out of scope), so no
//! cross-queue timeline is built.

use ash::vk;

use crate::device::{FrameGpuTimings, SubmitTiming};
use crate::graph::QueuePreference;

use super::MAX_FRAMES_IN_FLIGHT;
use super::barriers::{QUEUE_COUNT, QueueId};

/// Timestamp slots per query pool. Two per submit bracket plus two per pass;
/// this fits ~126 passes per slot. A frame with more passes than fit clamps
/// (drops the extras, warns once) — never asserts.
const QUERIES_PER_POOL: u32 = 256;

/// The `vkCmdWriteTimestamp2` stage for a region's *begin* marker: recorded
/// once all prior commands reach the top of the pipe (region start).
const STAGE_BEGIN: vk::PipelineStageFlags2 = vk::PipelineStageFlags2::TOP_OF_PIPE;
/// The stage for a region's *end* marker: recorded once all commands up to
/// here reach the bottom of the pipe (region complete).
const STAGE_END: vk::PipelineStageFlags2 = vk::PipelineStageFlags2::BOTTOM_OF_PIPE;

/// A pass's begin/end query indices plus its label, kept until readback pairs
/// them with resolved ticks. `None` indices mean the pool was exhausted and
/// this pass was dropped from timing (clamp policy).
struct PassQuery {
    name: String,
    begin: Option<u32>,
    end: Option<u32>,
}

/// One submit's query bookkeeping: the whole-submit bracket plus its passes.
struct SubmitQuery {
    begin: Option<u32>,
    end: Option<u32>,
    passes: Vec<PassQuery>,
}

/// A query pool for one (queue, slot): the Vulkan pool plus the CPU-side
/// bookkeeping for everything written into it since the last reset.
///
/// `next == 0` is the reset trigger: the first `begin_submit` of a slot-cycle
/// (and only the first — subsequent submits have `next > 0` after reserving
/// their bracket) records the pool reset. Read-back and error aborts set
/// `next` back to 0 so the pool is reset again before its next use — which
/// keeps a never-reset pool from ever being written (a validation error).
struct PoolSlot {
    pool: vk::QueryPool,
    /// Next free query index; 0 means "reset before the next write".
    next: u32,
    /// Submits recorded into this pool since the last reset.
    submits: Vec<SubmitQuery>,
}

/// All per-slot pools for one queue, plus the queue's `timestampValidBits`
/// mask width (a family reporting 0 is never given a `QueuePools` — it is
/// skipped entirely, so its passes simply report no timing).
struct QueuePools {
    preference: QueuePreference,
    valid_bits: u32,
    slots: Vec<PoolSlot>,
}

/// A submit currently being recorded. Held on the render thread's stack across
/// pass encoding so the per-pass timestamp writes need no lock (only the
/// reservation in [`TimestampManager::begin_submit`] and the hand-back in
/// [`TimestampManager::finish_submit`] touch the manager).
pub struct SubmitRecording {
    queue: QueueId,
    pool: vk::QueryPool,
    begin: Option<u32>,
    end: Option<u32>,
    passes: Vec<PassQuery>,
}

impl SubmitRecording {
    /// Write the begin timestamp for pass `i` and record its label.
    pub fn pass_begin(
        &mut self,
        sync2: &ash::Device,
        cmd: vk::CommandBuffer,
        i: usize,
        name: &str,
    ) {
        let pass = &mut self.passes[i];
        pass.name.clear();
        pass.name.push_str(name);
        if let Some(query) = pass.begin {
            unsafe { sync2.cmd_write_timestamp2(cmd, STAGE_BEGIN, self.pool, query) };
        }
    }

    /// Write the end timestamp for pass `i`.
    pub fn pass_end(&mut self, sync2: &ash::Device, cmd: vk::CommandBuffer, i: usize) {
        if let Some(query) = self.passes[i].end {
            unsafe { sync2.cmd_write_timestamp2(cmd, STAGE_END, self.pool, query) };
        }
    }

    /// Write the whole-submit end timestamp (call after the last pass, before
    /// `end_command_buffer`).
    pub fn submit_end(&self, sync2: &ash::Device, cmd: vk::CommandBuffer) {
        if let Some(query) = self.end {
            unsafe { sync2.cmd_write_timestamp2(cmd, STAGE_END, self.pool, query) };
        }
    }
}

/// Per-pass GPU timestamp collector — one per Vulkan backend, created only when
/// [`DeviceCapabilities::gpu_timestamps`](crate::device::DeviceCapabilities)
/// is true.
pub struct TimestampManager {
    /// Nanoseconds per timestamp tick (`VkPhysicalDeviceLimits::timestampPeriod`).
    period_ns: f32,
    /// Per-queue pools, indexed by `QueueId as usize`. `None` for a queue that
    /// does not exist or whose family reports `timestampValidBits == 0`.
    queues: [Option<QueuePools>; QUEUE_COUNT],
    /// Timings for the most recently retired slot ([`Self::latest`]).
    latest: FrameGpuTimings,
    /// Warn-once guard for pool exhaustion (clamp policy).
    overflow_warned: bool,
}

/// A queue eligible for timestamp collection: its id, engine-facing preference,
/// and family `timestampValidBits`.
pub struct QueueTimestampInfo {
    pub queue: QueueId,
    pub preference: QueuePreference,
    pub valid_bits: u32,
}

impl TimestampManager {
    /// Create query pools for every queue with non-zero `valid_bits`. Returns
    /// `None` (no collection) if none qualify or `period_ns` is zero.
    pub fn new(
        device: &ash::Device,
        period_ns: f32,
        queue_infos: &[QueueTimestampInfo],
    ) -> Option<Self> {
        if period_ns <= 0.0 {
            return None;
        }

        let mut queues: [Option<QueuePools>; QUEUE_COUNT] = Default::default();
        let mut any = false;

        for info in queue_infos {
            if info.valid_bits == 0 {
                continue;
            }
            let mut slots = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);
            let mut ok = true;
            for _ in 0..MAX_FRAMES_IN_FLIGHT {
                let create_info = vk::QueryPoolCreateInfo::default()
                    .query_type(vk::QueryType::TIMESTAMP)
                    .query_count(QUERIES_PER_POOL);
                match unsafe { device.create_query_pool(&create_info, None) } {
                    Ok(pool) => slots.push(PoolSlot {
                        pool,
                        next: 0,
                        submits: Vec::new(),
                    }),
                    Err(e) => {
                        log::warn!(
                            "timestamp query pool creation failed: {e:?}; disabling for queue {:?}",
                            info.queue
                        );
                        // Roll back the pools already created for this queue.
                        for slot in &slots {
                            unsafe { device.destroy_query_pool(slot.pool, None) };
                        }
                        ok = false;
                        break;
                    }
                }
            }
            if !ok {
                continue;
            }
            queues[info.queue as usize] = Some(QueuePools {
                preference: info.preference,
                valid_bits: info.valid_bits,
                slots,
            });
            any = true;
        }

        any.then_some(Self {
            period_ns,
            queues,
            latest: FrameGpuTimings::default(),
            overflow_warned: false,
        })
    }

    /// Reserve query indices for a submit of `num_passes` passes on `queue` in
    /// `slot`, recording the pool reset (if pending) and the submit-begin
    /// timestamp onto `cmd`. Returns `None` when this queue has no pools.
    pub fn begin_submit(
        &mut self,
        core: &ash::Device,
        sync2: &ash::Device,
        queue: QueueId,
        slot: usize,
        num_passes: usize,
        cmd: vk::CommandBuffer,
    ) -> Option<SubmitRecording> {
        let pools = self.queues[queue as usize].as_mut()?;
        let ps = &mut pools.slots[slot];
        let pool = ps.pool;

        // First write into this slot-cycle (`next == 0`): reset the whole pool
        // (command-based; the slot fence guaranteed the old queries are done).
        // Subsequent submits in the same slot continue appending after their
        // bracket, so they never re-reset and never wipe earlier queries.
        if ps.next == 0 {
            unsafe { core.cmd_reset_query_pool(cmd, pool, 0, QUERIES_PER_POOL) };
            ps.submits.clear();
        }

        // Reserve the submit bracket first so the whole-submit total survives
        // even when per-pass queries are clamped, then the passes.
        let mut overflow = false;
        let alloc = |ps: &mut PoolSlot, overflow: &mut bool| -> Option<u32> {
            if ps.next >= QUERIES_PER_POOL {
                *overflow = true;
                None
            } else {
                let q = ps.next;
                ps.next += 1;
                Some(q)
            }
        };

        let begin = alloc(ps, &mut overflow);
        let end = alloc(ps, &mut overflow);
        let mut passes = Vec::with_capacity(num_passes);
        for _ in 0..num_passes {
            let b = alloc(ps, &mut overflow);
            let e = alloc(ps, &mut overflow);
            passes.push(PassQuery {
                name: String::new(),
                begin: b,
                end: e,
            });
        }

        if overflow && !self.overflow_warned {
            self.overflow_warned = true;
            log::warn!(
                "GPU timestamp pool exhausted ({QUERIES_PER_POOL} queries/slot); \
                 some pass timings dropped this frame (#95)"
            );
        }

        // Submit-begin marker (top of pipe = when the submit starts running).
        if let Some(query) = begin {
            unsafe { sync2.cmd_write_timestamp2(cmd, STAGE_BEGIN, pool, query) };
        }

        Some(SubmitRecording {
            queue,
            pool,
            begin,
            end,
            passes,
        })
    }

    /// Hand a finished recording back to the manager's bookkeeping so its
    /// results are read at slot retirement.
    pub fn finish_submit(&mut self, slot: usize, rec: SubmitRecording) {
        let Some(pools) = self.queues[rec.queue as usize].as_mut() else {
            return;
        };
        pools.slots[slot].submits.push(SubmitQuery {
            begin: rec.begin,
            end: rec.end,
            passes: rec.passes,
        });
    }

    /// Read back `slot`'s results across every queue and store them as
    /// [`Self::latest`], then arm each pool for a fresh reset. Called from
    /// `advance_frame` for the retiring slot (its fence has signaled).
    pub fn read_slot(&mut self, core: &ash::Device, slot: usize) {
        let period_ns = self.period_ns;
        let mut submits = Vec::new();
        let mut scratch: Vec<u64> = Vec::new();

        for pools in self.queues.iter_mut().flatten() {
            let ps = &mut pools.slots[slot];
            if ps.next > 0 && !ps.submits.is_empty() {
                scratch.clear();
                scratch.resize(ps.next as usize, 0);
                // No WAIT bit: the slot fence already guarantees availability.
                // VK_NOT_READY (a partially-written pool from an errored submit)
                // is tolerated — only indices referenced by a stored submit are
                // read, and those were all written.
                let res = unsafe {
                    core.get_query_pool_results(
                        ps.pool,
                        0,
                        &mut scratch,
                        vk::QueryResultFlags::TYPE_64,
                    )
                };
                if matches!(res, Ok(()) | Err(vk::Result::NOT_READY)) {
                    let mask = valid_bits_mask(pools.valid_bits);
                    let preference = pools.preference;
                    for sq in &ps.submits {
                        let total_ms = duration_ms(&scratch, sq.begin, sq.end, mask, period_ns);
                        let passes = sq
                            .passes
                            .iter()
                            .map(|p| {
                                (
                                    p.name.clone(),
                                    duration_ms(&scratch, p.begin, p.end, mask, period_ns),
                                )
                            })
                            .collect();
                        submits.push(SubmitTiming {
                            queue: preference,
                            total_ms,
                            passes,
                        });
                    }
                } else if let Err(e) = res {
                    log::debug!("get_query_pool_results failed for slot {slot}: {e:?}");
                }
            }

            // Arm this pool for reuse regardless of read outcome: `next == 0`
            // makes the next `begin_submit` record a fresh reset.
            ps.next = 0;
            ps.submits.clear();
        }

        self.latest = FrameGpuTimings { submits };
    }

    /// Abandon an in-progress submit's queries for `queue`/`slot` after a
    /// recording error (the command buffer that carried the reset+writes was
    /// never submitted). Re-arms the pool so its next use resets it, avoiding a
    /// write to a never-reset pool.
    pub fn abort_submit(&mut self, queue: QueueId, slot: usize) {
        if let Some(pools) = self.queues[queue as usize].as_mut() {
            let ps = &mut pools.slots[slot];
            ps.next = 0;
            ps.submits.clear();
        }
    }

    /// The most recently retired slot's timings.
    pub fn latest(&self) -> FrameGpuTimings {
        self.latest.clone()
    }

    /// Destroy every query pool. Call before the logical device is destroyed.
    pub fn destroy(&mut self, core: &ash::Device) {
        for pools in self.queues.iter_mut().flatten() {
            for slot in &pools.slots {
                unsafe { core.destroy_query_pool(slot.pool, None) };
            }
        }
    }
}

/// Bitmask keeping only a family's valid timestamp bits.
fn valid_bits_mask(valid_bits: u32) -> u64 {
    if valid_bits >= 64 {
        u64::MAX
    } else {
        (1u64 << valid_bits) - 1
    }
}

/// Milliseconds between two query indices, masking to valid bits and handling
/// counter wrap. Returns 0.0 when either index was clamped (unavailable).
fn duration_ms(
    ticks: &[u64],
    begin: Option<u32>,
    end: Option<u32>,
    mask: u64,
    period_ns: f32,
) -> f32 {
    let (Some(b), Some(e)) = (begin, end) else {
        return 0.0;
    };
    let (b, e) = (b as usize, e as usize);
    if b >= ticks.len() || e >= ticks.len() {
        return 0.0;
    }
    let delta = (ticks[e] & mask).wrapping_sub(ticks[b] & mask) & mask;
    let ms = delta as f64 * period_ns as f64 / 1_000_000.0;
    (ms as f32).max(0.0)
}
