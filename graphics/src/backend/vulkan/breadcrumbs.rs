//! GPU crash breadcrumbs for the Vulkan backend (#97).
//!
//! Post-mortem diagnostics for `VK_ERROR_DEVICE_LOST`: a marker is written
//! around every pass so a hang/crash can report *which pass the GPU died in*
//! instead of a bare `DeviceLost`. Granularity is per-pass, never per-draw
//! (tens of markers per frame — unmeasurable overhead).
//!
//! Three mechanisms behind one interface, picked at backend creation by
//! extension availability:
//!
//! 1. [`Mechanism::NvCheckpoints`] — `VK_NV_device_diagnostic_checkpoints`:
//!    `vkCmdSetCheckpointNV` per marker; on device loss
//!    `vkGetQueueCheckpointDataNV` yields the last checkpoints reached per
//!    queue. No buffer needed (the checkpoint marker pointer carries the code).
//! 2. [`Mechanism::AmdBufferMarker`] — `VK_AMD_buffer_marker`:
//!    `vkCmdWriteBufferMarkerAMD` writes the code into a persistently mapped
//!    host-visible buffer (one region per queue per frame slot).
//! 3. [`Mechanism::Fallback`] — portable `vkCmdFillBuffer` writes into the same
//!    kind of buffer. Coarser guarantees (fill ordering is not tied to the
//!    pass's pipeline work) but still narrows the kill site; always available.
//!
//! All three encode the same [`marker code`](pack_code): a `u32` packing
//! `(slot, submit_seq, pass-or-bracket, begin/end)`. Readback — a buffer scan
//! or the NV checkpoint list — yields a set of *reached* codes, and one pure
//! [`diagnose_submit`] function turns a CPU-side trace plus that set into
//! "last completed / first incomplete pass", identically for every mechanism.
//!
//! `VK_EXT_device_fault` is orthogonal: where present, its structured fault
//! report is appended to the crash log.

use std::ffi::CStr;

use ash::vk;
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator};

use crate::graph::QueuePreference;

use super::MAX_FRAMES_IN_FLIGHT;
use super::barriers::{QUEUE_COUNT, QueueId};

/// `u32` marker slots per (queue, slot) region for the buffer mechanisms. Two
/// per submit bracket plus two per pass fits ~511 passes per slot-cycle; a
/// frame with more clamps (drops the extras, warns once) — never asserts.
const MARKERS_PER_POOL: u32 = 1024;

/// Bit layout of a marker code (`u32`), chosen so `submit_seq >= 1` makes every
/// code non-zero (0 = "not reached" when scanning a zero-initialized buffer).
const POB_SHIFT: u32 = 1; // bit 0 = begin(0)/end(1)
const SEQ_SHIFT: u32 = 18; // bits 1..18  = pass-or-bracket (0 = submit bracket, else pass+1)
const SLOT_SHIFT: u32 = 30; // bits 18..30 = submit_seq (1-based); bits 30..32 = slot

/// Which mechanism the backend selected at creation. Priority when several are
/// available: NV → AMD → portable fallback.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mechanism {
    NvCheckpoints,
    AmdBufferMarker,
    Fallback,
}

impl Mechanism {
    /// Human label for the startup "breadcrumbs active" log line.
    pub fn label(self) -> &'static str {
        match self {
            Self::NvCheckpoints => "VK_NV_device_diagnostic_checkpoints",
            Self::AmdBufferMarker => "VK_AMD_buffer_marker",
            Self::Fallback => "portable vkCmdFillBuffer fallback",
        }
    }
}

/// One of the four marker positions a submit records.
#[derive(Debug, Clone, Copy)]
enum MarkerKind {
    SubmitBegin,
    SubmitEnd,
    PassBegin(usize),
    PassEnd(usize),
}

impl MarkerKind {
    /// `(pass-or-bracket, end)` for [`pack_code`]: bracket is 0, pass `i` is
    /// `i + 1`.
    fn pob_end(self) -> (u32, bool) {
        match self {
            Self::SubmitBegin => (0, false),
            Self::SubmitEnd => (0, true),
            Self::PassBegin(i) => (i as u32 + 1, false),
            Self::PassEnd(i) => (i as u32 + 1, true),
        }
    }
}

/// Pack a marker code. `submit_seq` is 1-based (per slot-cycle), so the result
/// is never 0.
fn pack_code(slot: usize, submit_seq: u32, pob: u32, end: bool) -> u32 {
    ((slot as u32) << SLOT_SHIFT)
        | ((submit_seq & 0xFFF) << SEQ_SHIFT)
        | ((pob & 0x1_FFFF) << POB_SHIFT)
        | (end as u32)
}

/// The sync1 pipeline stage for a buffer marker: begin markers record once
/// prior work reaches the top of the pipe, end markers once it reaches the
/// bottom.
fn marker_stage(end: bool) -> vk::PipelineStageFlags {
    if end {
        vk::PipelineStageFlags::BOTTOM_OF_PIPE
    } else {
        vk::PipelineStageFlags::TOP_OF_PIPE
    }
}

/// A host-visible, persistently mapped `u32` buffer holding one (queue, slot)
/// region's markers (AMD / fallback mechanisms only).
struct MarkerBuffer {
    buffer: vk::Buffer,
    allocation: Option<Allocation>,
}

impl MarkerBuffer {
    fn create(
        device: &ash::Device,
        allocator: &mut Allocator,
    ) -> Result<Self, crate::error::GraphicsError> {
        use crate::error::GraphicsError;
        let size = (MARKERS_PER_POOL * 4) as u64;
        let info = vk::BufferCreateInfo::default()
            .size(size)
            .usage(vk::BufferUsageFlags::TRANSFER_DST)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = unsafe { device.create_buffer(&info, None) }.map_err(|e| {
            GraphicsError::ResourceCreationFailed(format!("breadcrumb buffer: {e:?}"))
        })?;
        let requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
        // CpuToGpu = host-visible + coherent: GPU marker writes are visible to
        // the CPU without an explicit invalidate, which matters on the device-
        // loss path where no fence is available to gate the read.
        let allocation = allocator
            .allocate(&AllocationCreateDesc {
                name: "gpu_breadcrumb_markers",
                requirements,
                location: gpu_allocator::MemoryLocation::CpuToGpu,
                linear: true,
                allocation_scheme: AllocationScheme::GpuAllocatorManaged,
            })
            .map_err(|e| {
                unsafe { device.destroy_buffer(buffer, None) };
                GraphicsError::ResourceCreationFailed(format!("breadcrumb memory: {e}"))
            })?;
        if let Err(e) =
            unsafe { device.bind_buffer_memory(buffer, allocation.memory(), allocation.offset()) }
        {
            let _ = allocator.free(allocation);
            unsafe { device.destroy_buffer(buffer, None) };
            return Err(GraphicsError::ResourceCreationFailed(format!(
                "breadcrumb bind: {e:?}"
            )));
        }
        Ok(Self {
            buffer,
            allocation: Some(allocation),
        })
    }

    /// The current marker values (host-visible read). Empty if unmapped.
    fn read(&self) -> Vec<u32> {
        let Some(ptr) = self.allocation.as_ref().and_then(|a| a.mapped_ptr()) else {
            return Vec::new();
        };
        // SAFETY: the buffer is MARKERS_PER_POOL `u32`s, persistently mapped and
        // host-coherent; we only read.
        let slice = unsafe {
            std::slice::from_raw_parts(ptr.as_ptr() as *const u32, MARKERS_PER_POOL as usize)
        };
        slice.to_vec()
    }

    fn destroy(&mut self, device: &ash::Device, allocator: &mut Allocator) {
        if let Some(allocation) = self.allocation.take() {
            let _ = allocator.free(allocation);
        }
        unsafe { device.destroy_buffer(self.buffer, None) };
    }
}

/// CPU-side record of one recorded submit: enough to re-derive its marker codes
/// and name the passes at report time.
#[derive(Clone)]
struct SubmitTrace {
    slot: usize,
    submit_seq: u32,
    timeline_value: u64,
    passes: Vec<String>,
}

/// Per (queue, slot) breadcrumb bookkeeping.
struct PoolSlot {
    /// The marker buffer (AMD / fallback); `None` for NV checkpoints.
    buffer: Option<MarkerBuffer>,
    /// Next free marker index in the buffer; 0 also means "reset the buffer
    /// before the next write" (start of a slot-cycle).
    next: u32,
    /// 1-based submit counter within the current slot-cycle.
    submit_seq: u32,
    /// Traces for the submits recorded since the last reset (still in flight).
    traces: Vec<SubmitTrace>,
}

/// All per-slot state for one queue, plus the queue handle (for NV checkpoint
/// readback) and its engine-facing preference (for the report).
struct QueuePools {
    preference: QueuePreference,
    queue: vk::Queue,
    slots: Vec<PoolSlot>,
}

/// A queue eligible for breadcrumbs: its id, engine-facing preference, and the
/// actual queue handle.
pub struct QueueBreadcrumbInfo {
    pub queue: QueueId,
    pub preference: QueuePreference,
    pub handle: vk::Queue,
}

/// How a submit writes its markers — the mechanism plus the reserved buffer
/// offsets (buffer mechanisms only). Held on the render thread's stack across
/// pass encoding so no manager lock is taken per pass.
///
/// The `Fallback` variant embeds a full `ash::Device` (a large fn table), so
/// the variants differ in size — but this is a short-lived, one-per-submit
/// stack value, never stored in a collection, so boxing it would only add a
/// per-submit allocation to satisfy a lint aimed at long-lived memory waste.
#[allow(clippy::large_enum_variant)]
enum Emitter {
    Nv(ash::nv::device_diagnostic_checkpoints::Device),
    Amd(ash::amd::buffer_marker::Device, BufferMarks),
    Fallback(ash::Device, BufferMarks),
}

/// Reserved buffer offsets for a submit's markers (buffer mechanisms).
struct BufferMarks {
    buffer: vk::Buffer,
    submit_begin: Option<u32>,
    submit_end: Option<u32>,
    /// `(begin, end)` offset per pass; `None` when the pool overflowed.
    passes: Vec<(Option<u32>, Option<u32>)>,
}

impl BufferMarks {
    fn offset(&self, kind: MarkerKind) -> Option<u32> {
        match kind {
            MarkerKind::SubmitBegin => self.submit_begin,
            MarkerKind::SubmitEnd => self.submit_end,
            MarkerKind::PassBegin(i) => self.passes.get(i).and_then(|p| p.0),
            MarkerKind::PassEnd(i) => self.passes.get(i).and_then(|p| p.1),
        }
    }
}

/// A submit currently recording breadcrumbs. Detached from the manager so the
/// per-pass marker writes need no lock.
pub struct SubmitBreadcrumbs {
    queue: QueueId,
    slot: usize,
    submit_seq: u32,
    timeline_value: u64,
    emitter: Emitter,
    /// Pass names, filled during encode and moved into the trace at finish.
    pass_names: Vec<String>,
}

impl SubmitBreadcrumbs {
    /// Emit one marker onto `cmd`.
    fn emit(&self, cmd: vk::CommandBuffer, kind: MarkerKind) {
        let (pob, end) = kind.pob_end();
        let code = pack_code(self.slot, self.submit_seq, pob, end);
        match &self.emitter {
            Emitter::Nv(dev) => unsafe {
                dev.cmd_set_checkpoint(cmd, code as usize as *const std::ffi::c_void);
            },
            Emitter::Amd(dev, marks) => {
                if let Some(off) = marks.offset(kind) {
                    unsafe {
                        dev.cmd_write_buffer_marker(
                            cmd,
                            marker_stage(end),
                            marks.buffer,
                            off as u64 * 4,
                            code,
                        );
                    }
                }
            }
            Emitter::Fallback(dev, marks) => {
                if let Some(off) = marks.offset(kind) {
                    unsafe { dev.cmd_fill_buffer(cmd, marks.buffer, off as u64 * 4, 4, code) };
                }
            }
        }
    }

    /// Record the begin marker for pass `i` and remember its name.
    pub fn pass_begin(&mut self, cmd: vk::CommandBuffer, i: usize, name: &str) {
        if let Some(slot) = self.pass_names.get_mut(i) {
            slot.clear();
            slot.push_str(name);
        }
        self.emit(cmd, MarkerKind::PassBegin(i));
    }

    /// Record the end marker for pass `i`.
    pub fn pass_end(&self, cmd: vk::CommandBuffer, i: usize) {
        self.emit(cmd, MarkerKind::PassEnd(i));
    }

    /// Record the whole-submit end marker (after the last pass).
    pub fn submit_end(&self, cmd: vk::CommandBuffer) {
        self.emit(cmd, MarkerKind::SubmitEnd);
    }
}

/// Per-pass GPU crash breadcrumb collector — one per Vulkan backend, created
/// only when breadcrumbs are enabled (#97).
pub struct BreadcrumbManager {
    mechanism: Mechanism,
    device: ash::Device,
    checkpoints: Option<ash::nv::device_diagnostic_checkpoints::Device>,
    buffer_marker: Option<ash::amd::buffer_marker::Device>,
    device_fault: Option<ash::ext::device_fault::Device>,
    queues: [Option<QueuePools>; QUEUE_COUNT],
    overflow_warned: bool,
}

impl BreadcrumbManager {
    /// Create the manager for the selected mechanism. Allocates one marker
    /// buffer per (queue, slot) for the buffer mechanisms; NV needs none.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &ash::Device,
        allocator: &mut Allocator,
        mechanism: Mechanism,
        checkpoints: Option<ash::nv::device_diagnostic_checkpoints::Device>,
        buffer_marker: Option<ash::amd::buffer_marker::Device>,
        device_fault: Option<ash::ext::device_fault::Device>,
        queue_infos: &[QueueBreadcrumbInfo],
    ) -> Option<Self> {
        let mut queues: [Option<QueuePools>; QUEUE_COUNT] = Default::default();
        for info in queue_infos {
            let mut slots: Vec<PoolSlot> = Vec::with_capacity(MAX_FRAMES_IN_FLIGHT);
            let mut ok = true;
            for _ in 0..MAX_FRAMES_IN_FLIGHT {
                let buffer = if mechanism == Mechanism::NvCheckpoints {
                    None
                } else {
                    match MarkerBuffer::create(device, allocator) {
                        Ok(b) => Some(b),
                        Err(e) => {
                            log::warn!(
                                "breadcrumb buffer creation failed: {e}; disabling for {:?}",
                                info.queue
                            );
                            for slot in &mut slots {
                                if let Some(b) = slot.buffer.as_mut() {
                                    b.destroy(device, allocator);
                                }
                            }
                            ok = false;
                            break;
                        }
                    }
                };
                slots.push(PoolSlot {
                    buffer,
                    next: 0,
                    submit_seq: 0,
                    traces: Vec::new(),
                });
            }
            if !ok {
                continue;
            }
            queues[info.queue as usize] = Some(QueuePools {
                preference: info.preference,
                queue: info.handle,
                slots,
            });
        }

        if queues.iter().all(Option::is_none) {
            return None;
        }
        Some(Self {
            mechanism,
            device: device.clone(),
            checkpoints,
            buffer_marker,
            device_fault,
            queues,
            overflow_warned: false,
        })
    }

    /// Reserve markers for a submit of `num_passes` passes on `queue`/`slot`,
    /// recording the buffer reset (if pending) and the submit-begin marker onto
    /// `cmd`. Returns `None` when this queue has no breadcrumbs.
    pub fn begin_submit(
        &mut self,
        queue: QueueId,
        slot: usize,
        num_passes: usize,
        timeline_value: u64,
        cmd: vk::CommandBuffer,
    ) -> Option<SubmitBreadcrumbs> {
        let mechanism = self.mechanism;
        let pools = self.queues[queue as usize].as_mut()?;
        let ps = &mut pools.slots[slot];

        // First write of a slot-cycle (`next == 0`): reset. For the buffer
        // mechanisms, zero the region and barrier it before the marker writes so
        // the clear cannot clobber a marker.
        if ps.next == 0 {
            ps.submits_reset();
            if let Some(b) = ps.buffer.as_ref() {
                unsafe {
                    self.device
                        .cmd_fill_buffer(cmd, b.buffer, 0, (MARKERS_PER_POOL * 4) as u64, 0);
                    let barrier = vk::MemoryBarrier2::default()
                        .src_stage_mask(vk::PipelineStageFlags2::ALL_TRANSFER)
                        .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                        .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                        .dst_access_mask(vk::AccessFlags2::MEMORY_WRITE);
                    let barriers = [barrier];
                    let dep = vk::DependencyInfo::default().memory_barriers(&barriers);
                    self.device.cmd_pipeline_barrier2(cmd, &dep);
                }
            }
        }

        ps.submit_seq += 1;
        let submit_seq = ps.submit_seq;

        // Reserve buffer offsets (buffer mechanisms only).
        let mut overflow = false;
        let mut alloc = |ps: &mut PoolSlot| -> Option<u32> {
            ps.buffer.as_ref()?;
            if ps.next >= MARKERS_PER_POOL {
                overflow = true;
                None
            } else {
                let q = ps.next;
                ps.next += 1;
                Some(q)
            }
        };
        let (submit_begin, submit_end, pass_offsets) = if mechanism == Mechanism::NvCheckpoints {
            // NV writes no buffer, but `next` must leave 0 so the reset trigger
            // only fires once per slot-cycle.
            ps.next = ps.next.max(1);
            (None, None, Vec::new())
        } else {
            let sb = alloc(ps);
            let se = alloc(ps);
            let mut passes = Vec::with_capacity(num_passes);
            for _ in 0..num_passes {
                let b = alloc(ps);
                let e = alloc(ps);
                passes.push((b, e));
            }
            (sb, se, passes)
        };

        if overflow && !self.overflow_warned {
            self.overflow_warned = true;
            log::warn!(
                "GPU breadcrumb pool exhausted ({MARKERS_PER_POOL} markers/slot); \
                 some pass breadcrumbs dropped this frame (#97)"
            );
        }

        let emitter = match mechanism {
            Mechanism::NvCheckpoints => Emitter::Nv(
                self.checkpoints
                    .clone()
                    .expect("NV mechanism without checkpoints device"),
            ),
            Mechanism::AmdBufferMarker => Emitter::Amd(
                self.buffer_marker
                    .clone()
                    .expect("AMD mechanism without buffer_marker device"),
                BufferMarks {
                    buffer: ps.buffer.as_ref().map_or(vk::Buffer::null(), |b| b.buffer),
                    submit_begin,
                    submit_end,
                    passes: pass_offsets,
                },
            ),
            Mechanism::Fallback => Emitter::Fallback(
                self.device.clone(),
                BufferMarks {
                    buffer: ps.buffer.as_ref().map_or(vk::Buffer::null(), |b| b.buffer),
                    submit_begin,
                    submit_end,
                    passes: pass_offsets,
                },
            ),
        };

        let rec = SubmitBreadcrumbs {
            queue,
            slot,
            submit_seq,
            timeline_value,
            emitter,
            pass_names: vec![String::new(); num_passes],
        };
        rec.emit(cmd, MarkerKind::SubmitBegin);
        Some(rec)
    }

    /// Hand a finished recording back so its trace survives for the reporter.
    pub fn finish_submit(&mut self, rec: SubmitBreadcrumbs) {
        let Some(pools) = self.queues[rec.queue as usize].as_mut() else {
            return;
        };
        pools.slots[rec.slot].traces.push(SubmitTrace {
            slot: rec.slot,
            submit_seq: rec.submit_seq,
            timeline_value: rec.timeline_value,
            passes: rec.pass_names,
        });
    }

    /// Drop an in-progress submit's bookkeeping after a recording/submit error
    /// (its command buffer never ran). Re-arms the slot's reset.
    pub fn abort_submit(&mut self, queue: QueueId, slot: usize) {
        if let Some(pools) = self.queues[queue as usize].as_mut() {
            pools.slots[slot].submits_reset();
        }
    }

    /// Retire `slot`: its fence has signaled, so its work completed and its
    /// breadcrumbs are no longer interesting. Re-arms the reset. Called from
    /// `advance_frame`.
    pub fn retire_slot(&mut self, slot: usize) {
        for pools in self.queues.iter_mut().flatten() {
            pools.slots[slot].submits_reset();
        }
    }

    /// Read every in-flight queue's markers and format a post-mortem report.
    /// Called once when a `VK_ERROR_DEVICE_LOST` is first observed.
    pub fn collect_report(&self, adapter: &crate::instance::AdapterInfo) -> String {
        let mut out = String::new();
        out.push_str("=== RedLilium GPU crash breadcrumbs (#97) ===\n");
        out.push_str(&format!(
            "adapter: {} ({}, {:04x}:{:04x})\n",
            adapter.name, adapter.vendor, adapter.vendor_id, adapter.device_id
        ));
        out.push_str(&format!("mechanism: {}\n", self.mechanism.label()));

        for pools in self.queues.iter().flatten() {
            let reached = self.reached_codes(pools);
            let traces: Vec<&SubmitTrace> =
                pools.slots.iter().flat_map(|s| s.traces.iter()).collect();
            out.push_str(&format!("\n[queue {:?}]\n", pools.preference));
            if traces.is_empty() {
                out.push_str("  (no submits in flight)\n");
                continue;
            }
            for trace in traces {
                let d = diagnose_submit(trace, &reached);
                out.push_str(&format_diagnosis(trace, &d));
            }
        }

        if let Some(fault) = self.read_device_fault() {
            out.push_str("\n[VK_EXT_device_fault]\n");
            out.push_str(&fault);
        }
        out.push_str("=== end breadcrumbs ===\n");
        out
    }

    /// The set of reached marker codes for one queue.
    fn reached_codes(&self, pools: &QueuePools) -> std::collections::HashSet<u32> {
        let mut reached = std::collections::HashSet::new();
        match self.mechanism {
            Mechanism::NvCheckpoints => {
                if let Some(cp) = &self.checkpoints {
                    unsafe {
                        let len = cp.get_queue_checkpoint_data_len(pools.queue);
                        let mut data = vec![vk::CheckpointDataNV::default(); len];
                        if len > 0 {
                            cp.get_queue_checkpoint_data(pools.queue, &mut data);
                        }
                        for d in data {
                            reached.insert(d.p_checkpoint_marker as usize as u32);
                        }
                    }
                }
            }
            Mechanism::AmdBufferMarker | Mechanism::Fallback => {
                for slot in &pools.slots {
                    if let Some(b) = &slot.buffer {
                        for value in b.read() {
                            if value != 0 {
                                reached.insert(value);
                            }
                        }
                    }
                }
            }
        }
        reached
    }

    /// Query `VK_EXT_device_fault` and format its description, if available.
    fn read_device_fault(&self) -> Option<String> {
        let fault = self.device_fault.as_ref()?;
        unsafe {
            let mut counts = vk::DeviceFaultCountsEXT::default();
            let fp = fault.fp().get_device_fault_info_ext;
            let r = fp(fault.device(), &mut counts, std::ptr::null_mut());
            if r != vk::Result::SUCCESS {
                return None;
            }
            let mut info = vk::DeviceFaultInfoEXT::default();
            let r = fp(fault.device(), &mut counts, &mut info);
            if r != vk::Result::SUCCESS {
                return None;
            }
            let desc = CStr::from_ptr(info.description.as_ptr()).to_string_lossy();
            Some(format!(
                "  description: {desc}\n  address faults: {}, vendor infos: {}\n",
                counts.address_info_count, counts.vendor_info_count
            ))
        }
    }

    /// Destroy all marker buffers. Call before the logical device is destroyed.
    pub fn destroy(&mut self, device: &ash::Device, allocator: &mut Allocator) {
        for pools in self.queues.iter_mut().flatten() {
            for slot in &mut pools.slots {
                if let Some(b) = slot.buffer.as_mut() {
                    b.destroy(device, allocator);
                }
            }
        }
    }
}

impl PoolSlot {
    /// Re-arm the reset: forget this slot-cycle's submits and start numbering
    /// from scratch. The buffer is zeroed lazily by the next `begin_submit`.
    fn submits_reset(&mut self) {
        self.next = 0;
        self.submit_seq = 0;
        self.traces.clear();
    }
}

/// Which pass a submit reached, resolved from a set of reached marker codes.
/// The single pure function every mechanism's readback feeds into.
#[derive(Debug, PartialEq, Eq)]
pub struct SubmitDiagnosis {
    pub submit_began: bool,
    pub submit_ended: bool,
    /// Highest pass index whose end marker was reached.
    pub last_completed: Option<usize>,
    /// Highest pass index whose begin marker was reached.
    pub last_started: Option<usize>,
}

impl SubmitDiagnosis {
    /// The pass the GPU most likely died in: the last one that started but did
    /// not finish, or (if all started passes finished) the next pass that never
    /// started. `None` when every pass completed.
    pub fn guilty_pass(&self, num_passes: usize) -> Option<usize> {
        match (self.last_started, self.last_completed) {
            (Some(started), completed) if Some(started) != completed => Some(started),
            (_, Some(completed)) if completed + 1 < num_passes => Some(completed + 1),
            (None, None) => (num_passes > 0).then_some(0),
            _ => None,
        }
    }
}

/// Resolve which passes a submit reached (pure — the unit-test target).
fn diagnose_submit(
    trace: &SubmitTrace,
    reached: &std::collections::HashSet<u32>,
) -> SubmitDiagnosis {
    let code = |pob: u32, end: bool| pack_code(trace.slot, trace.submit_seq, pob, end);
    let mut last_started = None;
    let mut last_completed = None;
    for i in 0..trace.passes.len() {
        let pob = i as u32 + 1;
        if reached.contains(&code(pob, false)) {
            last_started = Some(i);
        }
        if reached.contains(&code(pob, true)) {
            last_completed = Some(i);
        }
    }
    SubmitDiagnosis {
        submit_began: reached.contains(&code(0, false)),
        submit_ended: reached.contains(&code(0, true)),
        last_completed,
        last_started,
    }
}

/// Format one submit's diagnosis into the crash report.
fn format_diagnosis(trace: &SubmitTrace, d: &SubmitDiagnosis) -> String {
    let name = |i: Option<usize>| {
        i.and_then(|i| trace.passes.get(i))
            .map_or("<none>", String::as_str)
    };
    if !d.submit_began {
        return format!(
            "  submit (timeline {}): never started on the GPU\n",
            trace.timeline_value
        );
    }
    if d.submit_ended {
        return format!(
            "  submit (timeline {}): completed all {} passes\n",
            trace.timeline_value,
            trace.passes.len()
        );
    }
    let guilty = d.guilty_pass(trace.passes.len());
    format!(
        "  submit (timeline {}): INCOMPLETE\n    last completed pass: {} ({:?})\n    \
         last started pass:   {} ({:?})\n    >>> died in pass:    {} ({:?})\n",
        trace.timeline_value,
        d.last_completed.map_or(-1, |i| i as i64),
        name(d.last_completed),
        d.last_started.map_or(-1, |i| i as i64),
        name(d.last_started),
        guilty.map_or(-1, |i| i as i64),
        name(guilty),
    )
}

/// Write the crash report next to the executable (a hung app often loses its
/// log tail). Returns the path written, for logging.
pub fn write_crash_file(report: &str) -> Option<std::path::PathBuf> {
    use std::time::{SystemTime, UNIX_EPOCH};
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);
    let dir = std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(|p| p.to_path_buf()))
        .unwrap_or_else(|| std::path::PathBuf::from("."));
    let path = dir.join(format!("redlilium-gpu-crash-{millis}.txt"));
    match std::fs::write(&path, report) {
        Ok(()) => Some(path),
        Err(e) => {
            log::error!("failed to write GPU crash file {}: {e}", path.display());
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashSet;

    fn trace(passes: &[&str]) -> SubmitTrace {
        SubmitTrace {
            slot: 1,
            submit_seq: 2,
            timeline_value: 42,
            passes: passes.iter().map(|s| s.to_string()).collect(),
        }
    }

    /// Codes are non-zero and unique per (slot, seq, pass, begin/end) so a
    /// zero-initialized buffer scan never reports a false "reached".
    #[test]
    fn codes_are_nonzero_and_distinct() {
        let mut seen = HashSet::new();
        for slot in 0..MAX_FRAMES_IN_FLIGHT {
            for seq in 1..4u32 {
                for pob in 0..5u32 {
                    for end in [false, true] {
                        let c = pack_code(slot, seq, pob, end);
                        assert_ne!(c, 0);
                        assert!(seen.insert(c), "collision at {slot},{seq},{pob},{end}");
                    }
                }
            }
        }
    }

    /// The GPU finished pass 1's begin but not its end: pass 1 is the guilty
    /// pass, pass 0 is the last completed.
    #[test]
    fn diagnoses_the_incomplete_pass() {
        let t = trace(&["shadow", "gbuffer", "lighting"]);
        let c = |pob, end| pack_code(t.slot, t.submit_seq, pob, end);
        // Submit began; pass 0 completed; pass 1 started but never finished.
        let reached: HashSet<u32> = [
            c(0, false), // submit begin
            c(1, false), // pass 0 begin
            c(1, true),  // pass 0 end
            c(2, false), // pass 1 begin
        ]
        .into_iter()
        .collect();

        let d = diagnose_submit(&t, &reached);
        assert!(d.submit_began);
        assert!(!d.submit_ended);
        assert_eq!(d.last_completed, Some(0));
        assert_eq!(d.last_started, Some(1));
        assert_eq!(d.guilty_pass(t.passes.len()), Some(1));

        let report = format_diagnosis(&t, &d);
        assert!(
            report.contains("gbuffer"),
            "report should name pass 1: {report}"
        );
    }

    /// Every marker reached → the submit completed, no guilty pass.
    #[test]
    fn diagnoses_a_complete_submit() {
        let t = trace(&["only"]);
        let c = |pob, end| pack_code(t.slot, t.submit_seq, pob, end);
        let reached: HashSet<u32> = [c(0, false), c(1, false), c(1, true), c(0, true)]
            .into_iter()
            .collect();
        let d = diagnose_submit(&t, &reached);
        assert!(d.submit_ended);
        assert_eq!(d.guilty_pass(t.passes.len()), None);
    }

    /// No markers reached → the submit never ran; not attributed to a pass.
    #[test]
    fn diagnoses_a_submit_that_never_started() {
        let t = trace(&["a", "b"]);
        let d = diagnose_submit(&t, &HashSet::new());
        assert!(!d.submit_began);
        let report = format_diagnosis(&t, &d);
        assert!(report.contains("never started"), "{report}");
    }
}
