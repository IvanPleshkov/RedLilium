//! Transparent BLAS compaction driver (#110 phase 3, ADR-032).
//!
//! Compaction is a multi-frame dance and this driver owns the graphics-side
//! choreography (the GPU query lives in the Vulkan backend). Its two entry
//! points run at different times:
//!
//! - [`flush`](AsCompactionDriver::flush) — driven by the per-frame
//!   maintenance flush ([`RenderGraph::flush_acceleration_structure_compaction`]).
//!   For every BLAS whose compacted size has come back it allocates the smaller
//!   backing, creates the compacted handle, and adds a
//!   `vkCmdCopyAccelerationStructureKHR(COMPACT)` to the frame's graph. It does
//!   **not** swap the BLAS yet — the copy is still in flight.
//! - [`advance`](AsCompactionDriver::advance) — driven once per frame from
//!   [`GraphicsDevice::advance_frame`], so its clock is real frame completion
//!   regardless of how often the flush runs. It swaps a BLAS onto its compacted
//!   structure only once the copy is guaranteed done, then holds the original
//!   for one more frames-in-flight window — in-flight TLAS instance buffers may
//!   still carry the original's device address.
//!
//! The window is [`MAX_FRAMES_IN_FLIGHT`](crate::pipeline::MAX_FRAMES_IN_FLIGHT)
//! frames, the conservative compile-time bound: waiting longer than the runtime
//! frames-in-flight is always safe, so no resource is ever freed while the GPU
//! (or an in-flight TLAS instance buffer) might still read it.
//!
//! [`RenderGraph::flush_acceleration_structure_compaction`]: crate::RenderGraph::flush_acceleration_structure_compaction
//! [`GraphicsDevice::advance_frame`]: crate::device::GraphicsDevice

use std::sync::{Arc, Weak};

use crate::device::GraphicsDevice;
use crate::graph::{AccelerationStructureBuildPass, RenderGraph};
use crate::resources::{Blas, BlasBacking, CompactionCopy, CompactionPhase};
use crate::types::{BufferDescriptor, BufferUsage};

/// Frames a resource must outlive its last possible in-flight use before it can
/// be freed — one frames-in-flight window, taken as the compile-time maximum
/// (waiting longer than the runtime value is always safe).
const RETIRE_DELAY: u64 = crate::pipeline::MAX_FRAMES_IN_FLIGHT as u64;

/// A compaction copy encoded during a flush; the BLAS is swapped onto the
/// compacted structure once `swap_at` is guaranteed complete.
struct PendingSwap {
    blas: Arc<Blas>,
    new: BlasBacking,
    swap_at: u64,
}

/// Graphics-side driver for transparent BLAS compaction (#110 phase 3).
#[derive(Default)]
pub(crate) struct AsCompactionDriver {
    /// Compactable BLASes, registered at creation. `Weak` so a BLAS the user
    /// dropped falls out here and is freed by its own `Drop`.
    tracked: Vec<Weak<Blas>>,
    /// Copies awaiting their swap.
    pending_swaps: Vec<PendingSwap>,
    /// Original (backing, handle) pairs held until in-flight TLAS instance
    /// buffers that captured their device address are guaranteed done:
    /// `(pair, drop_at_frame)`.
    retiring: Vec<(BlasBacking, u64)>,
}

impl AsCompactionDriver {
    /// Register a compactable BLAS at creation so the flush can find it once
    /// its compacted size comes back.
    pub(crate) fn register(&mut self, blas: &Arc<Blas>) {
        self.tracked.push(Arc::downgrade(blas));
    }

    /// Encode a compaction copy for every tracked BLAS whose compacted size is
    /// ready, adding them to `graph` as one build pass (#110 phase 3).
    pub(crate) fn flush(
        &mut self,
        device: &Arc<GraphicsDevice>,
        graph: &mut RenderGraph,
        frame_index: u64,
    ) {
        let mut pass: Option<AccelerationStructureBuildPass> = None;
        self.tracked.retain(|weak| {
            let Some(blas) = weak.upgrade() else {
                return false; // dropped — stop tracking
            };
            let Some(compaction) = blas.compaction() else {
                return true;
            };
            if let CompactionPhase::SizeReady(size) = compaction.phase()
                && let Some((copy, new)) = Self::allocate_compacted(device, &blas, size)
            {
                pass.get_or_insert_with(|| {
                    AccelerationStructureBuildPass::new("blas compaction".into())
                })
                .add_compaction_copy(copy);
                let swap_at = frame_index + RETIRE_DELAY;
                compaction.set_phase(CompactionPhase::Copying { swap_at });
                self.pending_swaps.push(PendingSwap {
                    blas: Arc::clone(&blas),
                    new,
                    swap_at,
                });
            }
            true
        });
        if let Some(pass) = pass {
            graph.add_acceleration_structure_build_pass(pass);
        }
    }

    /// Allocate the compacted backing + handle and assemble the copy. Returns
    /// `None` (leaving the BLAS in `SizeReady` to retry) if allocation fails.
    fn allocate_compacted(
        device: &Arc<GraphicsDevice>,
        blas: &Arc<Blas>,
        size: u64,
    ) -> Option<(CompactionCopy, BlasBacking)> {
        let label = blas.label().unwrap_or("blas").to_string();
        let backing = device
            .create_buffer(
                &BufferDescriptor::new(size, BufferUsage::ACCELERATION_STRUCTURE_STORAGE)
                    .with_label(format!("{label} compacted")),
            )
            .map_err(|e| log::warn!("BLAS compaction backing alloc failed: {e}"))
            .ok()?;
        let handle = device
            .instance()
            .backend()
            .create_blas_handle(backing.gpu_handle(), size)
            .map_err(|e| log::warn!("BLAS compaction handle create failed: {e}"))
            .ok()?;
        let old = blas.backing_snapshot();
        let original_size = old.backing.size();
        log::info!(
            "BLAS '{label}' compacting: {original_size} -> {size} bytes ({:.0}% smaller)",
            100.0 * (1.0 - size as f64 / original_size.max(1) as f64),
        );
        let new = BlasBacking {
            backing: Arc::clone(&backing),
            gpu_handle: Arc::new(handle),
        };
        let copy = CompactionCopy {
            src_handle: old.gpu_handle,
            dst_handle: Arc::clone(&new.gpu_handle),
            src_backing: old.backing,
            dst_backing: backing,
        };
        Some((copy, new))
    }

    /// Advance the swap/retire lifecycle to `frame_index` (#110 phase 3).
    /// Called once per frame from [`GraphicsDevice::advance_frame`].
    pub(crate) fn advance(&mut self, frame_index: u64) {
        // Swap in compacted structures whose copy is guaranteed complete; the
        // original enters deferred retirement (in-flight instance buffers may
        // still reference its device address until the CPU's next writes).
        let mut i = 0;
        while i < self.pending_swaps.len() {
            if frame_index >= self.pending_swaps[i].swap_at {
                let swap = self.pending_swaps.remove(i);
                let old = swap.blas.swap_backing(swap.new);
                if let Some(compaction) = swap.blas.compaction() {
                    compaction.set_phase(CompactionPhase::Done);
                }
                self.retiring.push((old, frame_index + RETIRE_DELAY));
            } else {
                i += 1;
            }
        }
        // Free originals whose retirement window has passed (dropping the last
        // Arc destroys the buffer and the acceleration structure).
        self.retiring.retain(|(_, drop_at)| frame_index < *drop_at);
    }

    /// Number of copies awaiting a swap plus originals awaiting retirement — for
    /// tests and diagnostics.
    #[cfg(test)]
    pub(crate) fn in_flight(&self) -> usize {
        self.pending_swaps.len() + self.retiring.len()
    }

    /// Number of BLASes still tracked (for tests).
    #[cfg(test)]
    pub(crate) fn tracked_count(&self) -> usize {
        self.tracked.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::resource_usage::BufferAccessMode;
    use crate::graph::{AccelerationStructureBuild, Pass};
    use crate::instance::{BackendType, GraphicsInstance, InstanceParameters};
    use crate::mesh::IndexFormat;
    use crate::resources::{BlasDescriptor, BlasTriangles};

    fn dummy_device() -> Arc<GraphicsDevice> {
        GraphicsInstance::with_parameters(
            InstanceParameters::new().with_backend(BackendType::Dummy),
        )
        .unwrap()
        .create_device()
        .unwrap()
    }

    fn compactable_blas(device: &Arc<GraphicsDevice>) -> Arc<Blas> {
        let vertex_buffer = device
            .create_buffer(&BufferDescriptor::new(
                36,
                BufferUsage::ACCELERATION_STRUCTURE_INPUT,
            ))
            .unwrap();
        device
            .create_blas(
                &BlasDescriptor::new(vec![BlasTriangles {
                    vertex_buffer,
                    vertex_offset: 0,
                    vertex_stride: 12,
                    vertex_count: 3,
                    index_buffer: None,
                    index_offset: 0,
                    index_format: IndexFormat::Uint32,
                    triangle_count: 1,
                    opaque: true,
                }])
                .with_label("test")
                .with_compaction(),
            )
            .unwrap()
    }

    /// The compaction copy pass declares an AS read of the original backing and
    /// an AS write of the compacted one — the two the automatic barrier system
    /// keys on (#110 phase 3).
    #[test]
    fn flush_encodes_copy_and_enters_copying() {
        let device = dummy_device();
        let mut driver = AsCompactionDriver::default();
        let blas = compactable_blas(&device);
        driver.register(&blas);
        blas.compaction()
            .unwrap()
            .set_phase(CompactionPhase::SizeReady(64));

        let mut graph = RenderGraph::new();
        driver.flush(&device, &mut graph, 0);

        // Exactly one compaction-copy pass was added.
        let pass = graph
            .passes()
            .iter()
            .find_map(|p| match p {
                Pass::AccelerationStructureBuild(p) => Some(p),
                _ => None,
            })
            .expect("a compaction build pass");
        assert!(matches!(
            pass.builds(),
            [AccelerationStructureBuild::Compact(_)]
        ));
        let usage = pass.infer_resource_usage();
        assert_eq!(
            usage
                .buffer_usages
                .iter()
                .filter(|u| u.access == BufferAccessMode::AccelerationStructureBuildRead)
                .count(),
            1
        );
        assert_eq!(
            usage
                .buffer_usages
                .iter()
                .filter(|u| u.access == BufferAccessMode::AccelerationStructureWrite)
                .count(),
            1
        );

        assert!(matches!(
            blas.compaction().unwrap().phase(),
            CompactionPhase::Copying { .. }
        ));
    }

    /// The swap is deferred a full frames-in-flight window, then the original is
    /// held one more window before being freed (#110 phase 3).
    #[test]
    fn advance_defers_swap_then_retires_original() {
        let device = dummy_device();
        let mut driver = AsCompactionDriver::default();
        let blas = compactable_blas(&device);
        driver.register(&blas);
        blas.compaction()
            .unwrap()
            .set_phase(CompactionPhase::SizeReady(64));

        let mut graph = RenderGraph::new();
        driver.flush(&device, &mut graph, 0);
        let original = blas.gpu_handle();

        // Before the window elapses: no swap.
        driver.advance(RETIRE_DELAY - 1);
        assert!(matches!(
            blas.compaction().unwrap().phase(),
            CompactionPhase::Copying { .. }
        ));
        assert!(Arc::ptr_eq(&original, &blas.gpu_handle()));

        // Window elapsed: swap onto the compacted structure, original retiring
        // (in-flight TLAS instance buffers may still hold its address).
        driver.advance(RETIRE_DELAY);
        assert_eq!(blas.compaction().unwrap().phase(), CompactionPhase::Done);
        assert!(!Arc::ptr_eq(&original, &blas.gpu_handle()));
        assert_eq!(driver.in_flight(), 1);

        // One more window: the original is released from the driver.
        driver.advance(RETIRE_DELAY * 2);
        assert_eq!(driver.in_flight(), 0);
    }

    /// A BLAS the user dropped falls out of tracking at the next flush without
    /// panicking (Weak handles).
    #[test]
    fn dropped_blas_falls_out_of_tracking() {
        let device = dummy_device();
        let mut driver = AsCompactionDriver::default();
        let blas = compactable_blas(&device);
        driver.register(&blas);
        assert_eq!(driver.tracked_count(), 1);

        drop(blas);
        let mut graph = RenderGraph::new();
        driver.flush(&device, &mut graph, 0);
        assert_eq!(driver.tracked_count(), 0);
    }
}
