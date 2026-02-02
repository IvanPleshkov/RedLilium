//! Texture layout tracking for automatic barrier placement.
//!
//! This module provides automatic image layout tracking for the Vulkan backend.
//! Instead of manually specifying layout transitions, the system tracks the current
//! layout of each texture and generates barriers automatically based on pass
//! resource usage.
//!
//! # Architecture
//!
//! The system consists of three main components:
//!
//! 1. [`TextureLayout`] - Represents Vulkan image layout states
//! 2. [`TextureUsageGraph`] - Defines valid transitions based on texture usage flags
//! 3. [`TextureLayoutTracker`] - Tracks per-frame layout state and generates barriers
//!
//! # Example
//!
//! ```ignore
//! // The tracker automatically handles layout transitions:
//! // Pass 1: Render to texture (Undefined → ColorAttachment)
//! // Pass 2: Sample texture (ColorAttachment → ShaderReadOnly)
//! // Pass 3: Readback texture (ShaderReadOnly → TransferSrc)
//! ```

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use ash::vk;
use ash::vk::Handle;
use parking_lot::RwLock;

use crate::types::TextureUsage;

/// Number of distinct texture layout states.
const TEXTURE_LAYOUT_COUNT: usize = 9;

/// Vulkan image layout states that textures can be in.
///
/// These correspond to `VkImageLayout` values but are abstracted
/// for use in the layout tracking system.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
#[repr(usize)]
pub enum TextureLayout {
    /// Initial state, contents undefined. Can transition to any layout.
    #[default]
    Undefined = 0,
    /// Optimal for color attachment writes.
    ColorAttachment = 1,
    /// Optimal for depth/stencil attachment writes.
    DepthStencilAttachment = 2,
    /// Optimal for depth read-only (sampling + depth testing).
    DepthStencilReadOnly = 3,
    /// Optimal for shader sampling (texture reads).
    ShaderReadOnly = 4,
    /// Optimal for transfer source operations.
    TransferSrc = 5,
    /// Optimal for transfer destination operations.
    TransferDst = 6,
    /// Optimal for presentation to swapchain.
    PresentSrc = 7,
    /// General layout (least optimal but most flexible).
    General = 8,
}

impl TextureLayout {
    /// Convert to Vulkan image layout.
    pub fn to_vk(self) -> vk::ImageLayout {
        match self {
            Self::Undefined => vk::ImageLayout::UNDEFINED,
            Self::ColorAttachment => vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL,
            Self::DepthStencilAttachment => vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL,
            Self::DepthStencilReadOnly => vk::ImageLayout::DEPTH_STENCIL_READ_ONLY_OPTIMAL,
            Self::ShaderReadOnly => vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            Self::TransferSrc => vk::ImageLayout::TRANSFER_SRC_OPTIMAL,
            Self::TransferDst => vk::ImageLayout::TRANSFER_DST_OPTIMAL,
            Self::PresentSrc => vk::ImageLayout::PRESENT_SRC_KHR,
            Self::General => vk::ImageLayout::GENERAL,
        }
    }

    /// Get the access mask for this layout (as source).
    pub fn src_access_mask(self) -> vk::AccessFlags {
        match self {
            Self::Undefined => vk::AccessFlags::empty(),
            Self::ColorAttachment => vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            Self::DepthStencilAttachment => vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            Self::DepthStencilReadOnly => vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ,
            Self::ShaderReadOnly => vk::AccessFlags::SHADER_READ,
            Self::TransferSrc => vk::AccessFlags::TRANSFER_READ,
            Self::TransferDst => vk::AccessFlags::TRANSFER_WRITE,
            Self::PresentSrc => vk::AccessFlags::empty(),
            Self::General => vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
        }
    }

    /// Get the access mask for this layout (as destination).
    pub fn dst_access_mask(self) -> vk::AccessFlags {
        match self {
            Self::Undefined => vk::AccessFlags::empty(),
            Self::ColorAttachment => vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
            Self::DepthStencilAttachment => vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
            Self::DepthStencilReadOnly => vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ,
            Self::ShaderReadOnly => vk::AccessFlags::SHADER_READ,
            Self::TransferSrc => vk::AccessFlags::TRANSFER_READ,
            Self::TransferDst => vk::AccessFlags::TRANSFER_WRITE,
            Self::PresentSrc => vk::AccessFlags::empty(),
            Self::General => vk::AccessFlags::SHADER_READ | vk::AccessFlags::SHADER_WRITE,
        }
    }

    /// Get the pipeline stage for this layout (as source).
    pub fn src_stage(self) -> vk::PipelineStageFlags {
        match self {
            Self::Undefined => vk::PipelineStageFlags::TOP_OF_PIPE,
            Self::ColorAttachment => vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            Self::DepthStencilAttachment => vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
            Self::DepthStencilReadOnly => vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
            Self::ShaderReadOnly => vk::PipelineStageFlags::FRAGMENT_SHADER,
            Self::TransferSrc => vk::PipelineStageFlags::TRANSFER,
            Self::TransferDst => vk::PipelineStageFlags::TRANSFER,
            Self::PresentSrc => vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            Self::General => vk::PipelineStageFlags::COMPUTE_SHADER,
        }
    }

    /// Get the pipeline stage for this layout (as destination).
    pub fn dst_stage(self) -> vk::PipelineStageFlags {
        match self {
            Self::Undefined => vk::PipelineStageFlags::TOP_OF_PIPE,
            Self::ColorAttachment => vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
            Self::DepthStencilAttachment => vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
            Self::DepthStencilReadOnly => vk::PipelineStageFlags::EARLY_FRAGMENT_TESTS,
            Self::ShaderReadOnly => vk::PipelineStageFlags::FRAGMENT_SHADER,
            Self::TransferSrc => vk::PipelineStageFlags::TRANSFER,
            Self::TransferDst => vk::PipelineStageFlags::TRANSFER,
            Self::PresentSrc => vk::PipelineStageFlags::BOTTOM_OF_PIPE,
            Self::General => vk::PipelineStageFlags::COMPUTE_SHADER,
        }
    }

    /// Check if this is a depth/stencil layout.
    pub fn is_depth_stencil(self) -> bool {
        matches!(
            self,
            Self::DepthStencilAttachment | Self::DepthStencilReadOnly
        )
    }
}

/// Defines valid state transitions for a texture based on its usage flags.
///
/// This is shared via `Arc` between textures with the same usage pattern,
/// avoiding redundant graph allocations.
///
/// The graph is built from [`TextureUsage`] flags and defines which layout
/// transitions are valid for textures with those capabilities.
#[derive(Debug)]
pub struct TextureUsageGraph {
    /// The TextureUsage flags this graph was created for.
    usage: TextureUsage,
    /// Valid transitions from each state.
    /// Indexed by `TextureLayout as usize`.
    transitions: [HashSet<TextureLayout>; TEXTURE_LAYOUT_COUNT],
}

impl TextureUsageGraph {
    /// Create a usage graph from TextureUsage flags.
    pub fn from_usage(usage: TextureUsage) -> Self {
        let mut transitions: [HashSet<TextureLayout>; TEXTURE_LAYOUT_COUNT] =
            std::array::from_fn(|_| HashSet::new());

        // UNDEFINED can transition to any valid destination based on usage
        let undefined_dests = &mut transitions[TextureLayout::Undefined as usize];

        if usage.contains(TextureUsage::RENDER_ATTACHMENT) {
            undefined_dests.insert(TextureLayout::ColorAttachment);
            undefined_dests.insert(TextureLayout::DepthStencilAttachment);
        }
        if usage.contains(TextureUsage::TEXTURE_BINDING) {
            undefined_dests.insert(TextureLayout::ShaderReadOnly);
        }
        if usage.contains(TextureUsage::STORAGE_BINDING) {
            undefined_dests.insert(TextureLayout::General);
        }
        if usage.contains(TextureUsage::COPY_SRC) {
            undefined_dests.insert(TextureLayout::TransferSrc);
        }
        if usage.contains(TextureUsage::COPY_DST) {
            undefined_dests.insert(TextureLayout::TransferDst);
        }

        // ColorAttachment can transition to...
        if usage.contains(TextureUsage::RENDER_ATTACHMENT) {
            let ca_dests = &mut transitions[TextureLayout::ColorAttachment as usize];
            if usage.contains(TextureUsage::TEXTURE_BINDING) {
                ca_dests.insert(TextureLayout::ShaderReadOnly);
            }
            if usage.contains(TextureUsage::COPY_SRC) {
                ca_dests.insert(TextureLayout::TransferSrc);
            }
            ca_dests.insert(TextureLayout::PresentSrc);
            ca_dests.insert(TextureLayout::ColorAttachment); // Stay as render target
        }

        // DepthStencilAttachment can transition to...
        if usage.contains(TextureUsage::RENDER_ATTACHMENT) {
            let ds_dests = &mut transitions[TextureLayout::DepthStencilAttachment as usize];
            if usage.contains(TextureUsage::TEXTURE_BINDING) {
                ds_dests.insert(TextureLayout::ShaderReadOnly);
                ds_dests.insert(TextureLayout::DepthStencilReadOnly);
            }
            if usage.contains(TextureUsage::COPY_SRC) {
                ds_dests.insert(TextureLayout::TransferSrc);
            }
            ds_dests.insert(TextureLayout::DepthStencilAttachment); // Stay
        }

        // DepthStencilReadOnly can transition to...
        if usage.contains(TextureUsage::TEXTURE_BINDING) {
            let dsr_dests = &mut transitions[TextureLayout::DepthStencilReadOnly as usize];
            if usage.contains(TextureUsage::RENDER_ATTACHMENT) {
                dsr_dests.insert(TextureLayout::DepthStencilAttachment);
            }
            if usage.contains(TextureUsage::COPY_SRC) {
                dsr_dests.insert(TextureLayout::TransferSrc);
            }
            dsr_dests.insert(TextureLayout::ShaderReadOnly);
            dsr_dests.insert(TextureLayout::DepthStencilReadOnly); // Stay
        }

        // ShaderReadOnly can transition to...
        if usage.contains(TextureUsage::TEXTURE_BINDING) {
            let sr_dests = &mut transitions[TextureLayout::ShaderReadOnly as usize];
            if usage.contains(TextureUsage::RENDER_ATTACHMENT) {
                sr_dests.insert(TextureLayout::ColorAttachment);
                sr_dests.insert(TextureLayout::DepthStencilAttachment);
            }
            if usage.contains(TextureUsage::COPY_SRC) {
                sr_dests.insert(TextureLayout::TransferSrc);
            }
            sr_dests.insert(TextureLayout::ShaderReadOnly); // Stay
        }

        // TransferSrc can transition to...
        if usage.contains(TextureUsage::COPY_SRC) {
            let ts_dests = &mut transitions[TextureLayout::TransferSrc as usize];
            if usage.contains(TextureUsage::TEXTURE_BINDING) {
                ts_dests.insert(TextureLayout::ShaderReadOnly);
            }
            if usage.contains(TextureUsage::RENDER_ATTACHMENT) {
                ts_dests.insert(TextureLayout::ColorAttachment);
                ts_dests.insert(TextureLayout::DepthStencilAttachment);
            }
            ts_dests.insert(TextureLayout::TransferSrc); // Stay
        }

        // TransferDst can transition to...
        if usage.contains(TextureUsage::COPY_DST) {
            let td_dests = &mut transitions[TextureLayout::TransferDst as usize];
            if usage.contains(TextureUsage::TEXTURE_BINDING) {
                td_dests.insert(TextureLayout::ShaderReadOnly);
            }
            if usage.contains(TextureUsage::RENDER_ATTACHMENT) {
                td_dests.insert(TextureLayout::ColorAttachment);
                td_dests.insert(TextureLayout::DepthStencilAttachment);
            }
            if usage.contains(TextureUsage::COPY_SRC) {
                td_dests.insert(TextureLayout::TransferSrc);
            }
            td_dests.insert(TextureLayout::TransferDst); // Stay
        }

        // PresentSrc can transition to...
        {
            let ps_dests = &mut transitions[TextureLayout::PresentSrc as usize];
            if usage.contains(TextureUsage::RENDER_ATTACHMENT) {
                ps_dests.insert(TextureLayout::ColorAttachment);
            }
            ps_dests.insert(TextureLayout::PresentSrc); // Stay
        }

        // General can transition to anything valid
        if usage.contains(TextureUsage::STORAGE_BINDING) {
            let gen_dests = &mut transitions[TextureLayout::General as usize];
            if usage.contains(TextureUsage::TEXTURE_BINDING) {
                gen_dests.insert(TextureLayout::ShaderReadOnly);
            }
            if usage.contains(TextureUsage::RENDER_ATTACHMENT) {
                gen_dests.insert(TextureLayout::ColorAttachment);
                gen_dests.insert(TextureLayout::DepthStencilAttachment);
            }
            if usage.contains(TextureUsage::COPY_SRC) {
                gen_dests.insert(TextureLayout::TransferSrc);
            }
            if usage.contains(TextureUsage::COPY_DST) {
                gen_dests.insert(TextureLayout::TransferDst);
            }
            gen_dests.insert(TextureLayout::General); // Stay
        }

        Self { usage, transitions }
    }

    /// Get the usage flags this graph was created for.
    pub fn usage(&self) -> TextureUsage {
        self.usage
    }

    /// Check if a transition is valid.
    pub fn is_valid_transition(&self, from: TextureLayout, to: TextureLayout) -> bool {
        self.transitions[from as usize].contains(&to)
    }

    /// Get valid destination layouts from a given state.
    pub fn valid_destinations(&self, from: TextureLayout) -> &HashSet<TextureLayout> {
        &self.transitions[from as usize]
    }
}

/// Cache for sharing `TextureUsageGraph` instances between textures.
///
/// Textures with identical usage flags share the same `Arc<TextureUsageGraph>`,
/// reducing memory allocation and enabling efficient comparison.
#[derive(Debug, Default)]
pub struct TextureUsageGraphCache {
    cache: RwLock<HashMap<TextureUsage, Arc<TextureUsageGraph>>>,
}

impl TextureUsageGraphCache {
    /// Create a new empty cache.
    pub fn new() -> Self {
        Self::default()
    }

    /// Get or create a usage graph for the given usage flags.
    pub fn get_or_create(&self, usage: TextureUsage) -> Arc<TextureUsageGraph> {
        // Fast path: read lock
        if let Some(graph) = self.cache.read().get(&usage) {
            return Arc::clone(graph);
        }

        // Slow path: write lock
        let mut cache = self.cache.write();
        cache
            .entry(usage)
            .or_insert_with(|| Arc::new(TextureUsageGraph::from_usage(usage)))
            .clone()
    }
}

/// Unique identifier for a Vulkan image within the layout tracker.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct TextureId(u64);

impl From<vk::Image> for TextureId {
    fn from(image: vk::Image) -> Self {
        Self(image.as_raw())
    }
}

impl TextureId {
    /// Create a texture ID from a raw Vulkan image handle.
    pub fn from_raw(handle: u64) -> Self {
        Self(handle)
    }

    /// Get the raw handle value.
    pub fn raw(&self) -> u64 {
        self.0
    }
}

/// Per-frame texture layout state.
///
/// Each frame-in-flight has its own state because the same texture
/// might be in different layouts in different frames.
#[derive(Debug, Default)]
pub struct FrameLayoutState {
    /// Current layout of each tracked texture.
    layouts: HashMap<TextureId, TextureLayout>,
}

impl FrameLayoutState {
    /// Create a new empty frame state.
    pub fn new() -> Self {
        Self::default()
    }

    /// Get the current layout of a texture, or `Undefined` if not tracked.
    pub fn get_layout(&self, id: TextureId) -> TextureLayout {
        self.layouts
            .get(&id)
            .copied()
            .unwrap_or(TextureLayout::Undefined)
    }

    /// Update the layout after a transition.
    pub fn set_layout(&mut self, id: TextureId, layout: TextureLayout) {
        self.layouts.insert(id, layout);
    }

    /// Clear tracking for a specific texture (e.g., when texture is destroyed).
    pub fn remove(&mut self, id: TextureId) {
        self.layouts.remove(&id);
    }

    /// Reset all layouts to `Undefined`.
    ///
    /// This is called at the start of a new frame to ensure textures
    /// start from a known state.
    pub fn reset(&mut self) {
        self.layouts.clear();
    }

    /// Get the number of tracked textures.
    pub fn len(&self) -> usize {
        self.layouts.len()
    }

    /// Check if any textures are being tracked.
    pub fn is_empty(&self) -> bool {
        self.layouts.is_empty()
    }
}

/// Controller for texture layout tracking across frames.
///
/// This struct manages per-frame layout state and provides the usage graph
/// cache for efficient sharing between textures.
#[derive(Debug)]
pub struct TextureLayoutTracker {
    /// Layout state per frame in flight.
    frame_states: Vec<FrameLayoutState>,
    /// Current frame index.
    current_frame: usize,
    /// Usage graph cache for sharing.
    usage_graph_cache: TextureUsageGraphCache,
}

impl TextureLayoutTracker {
    /// Create a new tracker for the specified number of frames in flight.
    pub fn new(frames_in_flight: usize) -> Self {
        Self {
            frame_states: (0..frames_in_flight)
                .map(|_| FrameLayoutState::new())
                .collect(),
            current_frame: 0,
            usage_graph_cache: TextureUsageGraphCache::new(),
        }
    }

    /// Advance to the next frame.
    ///
    /// This resets the layout state for the new frame slot, ensuring
    /// textures start from `Undefined` state.
    pub fn advance_frame(&mut self) {
        self.current_frame = (self.current_frame + 1) % self.frame_states.len();
        // Reset the frame state for the new frame
        self.frame_states[self.current_frame].reset();
    }

    /// Get the current frame index.
    pub fn current_frame(&self) -> usize {
        self.current_frame
    }

    /// Get the current frame's layout state (immutable).
    pub fn current_state(&self) -> &FrameLayoutState {
        &self.frame_states[self.current_frame]
    }

    /// Get the current frame's layout state (mutable).
    pub fn current_state_mut(&mut self) -> &mut FrameLayoutState {
        &mut self.frame_states[self.current_frame]
    }

    /// Get or create a usage graph for the given usage flags.
    pub fn get_usage_graph(&self, usage: TextureUsage) -> Arc<TextureUsageGraph> {
        self.usage_graph_cache.get_or_create(usage)
    }

    /// Get the current layout of a texture in the current frame.
    pub fn get_layout(&self, id: TextureId) -> TextureLayout {
        self.current_state().get_layout(id)
    }

    /// Set the layout of a texture in the current frame.
    pub fn set_layout(&mut self, id: TextureId, layout: TextureLayout) {
        self.current_state_mut().set_layout(id, layout);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_texture_layout_to_vk() {
        assert_eq!(TextureLayout::Undefined.to_vk(), vk::ImageLayout::UNDEFINED);
        assert_eq!(
            TextureLayout::ColorAttachment.to_vk(),
            vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL
        );
        assert_eq!(
            TextureLayout::ShaderReadOnly.to_vk(),
            vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL
        );
        assert_eq!(
            TextureLayout::TransferSrc.to_vk(),
            vk::ImageLayout::TRANSFER_SRC_OPTIMAL
        );
        assert_eq!(
            TextureLayout::TransferDst.to_vk(),
            vk::ImageLayout::TRANSFER_DST_OPTIMAL
        );
    }

    #[test]
    fn test_usage_graph_render_attachment() {
        let usage = TextureUsage::RENDER_ATTACHMENT | TextureUsage::TEXTURE_BINDING;
        let graph = TextureUsageGraph::from_usage(usage);

        // Can go from Undefined to ColorAttachment
        assert!(
            graph.is_valid_transition(TextureLayout::Undefined, TextureLayout::ColorAttachment)
        );

        // Can go from ColorAttachment to ShaderReadOnly
        assert!(graph.is_valid_transition(
            TextureLayout::ColorAttachment,
            TextureLayout::ShaderReadOnly
        ));

        // Can go from ShaderReadOnly to ColorAttachment
        assert!(graph.is_valid_transition(
            TextureLayout::ShaderReadOnly,
            TextureLayout::ColorAttachment
        ));

        // Cannot go directly to TransferSrc (no COPY_SRC)
        assert!(
            !graph.is_valid_transition(TextureLayout::ColorAttachment, TextureLayout::TransferSrc)
        );
    }

    #[test]
    fn test_usage_graph_copy_src() {
        let usage = TextureUsage::RENDER_ATTACHMENT
            | TextureUsage::COPY_SRC
            | TextureUsage::TEXTURE_BINDING;
        let graph = TextureUsageGraph::from_usage(usage);

        // Can go from ColorAttachment to TransferSrc
        assert!(
            graph.is_valid_transition(TextureLayout::ColorAttachment, TextureLayout::TransferSrc)
        );

        // Can go from TransferSrc to ShaderReadOnly
        assert!(
            graph.is_valid_transition(TextureLayout::TransferSrc, TextureLayout::ShaderReadOnly)
        );
    }

    #[test]
    fn test_usage_graph_cache() {
        let cache = TextureUsageGraphCache::new();

        let usage = TextureUsage::RENDER_ATTACHMENT | TextureUsage::TEXTURE_BINDING;

        let graph1 = cache.get_or_create(usage);
        let graph2 = cache.get_or_create(usage);

        // Should return the same Arc
        assert!(Arc::ptr_eq(&graph1, &graph2));
    }

    #[test]
    fn test_frame_layout_state() {
        let mut state = FrameLayoutState::new();

        let id = TextureId::from_raw(12345);

        // Initially undefined
        assert_eq!(state.get_layout(id), TextureLayout::Undefined);

        // Set layout
        state.set_layout(id, TextureLayout::ColorAttachment);
        assert_eq!(state.get_layout(id), TextureLayout::ColorAttachment);

        // Reset clears everything
        state.reset();
        assert_eq!(state.get_layout(id), TextureLayout::Undefined);
    }

    #[test]
    fn test_layout_tracker_advance_frame() {
        let mut tracker = TextureLayoutTracker::new(3);

        let id = TextureId::from_raw(12345);

        // Set layout in frame 0
        tracker.set_layout(id, TextureLayout::ColorAttachment);
        assert_eq!(tracker.get_layout(id), TextureLayout::ColorAttachment);

        // Advance to frame 1
        tracker.advance_frame();
        assert_eq!(tracker.current_frame(), 1);
        // New frame starts undefined
        assert_eq!(tracker.get_layout(id), TextureLayout::Undefined);

        // Set different layout in frame 1
        tracker.set_layout(id, TextureLayout::ShaderReadOnly);
        assert_eq!(tracker.get_layout(id), TextureLayout::ShaderReadOnly);
    }
}
