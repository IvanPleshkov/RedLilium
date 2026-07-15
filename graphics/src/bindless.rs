//! Bindless texture/sampler heap (#117) — engine-side slot management.
//!
//! The device owns one update-after-bind descriptor set (the *heap*) with a
//! runtime-sized sampled-texture array and a sampler array. Textures and
//! samplers are registered explicitly
//! ([`GraphicsDevice::bindless_register_texture`]) and receive a stable slot
//! index the shader uses via `NonUniformResourceIndex`. This module owns the
//! CPU side: slot allocation, keep-alive of registered resources, and
//! fence-deferred slot recycling.
//!
//! **Recycling invariant:** update-after-bind allows writing a descriptor
//! while the heap is bound to in-flight work only if that slot is not
//! *dynamically used* by it. A freshly allocated slot was never handed to any
//! shader, and an unregistered slot becomes reusable only after
//! `MAX_FRAMES_IN_FLIGHT` frame advances — by which point every frame that
//! could have indexed it has retired.
//!
//! [`GraphicsDevice::bindless_register_texture`]: crate::GraphicsDevice::bindless_register_texture

use std::collections::HashMap;
use std::sync::Arc;

use parking_lot::Mutex;

use crate::error::GraphicsError;
use crate::pipeline::MAX_FRAMES_IN_FLIGHT;
use crate::resources::{Sampler, Texture};

/// Binding index of the sampled-texture array within the heap group.
pub const BINDLESS_TEXTURES_BINDING: u32 = 0;
/// Binding index of the sampler array within the heap group.
pub const BINDLESS_SAMPLERS_BINDING: u32 = 1;

/// A registered resource retired by unregistration, waiting out the frames
/// that might still index its slot.
enum PendingFree {
    Texture {
        slot: u32,
        keep_alive: Arc<Texture>,
    },
    Sampler {
        slot: u32,
        /// Never read — exists purely to keep the sampler alive through the
        /// retirement window (textures additionally surface through
        /// `for_each_live_texture` for barrier declaration).
        #[allow(dead_code)]
        keep_alive: Arc<Sampler>,
    },
}

#[derive(Default)]
struct SlotState {
    next_texture: u32,
    free_textures: Vec<u32>,
    next_sampler: u32,
    free_samplers: Vec<u32>,
    live_textures: HashMap<u32, Arc<Texture>>,
    live_samplers: HashMap<u32, Arc<Sampler>>,
    /// `(frame the slot was freed at, resource)` — recycled once
    /// `frame + MAX_FRAMES_IN_FLIGHT <= current`.
    pending: Vec<(u64, PendingFree)>,
    frame: u64,
}

/// Shared slot table of the bindless heap. Held by the device and by the
/// heap's [`BoundResource::BindlessHeap`](crate::BoundResource) entry, which
/// is how pass resource inference reaches the live textures (every bindless
/// draw declares them as `ShaderRead`, preserving automatic layout
/// transitions).
pub struct BindlessSlots {
    texture_capacity: u32,
    sampler_capacity: u32,
    state: Mutex<SlotState>,
}

impl BindlessSlots {
    pub(crate) fn new(texture_capacity: u32, sampler_capacity: u32) -> Self {
        Self {
            texture_capacity,
            sampler_capacity,
            state: Mutex::new(SlotState::default()),
        }
    }

    /// Heap capacities `(textures, samplers)`.
    pub fn capacities(&self) -> (u32, u32) {
        (self.texture_capacity, self.sampler_capacity)
    }

    /// Numbers of currently registered `(textures, samplers)`.
    pub fn live_counts(&self) -> (usize, usize) {
        let state = self.state.lock();
        (state.live_textures.len(), state.live_samplers.len())
    }

    /// Allocate a texture slot and record the keep-alive. The descriptor
    /// write is the caller's (the device's) job.
    pub(crate) fn allocate_texture(&self, texture: Arc<Texture>) -> Result<u32, GraphicsError> {
        let mut state = self.state.lock();
        let slot = match state.free_textures.pop() {
            Some(slot) => slot,
            None if state.next_texture < self.texture_capacity => {
                let slot = state.next_texture;
                state.next_texture += 1;
                slot
            }
            None => {
                return Err(GraphicsError::ResourceCreationFailed(format!(
                    "bindless texture heap is full ({} slots; #117)",
                    self.texture_capacity
                )));
            }
        };
        state.live_textures.insert(slot, texture);
        Ok(slot)
    }

    /// Allocate a sampler slot and record the keep-alive.
    pub(crate) fn allocate_sampler(&self, sampler: Arc<Sampler>) -> Result<u32, GraphicsError> {
        let mut state = self.state.lock();
        let slot = match state.free_samplers.pop() {
            Some(slot) => slot,
            None if state.next_sampler < self.sampler_capacity => {
                let slot = state.next_sampler;
                state.next_sampler += 1;
                slot
            }
            None => {
                return Err(GraphicsError::ResourceCreationFailed(format!(
                    "bindless sampler heap is full ({} slots; #117)",
                    self.sampler_capacity
                )));
            }
        };
        state.live_samplers.insert(slot, sampler);
        Ok(slot)
    }

    /// Retire a texture slot: the resource stays alive (and the slot
    /// unreusable) until every frame that might index it has passed a fence.
    pub(crate) fn release_texture(&self, slot: u32) -> Result<(), GraphicsError> {
        let mut state = self.state.lock();
        let keep_alive = state.live_textures.remove(&slot).ok_or_else(|| {
            GraphicsError::InvalidParameter(format!(
                "bindless texture slot {slot} is not registered"
            ))
        })?;
        let frame = state.frame;
        state
            .pending
            .push((frame, PendingFree::Texture { slot, keep_alive }));
        Ok(())
    }

    /// Retire a sampler slot (see [`release_texture`](Self::release_texture)).
    pub(crate) fn release_sampler(&self, slot: u32) -> Result<(), GraphicsError> {
        let mut state = self.state.lock();
        let keep_alive = state.live_samplers.remove(&slot).ok_or_else(|| {
            GraphicsError::InvalidParameter(format!(
                "bindless sampler slot {slot} is not registered"
            ))
        })?;
        let frame = state.frame;
        state
            .pending
            .push((frame, PendingFree::Sampler { slot, keep_alive }));
        Ok(())
    }

    /// Advance the frame counter and recycle slots whose retirement window
    /// has passed. Called from `GraphicsDevice::advance_frame`, i.e. after a
    /// frame-fence wait.
    pub(crate) fn advance_frame(&self) {
        let mut state = self.state.lock();
        state.frame += 1;
        let current = state.frame;
        let pending = std::mem::take(&mut state.pending);
        for (freed_at, entry) in pending {
            if freed_at + MAX_FRAMES_IN_FLIGHT as u64 <= current {
                match entry {
                    PendingFree::Texture { slot, .. } => state.free_textures.push(slot),
                    PendingFree::Sampler { slot, .. } => state.free_samplers.push(slot),
                }
            } else {
                state.pending.push((freed_at, entry));
            }
        }
    }

    /// Visit every live (registered) texture — pass resource inference
    /// declares each as a shader read.
    pub(crate) fn for_each_live_texture(&self, mut f: impl FnMut(&Arc<Texture>)) {
        let state = self.state.lock();
        for texture in state.live_textures.values() {
            f(texture);
        }
        // Retired-but-pending textures may still be indexed by in-flight
        // frames; new frames declaring them keeps their layout stable until
        // the slot recycles.
        for (_, pending) in &state.pending {
            if let PendingFree::Texture { keep_alive, .. } = pending {
                f(keep_alive);
            }
        }
    }
}

impl std::fmt::Debug for BindlessSlots {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let (textures, samplers) = self.live_counts();
        f.debug_struct("BindlessSlots")
            .field("texture_capacity", &self.texture_capacity)
            .field("sampler_capacity", &self.sampler_capacity)
            .field("live_textures", &textures)
            .field("live_samplers", &samplers)
            .finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::instance::GraphicsInstance;
    use crate::types::{TextureDescriptor, TextureFormat, TextureUsage};

    fn test_texture() -> Arc<Texture> {
        let device = GraphicsInstance::new().unwrap().create_device().unwrap();
        device
            .create_texture(&TextureDescriptor::new_2d(
                4,
                4,
                TextureFormat::Rgba8Unorm,
                TextureUsage::TEXTURE_BINDING,
            ))
            .unwrap()
    }

    /// Slots grow monotonically until capacity, error when full, and a
    /// released slot recycles only after MAX_FRAMES_IN_FLIGHT advances.
    #[test]
    fn slot_allocation_and_deferred_recycling() {
        let texture = test_texture();
        let slots = BindlessSlots::new(2, 1);

        assert_eq!(slots.allocate_texture(texture.clone()).unwrap(), 0);
        assert_eq!(slots.allocate_texture(texture.clone()).unwrap(), 1);
        assert!(slots.allocate_texture(texture.clone()).is_err());
        assert_eq!(slots.live_counts().0, 2);

        // Release slot 0: not reusable until the in-flight window passes.
        slots.release_texture(0).unwrap();
        assert_eq!(slots.live_counts().0, 1);
        assert!(slots.allocate_texture(texture.clone()).is_err());
        // The retired texture still counts as live for barrier declaration.
        let mut seen = 0;
        slots.for_each_live_texture(|_| seen += 1);
        assert_eq!(seen, 2);

        for _ in 0..MAX_FRAMES_IN_FLIGHT {
            slots.advance_frame();
        }
        assert_eq!(slots.allocate_texture(texture.clone()).unwrap(), 0);

        // Double release / unknown slot are errors.
        assert!(slots.release_texture(7).is_err());
    }
}
