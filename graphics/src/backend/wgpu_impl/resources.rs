//! Resource creation for the wgpu backend.

use std::sync::Mutex;

use crate::error::GraphicsError;
use crate::types::{BufferDescriptor, SamplerDescriptor, TextureDescriptor};

use super::super::{GpuBuffer, GpuFence, GpuSampler, GpuTexture};
use super::WgpuBackend;
use super::conversion::{
    convert_address_mode, convert_buffer_usage, convert_compare_function, convert_filter_mode,
    convert_mipmap_filter_mode, convert_texture_format, convert_texture_usage,
};

impl WgpuBackend {
    /// Create a buffer resource.
    pub fn create_buffer(&self, descriptor: &BufferDescriptor) -> Result<GpuBuffer, GraphicsError> {
        let usage = convert_buffer_usage(descriptor.usage);

        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: descriptor.label.as_deref(),
            size: descriptor.size,
            usage,
            mapped_at_creation: false,
        });

        Ok(GpuBuffer::Wgpu(buffer))
    }

    /// Create a texture resource.
    pub fn create_texture(
        &self,
        descriptor: &TextureDescriptor,
    ) -> Result<GpuTexture, GraphicsError> {
        use crate::types::TextureDimension;

        let format = convert_texture_format(descriptor.format);
        let usage = convert_texture_usage(descriptor.usage);

        // Convert our texture dimension to wgpu's
        let (wgpu_dimension, depth_or_array_layers) = match descriptor.dimension {
            TextureDimension::D1 => (wgpu::TextureDimension::D1, descriptor.size.depth),
            TextureDimension::D2 => (wgpu::TextureDimension::D2, descriptor.size.depth),
            TextureDimension::D3 => (wgpu::TextureDimension::D3, descriptor.size.depth),
            TextureDimension::Cube => (wgpu::TextureDimension::D2, 6),
            TextureDimension::CubeArray => (wgpu::TextureDimension::D2, descriptor.size.depth * 6),
        };

        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: descriptor.label.as_deref(),
            size: wgpu::Extent3d {
                width: descriptor.size.width,
                height: descriptor.size.height,
                depth_or_array_layers,
            },
            mip_level_count: descriptor.mip_level_count,
            sample_count: descriptor.sample_count,
            dimension: wgpu_dimension,
            format,
            usage,
            view_formats: &[],
        });

        // Create the appropriate view based on dimension
        let view_dimension = match descriptor.dimension {
            TextureDimension::D1 => wgpu::TextureViewDimension::D1,
            TextureDimension::D2 => {
                if descriptor.size.depth > 1 {
                    wgpu::TextureViewDimension::D2Array
                } else {
                    wgpu::TextureViewDimension::D2
                }
            }
            TextureDimension::D3 => wgpu::TextureViewDimension::D3,
            TextureDimension::Cube => wgpu::TextureViewDimension::Cube,
            TextureDimension::CubeArray => wgpu::TextureViewDimension::CubeArray,
        };

        let view = texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(view_dimension),
            ..Default::default()
        });

        Ok(GpuTexture::Wgpu { texture, view })
    }

    /// Create a sampler resource.
    pub fn create_sampler(
        &self,
        descriptor: &SamplerDescriptor,
    ) -> Result<GpuSampler, GraphicsError> {
        let sampler = self.device.create_sampler(&wgpu::SamplerDescriptor {
            label: descriptor.label.as_deref(),
            address_mode_u: convert_address_mode(descriptor.address_mode_u),
            address_mode_v: convert_address_mode(descriptor.address_mode_v),
            address_mode_w: convert_address_mode(descriptor.address_mode_w),
            mag_filter: convert_filter_mode(descriptor.mag_filter),
            min_filter: convert_filter_mode(descriptor.min_filter),
            mipmap_filter: convert_mipmap_filter_mode(descriptor.mipmap_filter),
            lod_min_clamp: descriptor.lod_min_clamp,
            lod_max_clamp: descriptor.lod_max_clamp,
            compare: descriptor.compare.map(convert_compare_function),
            anisotropy_clamp: descriptor.anisotropy_clamp,
            border_color: None,
        });

        Ok(GpuSampler::Wgpu(sampler))
    }

    /// Create a fence for CPU-GPU synchronization.
    ///
    /// Note: wgpu fences work differently from Vulkan fences. Instead of a binary
    /// signaled/unsignaled state, wgpu tracks submission indices. A fence with no
    /// submission (None) is considered "signaled" (no work to wait for).
    ///
    /// The `signaled` parameter is acknowledged but has limited effect:
    /// - `signaled=true`: Fence starts with no submission (effectively signaled)
    /// - `signaled=false`: Same as above - wgpu cannot represent an unsignaled fence
    ///   without pending work. The fence becomes meaningful only after `execute_graph`
    ///   stores a submission index.
    pub fn create_fence(&self, signaled: bool) -> GpuFence {
        if !signaled {
            log::debug!(
                "wgpu fence created with signaled=false, but wgpu fences track submissions, \
                 not binary state. Fence will appear signaled until work is submitted."
            );
        }
        GpuFence::Wgpu {
            device: self.device.clone(),
            submission_index: Mutex::new(None),
        }
    }

    /// Wait for a fence to be signaled.
    pub fn wait_fence(&self, fence: &GpuFence) {
        if let GpuFence::Wgpu {
            device,
            submission_index,
        } = fence
            && let Ok(guard) = submission_index.lock()
            && let Some(idx) = guard.clone()
        {
            // Wait for the specific submission
            let _ = device.poll(wgpu::PollType::Wait {
                submission_index: Some(idx),
                timeout: Some(std::time::Duration::from_secs(10)),
            });
        }
    }

    /// Check if a fence is signaled (non-blocking).
    ///
    /// Returns `true` if:
    /// - No work has been submitted yet (fence is in initial state)
    /// - All submitted work has completed
    ///
    /// Returns `false` if:
    /// - Work is still pending on the GPU
    /// - Lock acquisition failed (conservative assumption)
    /// - Not a wgpu fence
    pub fn is_fence_signaled(&self, fence: &GpuFence) -> bool {
        let GpuFence::Wgpu {
            device,
            submission_index,
        } = fence
        else {
            return false; // Not a wgpu fence
        };

        let Ok(guard) = submission_index.lock() else {
            return false; // Lock failed, assume not signaled (conservative)
        };

        // No submission yet means fence is in initial "signaled" state
        if guard.is_none() {
            return true;
        }

        // Poll without blocking to check completion status.
        // Note: wgpu's non-blocking poll checks if ALL queue work is done,
        // not a specific submission. This is conservative but correct.
        match device.poll(wgpu::PollType::Poll) {
            Ok(status) => status.is_queue_empty(),
            Err(_) => false, // Poll failed, assume not signaled
        }
    }

    /// Signal a fence (for testing/dummy backend).
    pub fn signal_fence(&self, _fence: &GpuFence) {
        // wgpu fences are signaled automatically when GPU work completes
    }

    /// Write data to a buffer.
    pub fn write_buffer(&self, buffer: &GpuBuffer, offset: u64, data: &[u8]) {
        if let GpuBuffer::Wgpu(wgpu_buffer) = buffer {
            self.queue.write_buffer(wgpu_buffer, offset, data);
        }
    }

    /// Write data to a texture.
    pub fn write_texture(&self, texture: &GpuTexture, data: &[u8], descriptor: &TextureDescriptor) {
        use crate::types::TextureDimension;

        if let GpuTexture::Wgpu {
            texture: wgpu_texture,
            ..
        } = texture
        {
            let format = convert_texture_format(descriptor.format);
            let block_size = format.block_copy_size(None).unwrap_or(4);
            let bytes_per_row = descriptor.size.width * block_size;

            let depth_or_array_layers = match descriptor.dimension {
                TextureDimension::Cube => 6,
                TextureDimension::CubeArray => descriptor.size.depth * 6,
                _ => descriptor.size.depth,
            };

            self.queue.write_texture(
                wgpu::TexelCopyTextureInfo {
                    texture: wgpu_texture,
                    mip_level: 0,
                    origin: wgpu::Origin3d::ZERO,
                    aspect: wgpu::TextureAspect::All,
                },
                data,
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(bytes_per_row),
                    rows_per_image: Some(descriptor.size.height),
                },
                wgpu::Extent3d {
                    width: descriptor.size.width,
                    height: descriptor.size.height,
                    depth_or_array_layers,
                },
            );
        }
    }

    /// Read data from a buffer.
    pub fn read_buffer(&self, buffer: &GpuBuffer, offset: u64, size: u64) -> Vec<u8> {
        if let GpuBuffer::Wgpu(wgpu_buffer) = buffer {
            // Try to map the buffer directly first (works if buffer has MAP_READ)
            let slice = wgpu_buffer.slice(offset..offset + size);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result);
            });

            let _ = self.device.poll(wgpu::PollType::wait_indefinitely());

            if let Ok(Ok(())) = rx.recv() {
                // Direct mapping succeeded
                let data = slice.get_mapped_range().to_vec();
                let _ = slice;
                wgpu_buffer.unmap();
                return data;
            }

            // Direct mapping failed - use staging buffer approach
            // This requires the source buffer to have COPY_SRC
            let staging = self.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("Read Staging Buffer"),
                size,
                usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                mapped_at_creation: false,
            });

            // Copy from source to staging
            let mut encoder = self
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Read Buffer Encoder"),
                });
            encoder.copy_buffer_to_buffer(wgpu_buffer, offset, &staging, 0, size);

            let idx = self.queue.submit(std::iter::once(encoder.finish()));

            // Wait for copy to complete
            let _ = self.device.poll(wgpu::PollType::Wait {
                submission_index: Some(idx),
                timeout: Some(std::time::Duration::from_secs(10)),
            });

            // Map and read
            let slice = staging.slice(..);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |result| {
                let _ = tx.send(result);
            });

            let _ = self.device.poll(wgpu::PollType::wait_indefinitely());
            let _ = rx.recv();

            let data = slice.get_mapped_range().to_vec();
            let _ = slice;
            staging.unmap();

            data
        } else {
            vec![0u8; size as usize]
        }
    }
}
