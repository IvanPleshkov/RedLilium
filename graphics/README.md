# RedLilium Graphics

Custom rendering engine for RedLilium Engine.

## Overview

This crate provides the rendering infrastructure built around an abstract **render graph** that enables:

- Declarative description of render passes and dependencies
- Automatic resource barrier and synchronization management
- Backend-agnostic rendering code
- Backend command encoding with automatic synchronization

## Architecture

### Render Graph

The render graph is the central abstraction for describing rendering operations:

```rust,ignore
use std::sync::Arc;
use redlilium_graphics::{
    BufferDescriptor, BufferUsage, TransferConfig, TransferOperation, TransferPass,
};

let buffer = device.create_buffer(&BufferDescriptor::new(
    256, BufferUsage::VERTEX | BufferUsage::COPY_DST,
))?;
let mut schedule = pipeline.begin_frame()?;
let mut graph = schedule.acquire_graph();
let mut upload = TransferPass::new("upload vertices".into());
upload.set_transfer_config(TransferConfig::new().with_operation(
    TransferOperation::write_buffer(buffer.clone(), 0, Arc::from(vertex_bytes)),
));
graph.add_transfer_pass(upload);
// Add graphics passes using buffer; the graph derives upload → draw ordering.
schedule.submit(graph)?;
pipeline.end_frame(schedule);
```

Resource transfers go through the graph; `GraphicsDevice` only creates resources.
`submit` reports failures through `Result`. Dropping a schedule automatically
returns its submitted resources to the pipeline for retirement after GPU completion.
There is one live frame pipeline per graphics instance and one active schedule per
pipeline. `submit_with_mode` selects Strict compilation when ambiguous writers
should be treated as errors instead of using addition order.

### Resource and command validation

Resource creation validates binding completeness and uniqueness, device ownership,
usage, buffer ranges/alignment/binding-size limits, texture dimensions/sample
types, and sampler comparison mode. Texture descriptors use dimension-specific
limits (including array layers), legal mip counts and MSAA constraints. Vulkan
queries the requested format/dimension/usage combination; wgpu checks granted
format features and limits. Compressed base extents on wgpu must be block-aligned;
smaller edge mips remain supported.

Before submission, every graphics/compute command is checked for pipeline stage,
binding-layout compatibility, dynamic uniform offsets, resource ownership and
pending CPU mappings. Draw constructors only describe commands; their former
debug-only checks now return submission errors in every build. Graphics pipelines must match their attachments; direct
compute and mesh dispatch counts must fit device limits. An invalid command
rejects the graph before any of its transfers execute. Compute and indirect
raster commands currently have no dynamic-offset parameter, so materials requiring
dynamic offsets cannot be used with those commands.

Native wgpu resource creation and command recording use error scopes, returning
validation failures as `InvalidParameter`, allocation failures as `OutOfMemory`,
and internal errors as `Internal`. Lost devices continue returning `DeviceLost`.
Browser WebGPU validation is asynchronous: a creation call can return before the
browser validates it. Such errors, and errors reported after queue submission,
are retained and returned by the next resource creation or graph submission.
Submitted resources remain protected by their fence even when a later error arrives.

Texture views select validated mip/layer ranges and aspects. Create them with
`device.create_texture_view(&texture, &TextureViewDescriptor::new(dimension))`,
then bind with `with_texture_view` or attach with `RenderTarget::from_view`.
Attachment views select one 2D mip/layer; D3 views sample volumes, not individual
Z slices. Cube views select six aligned faces. The texture keeps weak references
to cached views; each live view retains its parent texture. Legacy texture binding
and attachment helpers remain available.

`BindingType::SampledTexture` specifies view dimension, Float/Sint/Uint/Depth sample
type and multisampling. Existing texture binding variants remain equivalent
conveniences. Storage texture bindings and format reinterpretation are not added.

Graph dependencies and Vulkan barriers track mip/layer ranges across passes,
submissions and queues. Whole-texture bindings declare all levels and layers.
Overlapping incompatible accesses in a graphics/compute pass are rejected;
read-only depth co-attachment still uses its explicit sampled-depth layout.
Disjoint subresources can be sampled and rendered in the same pass. Vulkan
conservatively tracks depth/stencil aspects together; D3 tracking is per mip.
The implementation and scope are described in [the design](TEXTURE_VIEWS_DESIGN.md).

### Readback results

Create a fresh `Readback::new()` for each `TransferOperation::readback_buffer`
operation. After a successful graph submission and frame-slot retirement,
`take_result()` yields `Some(Ok(bytes))` or `Some(Err(error))` once; `status()`
distinguishes Pending, Ready and Consumed. Empty successful reads are explicit.
Submission errors must be handled separately: an unsubmitted request is still
pending. Reusing a result handle in another submitted operation is rejected.
Readbacks start when `begin_frame` retires a slot or `recycle_all_graphs` drains
completed graphs. wgpu mapping then needs device polling (driven by subsequent
frames); `wait_idle` alone does not consume graph readback requests.

Requests from one retiring slot sharing a buffer use a single mapping of the
union of their ranges. Further batches queue until unmap; requests are never
silently skipped. GPU access to a buffer with a pending CPU mapping is rejected.
This is a post-fence read of the readback buffer, not a snapshot at each marker:
use separate source buffers if distinct intermediate GPU results are required.
Editor picking and screenshots consume these results and report failures.

Migration: replace `Arc<Mutex<Vec<u8>>>` destinations with `Readback`, and polling
an empty vector with `take_result()`/`status()`. For repeated operations create
new handles rather than clearing and reusing the previous result.

### Transfer validation

Before recording commands, `submit` validates every transfer operation in the
graph on all backends, including Dummy. Invalid parameters return
`GraphicsError::InvalidParameter` with the pass name and operation index; no
commands from that graph execute, and the schedule remains usable. Unsupported
mip generation returns `FeatureNotSupported`.

Validation checks resource ownership, usage flags, pending CPU mappings, buffer
ranges, mip/layer bounds, formats, sample counts, and copy layouts. The common
contract includes:

- Buffer copies and nonempty writes require 4-byte aligned offsets and sizes.
  CPU readback requires `MAP_READ`, an 8-byte aligned offset, and a 4-byte aligned
  size. Empty writes and readbacks are allowed within buffer bounds.
- Buffer/texture offsets must be aligned to both 4 bytes and the format block
  size. Copies spanning multiple block rows or images need a 256-byte aligned
  row pitch. The last row only needs its actual texel bytes, without trailing
  padding. Compressed copies may end at a mip edge smaller than a block.
- Copy regions must be nonempty and have nonzero extents. Array layers do not
  shrink with mip level; 3D depth does. Buffer/texture copies require one sample.
  Depth/stencil and multisampled texture copies cover the full mip width and
  height. Buffer copies of combined depth/stencil or `Depth24Plus`, and uploads
  of `Depth32Float`, are unsupported by the common contract.
- Texture copies require matching dimension classes, sample counts, and formats
  (linear/sRGB counterparts are compatible). Buffer copies require different
  resources. Same-texture copies are allowed
  only when all source and destination mip/layer ranges are disjoint; disjoint
  pixel rectangles within the same subresource remain unsupported.
- Mip generation requires `COPY_SRC | COPY_DST` and, for multiple levels, a
  single-sampled 2D texture, 2D array, cubemap, cube array or 3D volume with a
  supported format. A single level is a no-op.

`upload_texture_data` and `upload_texture_level` prepare staging data with the
required padding; the destination upload is still an ordered graph operation.

### Mip generation

`TransferOperation::generate_mipmaps(texture)` generates the allocated levels
from mip 0 inside the graph. The resource needs only `COPY_SRC | COPY_DST`.
For 2D textures/arrays and 3D volumes, Vulkan uses linear blits. wgpu uses a cached
render pipeline and private scratch texture, with weighted box `textureLoad` reduction: float formats do not need
hardware linear filtering, and odd-size edges contribute to the result. Both
paths filter sRGB colors in linear light; their kernels may differ for odd sizes.
Check `device.supports_mipmap_generation(format)` before requesting GPU generation.

The ECS texture importer defaults to `generate_mips: true`. It uses GPU generation
for ordinary supported colors and a CPU worker fallback for other uncompressed
color formats (including float, normalized, and integer channels). The CPU path
uploads each finished level through graph transfers. Integer averages round to
the nearest value; depth/stencil is not treated as color.

`TextureSettings::mip_filter` selects the import filter; old records default to
`Color`. Example RON settings:

```ron
(mip_filter: NormalMap)
(mip_filter: AlphaCoverage(cutoff: 128))
```

- `Color`: regular color reduction, with sRGB decoded before averaging.
- `NormalMap`: CPU reduction of tangent-space XYZ stored in RGB [0, 1], followed
  by normalization. Cubemap normals must use a common basis across faces, such
  as world space; face-local tangent frames are not converted automatically.
  Ordinary image files are decoded as linear regardless of
  `srgb`; a cancelling vector becomes +Z. This is not an XY-only normal decoder.
- `AlphaCoverage`: CPU reduction with per-level alpha scaling toward mip 0's
  fraction of texels passing `alpha >= cutoff / 255`. Cutoff is 1..=254 and should
  match the material's alpha-test threshold. RGB is unaffected by the correction.
  Coverage is approximate because texel counts, equal alpha values, and format
  quantization limit achievable fractions. It does not model filtered sampling
  or alpha-to-coverage at render time.

2D arrays generate a chain for each layer on Vulkan, wgpu, and the CPU import
path. Layers never mix; normal filtering and alpha coverage are computed per
layer. A base-only array receives generated mips when `generate_mips` is enabled;
the number of layers stays constant at every level. Uploads for all base layers
precede the GPU generation operation.

Cubemaps and cube arrays use cross-face tent reduction on CPU, Vulkan, and wgpu.
Taps outside a face are projected onto neighboring faces, including at corners,
using the [KTX/Vulkan face convention](https://github.khronos.org/Vulkan-Site/spec/latest/chapters/textures.html)
`+X, -X, +Y, -Y, +Z, -Z`. Filtering happens in linear light for sRGB. Each cube
is isolated from other cube-array elements. Alpha coverage uses a single
correction per cube, so separate face corrections cannot introduce seams.
Vulkan retains temporary images, views, and descriptors until the frame slot's
fences retire; wgpu retains them through its command buffers. This costs a
temporary mip chain per generation operation. Stored base faces remain unchanged.

This is mip downsampling across face boundaries, not specular IBL convolution
by roughness or a repair for mismatched source faces. No radiance solid-angle
integration or face-local normal-basis conversion is performed.

3D textures shrink width, height and depth independently down to one. wgpu renders
each destination Z slice using volume-weighted `textureLoad` reduction; CPU import
uses the same box kernel. Vulkan uses trilinear blits, whose kernel may differ for
odd dimensions. Normal-map filtering normalizes each reduced voxel; alpha coverage
is corrected over the entire volume. All slices of each mip are stored and uploaded
together. MoltenVK uses a temporary chain starting at half size to avoid incorrect
Z normalization in cross-mip blits: it blits between equal mip indices and copies
the result into the target level. This scratch survives until the frame slot retires.

Supplied mip chains are preserved, including compressed KTX2 assets. BC/ETC/ASTC
need precomputed mips; no runtime recompression is performed. Generation remains
limited to 2D textures, 2D arrays, cubemaps, cube arrays and 3D volumes without MSAA.
1D textures are unsupported; multisampled sources need a resolve before generation.
CPU fallback broadens **asset import** support; a
direct GPU graph operation on an unsupported format still returns an error.

### GPU environment filtering (IBL)

`ibl::EnvironmentFilter` builds reusable filtering jobs using the public graph,
materials and resource APIs. Vulkan and wgpu execute the same WGSL shaders.
`new(device)` creates shared pipelines; `prepare(source, settings)` creates a job
with its own output cubemaps and bindings. Neither call uploads or submits work.
`job.add_to_graph(graph, dependencies)` adds parameter uploads and convolution
passes, returning the final pass for lighting to depend on. Reuse the filter
across probes and the job across captures of the same source texture.

Input is a square, single-sampled cubemap with `TEXTURE_BINDING` and a complete,
initialized **ordinary radiance mip chain**. RGBA16F is the HDR path; normalized
RGBA8/BGRA8, including sRGB, are also accepted. After scene capture, generate the
source mips with `TransferOperation::generate_mipmaps` and make filtering depend
on that pass. Across graphs submit capture/mips → filtering → lighting in order.
Input radiance must be finite, nonnegative and representable in f16.

Outputs are sampled/renderable RGBA16F cubemaps, also readable via graph copies:

- `job.specular()` stores GGX convolution with `alpha = roughness²`, `N = V = R`,
  and `roughness = mip / max_reflection_lod`. Mip zero directly resamples the
  source at a LOD appropriate for the output size. Use the existing BRDF LUT.
- `job.diffuse()` stores cosine-weighted irradiance **divided by PI**, matching
  the deferred shader's `irradiance * albedo` convention.
- `job.max_reflection_lod()` supplies the runtime roughness-to-LOD scale.

Hammersley sampling is deterministic. Source LOD follows each sample's PDF to
reduce noise and missed bright sources; see
[pre-filtered importance sampling](https://google.github.io/filament/main/filament.html#annex/importancesamplingfortheibl/pre-filteredimportancesampling).
Results are approximate and need not match the offline equirectangular baker
texel for texel. Sizes and sample counts are configurable; defaults are 128-pixel
specular faces / 512 samples and 32-pixel diffuse faces / 256 samples.

This implements filtering for future reflection probes. Scene capture, probe
placement/selection, parallax correction, blending, and a per-frame update budget
are not implemented here. Each call schedules a complete update and overwrites
its job's outputs. Use separate front/back jobs when old lighting must remain
available while a new capture is prepared; publish only completed results. The
existing asset-based environment resolver and offline `bake-ibl` remain usable.

### Public API boundary

Applications use `GraphicsDevice`, engine resources, `RenderGraph`, and
`FramePipeline`/`FrameSchedule`. Backend implementations, native handles, and
Vulkan layout/access conversions are internal. Uploads, copies, and readbacks
remain transfer operations in the graph.

`RenderTarget` has a private representation. Construct it with `from_texture`,
`from_texture_mip`, `from_texture_layer`, or `from_surface`. Inspect it through
`format`, `width`, `height`, `sample_count`, `is_surface`, `texture`, `mip_level`,
and `array_layer`. The last three return `None` for a surface target.

Migration from the previous API:

- Replace `RenderTarget::Texture { texture, mip_level, array_layer }` with
  `RenderTarget::from_texture_layer(texture, mip_level, array_layer)`.
- Obtain surface targets from an acquired `SurfaceTexture` using
  `RenderTarget::from_surface(&surface_texture)`.
- Replace variant matching with the accessors above. Resource `gpu_handle()`
  methods, `SurfaceTexture::gpu_texture()`, and `Blas::device_address()` are
  internal; there is currently no public native interop API.
- Vulkan validation counters are available through
  `diagnostics::vulkan::{validation_error_count, reset_validation_error_count}`
  with the `vulkan-backend` feature. They count errors on the calling thread;
  validation must be enabled and the validation layer available.

### Backend Support

The render graph supports three backends:

| Backend | Crate | Use Case |
|---------|-------|----------|
| Vulkan | `ash` | High-performance desktop rendering |
| wgpu | `wgpu` 28.0.0 | Cross-platform and web support |
| Dummy | - | Testing without GPU |

### Module Structure

```
redlilium-graphics
├── graph/           # Render graph infrastructure
│   ├── mod.rs       # Graph builder and compiler
│   ├── pass.rs      # Render pass definitions
│   └── resource_usage.rs # Inferred resource access
├── diagnostics.rs   # Public validation diagnostics
├── backend/         # Internal backend implementations
│   ├── mod.rs       # Enum-based backend dispatch
│   ├── vulkan/      # Vulkan backend (ash)
│   ├── wgpu_impl/   # wgpu backend
│   └── dummy.rs     # Dummy backend for testing
└── types/           # Common types and descriptors
```

## Building

```bash
# Build the crate
cargo build -p redlilium-graphics

# Run tests
cargo test -p redlilium-graphics

# Generate documentation
cargo doc -p redlilium-graphics --open
```

## Feature Flags

| Feature | Description |
|---------|-------------|
| `vulkan-backend` | Enable Vulkan backend (default on desktop) |
| `wgpu-backend` | Enable wgpu backend (default) |
| `dummy` | Enable dummy backend for testing |

## Thread Safety

The render graph is designed for multithreaded environments:

- Graph construction is single-threaded for determinism
- Frame submission is serialized by the pipeline; independent CPU work can run in parallel
- All public types implement `Send + Sync` where appropriate
