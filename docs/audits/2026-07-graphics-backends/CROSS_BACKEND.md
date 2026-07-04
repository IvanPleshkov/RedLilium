# Cross-Backend Divergences & Abstraction-Layer Findings

Findings where the two backends implement the same engine-level operation with different
semantics (a scene renders differently or breaks on exactly one backend), plus issues in the
shared trait layer (`graphics/src/backend/mod.rs`) and its callers (`device.rs`,
`swapchain.rs`, `instance.rs`, `scheduler/`). Line numbers refer to the tree at commit `1d6ee28`.

The conversion tables were verified semantically equivalent between backends (buffer/texture
usages, filters, address modes, compare functions, blend factors/ops, topology, step mode
enums, present modes, stencil/depth clears) **except** where noted below.

## Critical

### [x] XB-C1. Vertex attribute `shader_location` conventions diverge between backends
- **Where:** `wgpu_impl/resources.rs:163-177` (sequential `enumerate()` order) vs
  `vulkan/pipeline.rs:413-423` (`attr.semantic.index()`; fixed table Position=0…Weights=7 in
  `core/src/mesh/layout.rs:79-90`)
- **Category:** divergence
- The same `VertexLayout` produces different attribute locations per backend: wgpu assumes
  the shader's `@location`s follow struct-declaration order; Vulkan assumes SPIR-V locations
  equal the semantic index.
- **Failure:** layout `[Position, TexCoord0]` → locations {0,1} on wgpu, {0,3} on Vulkan.
  Any shader satisfies at most one convention — UVs/normals read from wrong offsets,
  geometry garbled on exactly one backend.
- **Fix:** pick one convention (semantic indices fit better — shaders compile once) and
  align the other backend.

## High

### [x] XB-H1. Vulkan ignores `VertexStepMode` — instancing broken on Vulkan only
- **Where:** `vulkan/pipeline.rs:401-411` (`input_rate(VERTEX)` hardcoded) vs
  `wgpu_impl/resources.rs:240-249` (`convert_step_mode`)
- **Category:** divergence / bug — details in VK-H9.

### [x] XB-H2. `StorageBuffer` read-only on wgpu, read-write on Vulkan
- **Where:** `wgpu_impl/conversion.rs:501-505` vs `vulkan/pipeline.rs:261`
- **Category:** divergence — details in WG-H1. Writable-storage compute works on Vulkan,
  fails pipeline creation on wgpu.

### [ ] XB-H3. `write_buffer` timing semantics differ; trait contract silent
- **Where:** `vulkan/mod.rs:1361-1391` (immediate memcpy into mapped memory) vs
  `wgpu_impl/resources.rs:518-532` (`queue.write_buffer`, staged & queue-ordered); mid-graph
  op: `wgpu_impl/pass_encoding.rs:400-419` executes before all passes of the submission,
  Vulkan encodes at the correct graph position
- **Category:** divergence / underspecified contract
- **Failure:** updating a uniform while the previous frame is in flight: wgpu applies the
  write for subsequent submissions only; Vulkan mutates memory the executing frame reads →
  flicker on Vulkan only. Mid-graph transfer writes are reordered on wgpu (WG-H4). The trait
  doc ("Write data to a buffer", `backend/mod.rs`) specifies no timing semantics — define
  them, then fix both backends to match.

### [ ] XB-H4. `LoadOp::DontCare`: true `DONT_CARE` on Vulkan, `Load` on wgpu
- **Where:** `vulkan/conversion.rs:234-235`, `:258-259`, `:288-289` vs
  `wgpu_impl/conversion.rs:294`, `:314`, `:331`
- **Category:** divergence
- **Failure:** a pass declaring `DontCare` whose shader writes only part of the target:
  previous contents preserved on wgpu, undefined garbage on Vulkan (especially tilers) —
  "works in dev on wgpu, broken on Vulkan". Also `Clear` with a mismatched `ClearValue`
  silently degrades to `Load` on **both** backends instead of erroring
  (`wgpu_impl/conversion.rs:303-305`, `vulkan/conversion.rs:247-249`).

### [x] XB-H5. Blending on integer color targets: wgpu disables it, Vulkan enables unconditionally
- **Where:** `wgpu_impl/resources.rs:219-235` (`format.is_integer()` → `blend: None`) vs
  `vulkan/pipeline.rs:473-485` + `vulkan/conversion.rs:372-387` (`blend_enable(true)`
  whenever the material has blend state, no format check)
- **Category:** divergence / bug
- **Failure:** blended material rendering to an `R32Uint` picking/ID buffer: works on wgpu,
  invalid pipeline on Vulkan.

## Medium

### [x] XB-M1. `Bgra10a2Unorm` channel order swapped between backends
- **Where:** `wgpu_impl/conversion.rs:66` vs `vulkan/conversion.rs:64` — details in WG-M1.

### [ ] XB-M2. Fence semantics inverted for freshly created unsignaled fences
- **Where:** `wgpu_impl/resources.rs:424-431`, `:460-485` (reports signaled when queue empty)
  vs `vulkan/mod.rs:1039-1055` (truly unsignaled); documented at `backend/mod.rs:221-246`
- **Category:** divergence
- `create_fence(false)` polls as signaled on wgpu, unsignaled on Vulkan. Any caller polling
  before first submit behaves differently per backend.
- **Related:** on submit failure in `FrameSchedule::render` (`scheduler/mod.rs:234-246`) the
  Vulkan fence has been reset and never signals → every subsequent `begin_frame` stalls the
  full 10 s `wait_fence` timeout; wgpu doesn't stall.

### [x] XB-M3. `wait_fence_timeout` semantics differ
- **Where:** `wgpu_impl/resources.rs:490-510` (returns `is_queue_empty()`, reflecting *all*
  queue work) vs `vulkan/mod.rs:1094-1111` (per-fence result)
- **Category:** divergence / bug — details in WG-H6.

### [ ] XB-M4. Depth/raster state hardcoded identically in both backends — fragile equilibrium
- **Where:** `vulkan/pipeline.rs:450-471` vs `wgpu_impl/resources.rs:269-287`
- **Category:** hardcode
- `LessEqual`, depth-write always on, cull off, 1 sample — consistent today, but
  `MaterialDescriptor` cannot express read-only depth (transparents), reversed-Z, culling,
  or MSAA, so the first backend-local change becomes an instant divergence. Note: Vulkan
  `front_face = CLOCKWISE` (Y-flip compensation) vs wgpu `Ccw` only cancels out because
  culling is `None` — enabling culling naïvely on one side breaks winding.

### [ ] XB-M5. Vulkan transfer `TextureToTexture` hardcodes COLOR aspect and layer 0
- **Where:** `vulkan/mod.rs:2619-2651` (ignores `origin.z` as layer, aspect `COLOR`) vs
  `wgpu_impl/pass_encoding.rs:551-595` (`TextureAspect::All`, honors `origin.z`)
- **Category:** divergence
- **Failure:** copying depth textures or between array layers/cube faces works on wgpu, is
  wrong/invalid on Vulkan (z offset on a 2D image + wrong aspect).

### [ ] XB-M6. `read_buffer` of GPU-only buffers: real data on wgpu, silent zeros on Vulkan
- **Where:** `vulkan/mod.rs:1394-1407` vs `wgpu_impl/resources.rs:591-653` (staging-copy
  fallback); trait contract at `backend/mod.rs:912-915` promises blocking semantics neither
  fully honors
- **Category:** divergence — details in VK-L14 / WG-M5.

### [ ] XB-M7. `DeviceCapabilities` hardcoded, never queried from the adapter
- **Where:** `graphics/src/device.rs:35-45` (`max_texture_dimension: 16384`,
  `max_buffer_size: 1 GB`) while wgpu is created with `Limits::default()` (8192 / 256 MB)
- **Category:** hardcode / divergence
- **Failure:** a 16K texture passes engine validation (`device.rs:167-175`) then panics with
  a wgpu validation error; the same texture works on Vulkan.

## Low

### [ ] XB-L1. `execute_graph(signal_fence: None)` has opposite blocking semantics
- **Where:** `wgpu_impl/mod.rs:299-305` (blocks up to 10 s) vs `vulkan/mod.rs:1119-1136`
  (returns immediately); trait doc at `backend/mod.rs:870-894`
- Both are individually documented and the sole production caller (`scheduler/mod.rs:239`)
  always passes a fence, but any new caller gets a synchronous backend on wgpu and an
  unsynchronized one on Vulkan. Unify in the trait-level contract.

### [ ] XB-L2. `supported_present_modes()` hardcodes all four modes
- **Where:** `graphics/src/swapchain.rs:229-237`
- Vulkan silently falls back to FIFO for unsupported modes (`vulkan/swapchain.rs:75-81`);
  wgpu panics inside `surface.configure`. Same app choice of `Mailbox` → silent vsync change
  on Vulkan, crash on wgpu.

### [ ] XB-L3. `enumerate_adapters` hardcodes a single "Dummy Adapter"
- **Where:** `graphics/src/instance.rs:298-306`
- Adapter selection is a no-op; multi-GPU selection impossible; device name misleads logs.
  On Vulkan, device selection never re-checks surface presentability (only errors at
  `ensure_compatible_with_surface`, `backend/mod.rs:1035-1045`), while wgpu re-requests a
  compatible adapter (`wgpu_impl/mod.rs:166-221`, with the orphaned-resources issue WG-M6)
  — divergent multi-GPU behavior.

### [ ] XB-L4. `wgpu::Surface<'static>` obtained via `std::mem::transmute`
- **Where:** `backend/mod.rs:957-971`
- Lifetime erased with only a doc comment as the contract; dropping the window before the
  `Surface` is a use-after-free with no compile- or run-time check. Consider
  `Arc<Window>`-based surface creation.

### [ ] XB-L5. Unknown `TextureFormat` silently falls back to `Rgba8Unorm` in both converters
- **Where:** `wgpu_impl/conversion.rs:223`; `vulkan/conversion.rs:137`
- Currently dead arms (all variants covered), but `TextureFormat` is `#[non_exhaustive]`
  (`core/src/texture/mod.rs:53`) — the first new format silently aliases to RGBA8 on both
  backends instead of failing loudly. Details in VK-L4 / WG-M2.

### [ ] XB-L6. MSAA sample counts accepted at texture creation, never plumbed into pipelines
- **Where:** `vulkan/mod.rs:747-753` (counts ∉ {1,2,4,8} silently → 1);
  `vulkan/pipeline.rs:462-464` and `wgpu_impl/resources.rs:288` always build 1-sample pipelines
- An MSAA texture attached to a pass mismatches the pipeline on both backends; wgpu errors
  loudly at creation for count 16, Vulkan silently downgrades — divergent failure modes.
