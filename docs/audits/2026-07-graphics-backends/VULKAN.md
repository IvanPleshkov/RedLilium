# Vulkan Backend Findings

Files: `graphics/src/backend/vulkan/` — `mod.rs`, `barriers.rs`, `swapchain.rs`, `pipeline.rs`,
`layout.rs`, `conversion.rs`, `device.rs`, `instance.rs`, `command.rs`, `allocator.rs`, `debug.rs`.
Line numbers refer to the tree at commit `1d6ee28`.

## Critical

### [ ] VK-C1. `is_fence_signaled` returns `true` for unsignaled fences
- **Where:** `vulkan/mod.rs:1083-1089`
- **Category:** bug (synchronization)
- ash 0.38 `get_fence_status` returns `Ok(false)` for `VK_NOT_READY`; the code checks
  `.is_ok()`, which is `true` either way.
- **Failure:** any caller polling the fence (non-blocking frame pacing, deferred destruction)
  believes GPU work finished immediately after submit → resources reused/freed while still
  executing.
- **Fix:** `matches!(device.get_fence_status(*fence), Ok(true))`.

### [ ] VK-C2. `wait_fence` swallows timeout and errors; callers then recycle in-flight resources
- **Where:** `vulkan/mod.rs:1061-1080` (10 s timeout), `mod.rs:387-411` (`advance_frame`)
- **Category:** bug (synchronization / error handling)
- On `TIMEOUT` or any error (including `ERROR_DEVICE_LOST`) the function logs and returns
  normally; the caller cannot distinguish "signaled" from "still running". `advance_frame`
  then resets the slot's command pool and descriptor pools while their command buffers /
  descriptor sets may still be executing.
- **Failure:** any GPU hitch > 10 s (shader compile storm, debugger attach) → command pool
  reset of in-flight buffers → `VK_ERROR_DEVICE_LOST`.
- **Fix:** return a `Result` and propagate; halt frame advance on timeout/device-lost.

### [ ] VK-C3. Buffer read barriers hardcode `TransferWrite` as source — compute writes never synchronized
- **Where:** `vulkan/mod.rs:1314-1353` (read: `1340-1352`, write: `1330-1339`), consumed by
  `barriers.rs:116-142`
- **Category:** bug (synchronization) — independently confirmed by 3 audit passes
- For reads the previous writer is assumed to be `BufferAccessMode::TransferWrite`
  (`srcStage=TRANSFER`, `srcAccess=TRANSFER_WRITE`), which does **not** cover
  `SHADER_WRITE` at compute/vertex/fragment stages despite the "conservative" comment.
  For writes, `src == dst == the new access`, so the barrier doesn't cover whatever actually
  happened before (no WAR dependency against prior reads either).
- **Failure:** compute pass writes a storage buffer; next pass reads it as vertex/uniform —
  no execution dependency on the compute stage → stale/garbage data, intermittent per GPU.
  This is exactly the GPU-driven pattern the compute-pass API invites.
- **Fix:** track the last access per buffer (analogous to `TextureLayoutTracker`) and emit
  precise src scopes. Also fixes the perf issue VK-P7 (redundant barriers every frame).

## High

### [ ] VK-H1. Same-layout transitions are skipped, dropping WAW/RAW barriers between passes
- **Where:** `barriers.rs:92-95` (early return when `old_layout == new_layout`); state update
  at `mod.rs:1301-1311`
- **Category:** bug (synchronization)
- Dynamic rendering has no implicit inter-pass dependencies, so two consecutive passes using
  an image in the same layout get no barrier at all.
- **Failure:** pass A renders to texture T, pass B renders to T again with `LoadOp::Load`
  (layout stays `ColorAttachment`) — WAW hazard; two compute passes read/writing the same
  storage image in `General` layout race each other.

### [ ] VK-H2. Every graphics pass writing the surface re-transitions it from `UNDEFINED`
- **Where:** `mod.rs:1877-1914`
- **Category:** bug
- The per-pass surface barrier hardcodes `old_layout = UNDEFINED` ("OK since we're clearing"),
  which legally discards prior contents — regardless of load op, and for every pass in the
  frame, not just the first.
- **Failure:** scene pass + UI overlay pass with `LoadOp::Load` on the surface: the second
  `UNDEFINED` transition discards the scene on drivers that exploit the discard (mobile, AMD).
  `SurfaceAccess::ReadWrite` (`graph/resource_usage.rs:303-308`) is unimplementable.

### [ ] VK-H3. Acquire semaphore not chained into the surface layout transition
- **Where:** barrier `srcStage = TOP_OF_PIPE` at `mod.rs:1899-1912`; submit waits
  `image_available` at `COLOR_ATTACHMENT_OUTPUT` (`mod.rs:1203-1210`)
- **Category:** bug (WSI synchronization)
- The transition's first sync scope is not ordered after the semaphore wait — the canonical
  WSI hazard (Khronos sync examples require `srcStage = COLOR_ATTACHMENT_OUTPUT` here).
- **Failure:** transition may execute while the presentation engine still reads the image;
  sync-validation failure, real corruption on tilers.

### [ ] VK-H4. `write_texture` uploads only mip 0 / layer 0 / COLOR aspect and bypasses the layout tracker
- **Where:** `mod.rs:1416-1717` (barrier subresource `1567-1573`, copy region `1590-1605`,
  final transition `1617-1644`); tracker default at `layout.rs:449-454`
- **Category:** bug
- The copy hardcodes `mip_level: 0`, `layer_count: 1`, `depth: 1`, aspect `COLOR`, tight
  packing. After the upload the image really is in `SHADER_READ_ONLY_OPTIMAL`, but
  `layout_tracker.set_layout` is never called, so the tracker still says `Undefined`.
- **Failure:** (a) cube maps / arrays / mips / 3D / depth formats: only the first 2D slice is
  uploaded, the rest stays uninitialized (and un-transitioned → validation errors);
  (b) the first render-graph use emits `UNDEFINED → SHADER_READ_ONLY`, which per spec may
  **discard the just-uploaded texels** — works on desktop by luck, breaks on tilers;
  (c) post-copy dst stage is `FRAGMENT_SHADER` only (`mod.rs:1638`) — vertex/compute sampling
  unsynchronized.

### [ ] VK-H5. Acquire failure path leaves the in-flight fence reset and unsignaled → deadlock
- **Where:** `swapchain.rs:319-358` (fence reset at `:330`, `begin_swapchain_frame` at `:342`
  **before** `vkAcquireNextImageKHR`)
- **Category:** bug (swapchain)
- If acquire returns `ERROR_OUT_OF_DATE_KHR` (resize/minimize), the slot's fence is left
  permanently unsignaled and stale semaphores stay registered in `swapchain_sync`.
- **Failure:** retry without reconfigure → next `wait_for_fences(..., u64::MAX)` on that slot
  never returns (render thread hangs). Reconfigure → semaphores registered by
  `begin_swapchain_frame` are destroyed while still registered; a later submit can consume
  dangling handles.
- **Related:** a frame that acquires but never presents (early error return, panic-unwind,
  headless tick) leaves the fence unsignaled the same way — the fence signal should not
  depend on the present path.

### [ ] VK-H6. Descriptor pool lacks `UNIFORM_BUFFER_DYNAMIC` capacity
- **Where:** `pipeline.rs:54-79` (pool template); layout maps `DynamicUniformBuffer →
  UNIFORM_BUFFER_DYNAMIC` at `pipeline.rs:260`; writes at `mod.rs:2157-2159`
- **Category:** bug (descriptor pool vs layout mismatch)
- **Failure:** on spec-conformant drivers (mobile, MoltenVK, best-practices validation),
  allocating a set with a dynamic UBO binding returns `ERROR_OUT_OF_POOL_MEMORY`; the
  grow-once retry creates an identical pool lacking the type and fails again — every draw
  with a dynamic-UBO material errors out. Works on desktop drivers that ignore pool typing.

### [ ] VK-H7. OUT_OF_DATE / SUBOPTIMAL swallowed — no recreation signal ever reaches the app
- **Where:** `swapchain.rs:345` (`_suboptimal` ignored), `:548-558` (present maps
  `SUBOPTIMAL_KHR` and `ERROR_OUT_OF_DATE_KHR` to `Ok(())`); caller drops errors at
  `backend/mod.rs:403-416`
- **Category:** bug (swapchain / error handling)
- **Failure:** on X11/Windows where resize yields `SUBOPTIMAL_KHR`, the engine presents a
  stale-extent swapchain indefinitely (stretched output). Where OUT_OF_DATE occurs, the frame
  is declared presented and the problem surfaces later as the VK-H5 deadlock.

### [ ] VK-H8. Pipelines never declare a stencil attachment format, but the encoder records one
- **Where:** `pipeline.rs:505-507` (no `.stencil_attachment_format`) vs `mod.rs:1823-1859`
  (stencil attachment recorded when `format().has_stencil()`)
- **Category:** bug (dynamic rendering format matching)
- **Failure:** any draw into a `Depth24PlusStencil8` / `Depth32FloatStencil8` target violates
  VUID dynamic-rendering format matching (`stencilAttachmentFormat = UNDEFINED` vs an actual
  stencil attachment) — validation error, UB on format-sensitive drivers. Also
  `stencil_test_enable(false)` at `pipeline.rs:471` makes stencil permanently unusable.

### [x] VK-H9. `VertexStepMode` ignored — per-instance vertex buffers broken
- **Where:** `pipeline.rs:401-411` (`.input_rate(vk::VertexInputRate::VERTEX)` hardcoded)
- **Category:** bug / cross-backend divergence (see XB-C2)
- **Failure:** instanced rendering with a `VertexStepMode::Instance` buffer advances per
  vertex on Vulkan (per instance on wgpu) — instances read wrong transforms on Vulkan only.

## Medium

### [x] VK-M1. Host writes to mapped buffers have no frame synchronization
- **Where:** `mod.rs:1361-1391` (`write_buffer`), `mod.rs:2377-2411` (`WriteBuffer` transfer op)
- **Category:** bug
- Both paths memcpy into the single persistently-mapped allocation with no per-frame
  buffering or fence check while up to `MAX_FRAMES_IN_FLIGHT` frames still reference the
  buffer. Also diverges from wgpu, where `queue.write_buffer` is queue-ordered (XB-M2).
- **Failure:** per-frame uniform updates (camera matrix) race the previous frame's GPU reads
  → geometry jitter. Needs per-slot ring buffering or a fence-guarded path.

### [ ] VK-M2. Negative scissor offsets passed through to `vkCmdSetScissor`
- **Where:** `mod.rs:2249-2266`
- **Category:** bug — Vulkan requires `offset >= 0` (VUID-vkCmdSetScissor-x-00595).
- **Failure:** egui-style clip rect partially above/left of the viewport → validation error,
  UB on some drivers. Clamp to 0 and shrink the extent.

### [ ] VK-M3. Silent surface-format fallback mismatches every pipeline
- **Where:** `swapchain.rs:68-72` (`.unwrap_or(formats[0])`)
- **Category:** bug + dangerous hardcode
- If the requested format isn't supported, an arbitrary format *and color space* is silently
  substituted while all pipelines were compiled for the requested one.
- **Failure:** request `Bgra8UnormSrgb`, driver lists `R8G8B8A8_UNORM` first → attachment
  format mismatch (validation error / double gamma / swapped channels). Compounded by
  `Surface::supported_formats()` falling back to `[Bgra8Unorm]` (`graphics/src/swapchain.rs:186-196`),
  which lets caller-side validation pass for unsupported formats.

### [ ] VK-M4. Present barrier hardcodes `old_layout = COLOR_ATTACHMENT_OPTIMAL` even when nothing rendered
- **Where:** `swapchain.rs:467-482` vs the "nothing rendered" fallback at `:509-513`
- **Category:** bug
- **Failure:** a frame that acquires but doesn't touch the swapchain (loading screen,
  headless tick): actual layout is `UNDEFINED`/`PRESENT_SRC_KHR`, not color-attachment —
  validation error, UB on tilers using layout for (de)compression.

### [ ] VK-M5. Attachment dst access masks omit read bits
- **Where:** `layout.rs:97-109`
- **Category:** bug (synchronization)
- `ColorAttachment` dst access lacks `COLOR_ATTACHMENT_READ` (needed for blending and
  `LoadOp::Load`); `DepthStencilAttachment` lacks `DEPTH_STENCIL_ATTACHMENT_READ` (depth
  test against loaded depth); `DepthStencilReadOnly` lacks `SHADER_READ` (sampling).
- **Failure:** `ShaderReadOnly → ColorAttachment` + alpha blending: the blend unit's read
  isn't in the visibility scope — intermittent stale blends. Shadow-map sampling has the
  same gap.

### [ ] VK-M6. Sampled-texture descriptors hardcode `SHADER_READ_ONLY_OPTIMAL`
- **Where:** `mod.rs:2098`, `mod.rs:2127` (and the compute mirror)
- **Category:** bug
- The barrier system legitimately places depth textures in `DEPTH_STENCIL_READ_ONLY_OPTIMAL`
  (`TextureAccessMode::DepthStencilReadOnly`), but descriptors always claim
  `SHADER_READ_ONLY_OPTIMAL`.
- **Failure:** shadow mapping (declare depth as `DepthStencilReadOnly`, sample it) →
  VUID-VkDescriptorImageInfo-imageLayout violation, undefined sampling on some hardware.

### [ ] VK-M7. `UniformRead` barrier stages omit `COMPUTE_SHADER`
- **Where:** `graph/resource_usage.rs:163-165`, `:183-185` (consumed by `barriers.rs:140-141`)
- **Category:** bug (synchronization)
- **Failure:** transfer pass updates a UBO, compute pass reads it as `UniformRead` — the
  dst stage doesn't block compute → stale uniforms in the dispatch.

### [ ] VK-M8. Barriers ignore declared mip/layer ranges — whole-image transitions only
- **Where:** `barriers.rs:182-188` (`REMAINING_MIP_LEVELS` / `REMAINING_ARRAY_LAYERS`);
  `TextureUsageDecl.mip_level/layer_count` (`resource_usage.rs:202-215`) never reach the barrier
- **Category:** bug
- **Failure:** mipmap generation (write mip N, sample mip N−1) or per-face cubemap rendering:
  the whole-image barrier's `old_layout` is wrong for half the subresources — validation
  errors, corrupted mips.

### [ ] VK-M9. `MAX_FRAMES_IN_FLIGHT = 3` never enforced against user-configurable frames-in-flight
- **Where:** `mod.rs:33`, `mod.rs:387-411`; `pipeline/mod.rs:162-167` accepts any value > 0;
  `SurfaceConfiguration::with_frames_in_flight` too
- **Category:** dangerous hardcode
- **Failure:** `RenderPipeline::new(4)`: `advance_frame` resets the pool of a slot whose
  command buffers are still executing → device lost. Needs an assert/clamp.

### [ ] VK-M10. Device features enabled without querying support
- **Where:** `device.rs:121-143` (`fill_mode_non_solid`, `shaderDrawParameters` unconditional);
  `VK_KHR_dynamic_rendering` support never verified either
- **Category:** dangerous hardcode
- **Failure:** GPU lacking `fillModeNonSolid` → `vkCreateDevice` returns
  `ERROR_FEATURE_NOT_PRESENT` and the whole backend fails to initialize, though the features
  are only needed for wireframe/instance-id paths.

### [ ] VK-M11. Linux instance hardcodes Xlib + Wayland surface extensions
- **Where:** `instance.rs:56-105` (Linux block `:72-76`)
- **Category:** dangerous hardcode
- Extensions pushed unconditionally, never checked against
  `enumerate_instance_extension_properties`; no headless path; `VK_KHR_xcb_surface` missing.
- **Failure:** Wayland-only system (no `VK_KHR_xlib_surface` in the loader) or headless CI:
  `vkCreateInstance` fails → backend unavailable even for offscreen rendering.

### [ ] VK-M12. Anisotropy and sample counts not clamped to device limits; pipelines always 1-sample
- **Where:** `mod.rs:857-859` (anisotropy unclamped), `mod.rs:747-753` (sample count 2/4/8
  unchecked, others silently → 1), `pipeline.rs:462-464` (`rasterization_samples = TYPE_1`)
- **Category:** dangerous hardcode
- **Failure:** `anisotropy_clamp = 16` on a max-8 device → validation error/UB. Any MSAA
  attachment mismatches its single-sample pipeline → broken rendering.

### [ ] VK-M13. Layout tracker grows without bound
- **Where:** `mod.rs:37` (monotonic texture ids), `layout.rs:421-458` (no removal API),
  `backend/mod.rs:649-671` (`GpuTexture::drop` doesn't unregister)
- **Category:** resource leak
- **Failure:** editor session with continuous render-target resizes / asset hot-reload:
  the tracker HashMap grows for the process lifetime, degrading lock-held lookups in
  `generate_barriers_for_pass`.

### [ ] VK-M14. Transfer ops guess 256-byte row alignment; block-compressed row length wrong
- **Where:** `mod.rs:2450-2468`, `:2545-2560` (transfer ops) vs `:1590-1605` (`write_texture`
  uses tight packing)
- **Category:** dangerous hardcode
- When `bytes_per_row` is unspecified and `height > 1`, buffer↔image copies assume wgpu's
  256-byte row alignment — a guess about caller memory layout, inconsistent with
  `write_texture`. `bytes_per_row / block_size` is also wrong for block-compressed formats
  (row length must be texels computed from blocks).
- **Failure:** tightly-packed readback buffer sized `w*h*4` with `bytes_per_row` omitted:
  copy strides at 256 bytes → sheared image + out-of-bounds access for widths not divisible
  by 64 texels.

### [ ] VK-M15. `Depth24PlusStencil8 → D24_UNORM_S8_UINT` without format-support query
- **Where:** `conversion.rs:76`
- **Category:** dangerous hardcode
- `D24_UNORM_S8_UINT` is optional and commonly missing on AMD.
- **Failure:** texture creation fails at runtime on those devices (wgpu emulates the format
  transparently — divergence). Fall back to `D32_SFLOAT_S8_UINT` via
  `vkGetPhysicalDeviceFormatProperties`.

### [ ] VK-M16. `composite_alpha` and `image_usage` hardcoded without capability check
- **Where:** `swapchain.rs:115-118` (`OPAQUE`, `COLOR_ATTACHMENT` only)
- **Category:** dangerous hardcode
- **Failure:** surfaces reporting only `INHERIT`/`PRE_MULTIPLIED` (common on Android/Wayland):
  `vkCreateSwapchainKHR` fails at startup, despite capabilities having been queried two lines
  earlier. Missing `TRANSFER_SRC` usage also precludes swapchain readback (screenshots).

### [ ] VK-M17. Failed graph submit leaks the consumed swapchain semaphore pair → present hangs
- **Where:** `mod.rs:1203-1250` (`take_swapchain_render_sync` marks consumed before
  `vkQueueSubmit`); present wait at `swapchain.rs:509-513`
- **Category:** bug (error handling)
- **Failure:** transient `OUT_OF_DEVICE_MEMORY` on submit: `image_render_finished` never
  signals but `swapchain_render_consumed()` returns true → present waits forever; the
  acquire's `image_available` signal is orphaned, poisoning the semaphore's next use.
  Related: the frame fence was reset before the failed submit and never signals → every
  subsequent `begin_frame` stalls the full 10 s timeout (see `scheduler/mod.rs:234-246`).

## Performance

### [ ] VK-P1. All `COPY_DST` buffers live in host-visible memory — no staging path to device-local
- **Where:** `mod.rs:586-603`; all mesh buffers get `COPY_DST` (`graphics/src/device.rs:451, 461`)
- **Severity:** high (perf)
- Every vertex/index/uniform buffer, including write-once static geometry, is allocated
  `CpuToGpu` and never migrated to `DEVICE_LOCAL`.
- **Impact:** on discrete GPUs every draw fetches geometry over PCIe BAR instead of VRAM —
  silent frame-rate loss scaling with scene size; invisible on integrated dev machines.

### [ ] VK-P2. Per-draw descriptor set allocation + write
- **Where:** `mod.rs:2039-2206` (graphics), `mod.rs:2713-2872` (compute)
- **Severity:** high (perf)
- Every draw allocates fresh descriptor sets and calls `vkUpdateDescriptorSets` even for
  unchanged material bindings; a `Vec<WriteDescriptorSet>` (`mod.rs:2135`) and a flattened
  dynamic-offset `Vec` (`mod.rs:2217-2218`) are heap-allocated per draw.
- **Impact:** 5k draws/frame → 5k+ allocations/updates per frame dominating CPU encode time.
  Contradicts the binding-frequency classification (Decision 7). Cache sets per
  (material instance, resource version).

### [ ] VK-P3. `write_texture`: full synchronous queue wait + staging/fence recreation per upload
- **Where:** `mod.rs:1665-1714`
- **Severity:** medium (perf)
- **Impact:** level load with hundreds of textures serializes N full GPU round-trips;
  mid-gameplay streaming upload hitches the frame by a full queue flush. Batch uploads on a
  transfer queue / reuse staging memory.

### [ ] VK-P4. No `VkPipelineCache`; no descriptor-set-layout dedup
- **Where:** `pipeline.rs:522-524`, `:559-561` (`PipelineCache::null()`);
  `create_descriptor_set_layout` per material with no dedup
- **Severity:** medium (perf)
- **Impact:** every run recompiles every pipeline (multi-second editor startup with many
  materials); identical layouts duplicated per material. Add an in-process + on-disk cache.

### [ ] VK-P5. `cull_mode = NONE`, depth state hardcoded in all pipelines
- **Where:** `pipeline.rs:458` (cull), `:459` (front face CW for Y-flip), `:466-471`
  (depth `LESS_OR_EQUAL`, write always on)
- **Severity:** medium (perf + expressiveness)
- **Impact:** every closed mesh shades ~2× fragments; transparent materials can't disable
  depth writes; reversed-Z impossible. NOTE: front-face conventions differ from wgpu (CW vs
  CCW) and currently only cancel out because culling is off — see XB-M4 before enabling.

### [ ] VK-P6. `FREE_DESCRIPTOR_SET` flag on pools that are only ever bulk-reset
- **Where:** `pipeline.rs:42`
- **Severity:** low (perf)
- Forces drivers into a general-purpose allocator instead of a linear one and enables the
  `ERROR_FRAGMENTED_POOL` path the code then handles (`pipeline.rs:336-343`).

### [ ] VK-P7. Redundant conservative buffer barriers every pass, every frame
- **Where:** `mod.rs:1314-1353`; stage-mask union at `barriers.rs:139-141`
- **Severity:** medium (perf; correctness aspect is VK-C3)
- No per-buffer state tracking → every read re-emits a `TRANSFER→X` barrier, and its stages
  join the batch-wide union, broadening every other barrier in the batch.
- **Impact:** a static scene with 50 buffers per pass issues ~50 no-op barriers per pass per
  frame with a merged src mask that drains the pipeline between every pass.

### [ ] VK-P8. Swapchain recreation stalls the whole device; `old_swapchain` unused
- **Where:** `swapchain.rs:121` (`old_swapchain(null)`), `:272` (`device_wait_idle` in
  destroy); destroy-then-create ordering at `backend/mod.rs:536-544`
- **Severity:** low (perf)
- **Impact:** every resize is a full GPU drain + visible hitch; passing the old swapchain
  allows seamless recreation.

### [ ] VK-P9. Minor per-frame allocations
- **Where:** `mod.rs:1195-1197` (semaphore Vecs), `barriers.rs:42-51` + `mod.rs:1270` (two
  fresh HashMaps per pass), `swapchain.rs:380-383` (Arc + device clone per acquire),
  `command.rs:12-14` (`RESET_COMMAND_BUFFER` on pools that are only bulk-reset)
- **Severity:** low (perf)

## Low

### [ ] VK-L1. Replaced barriers leave stale stage masks; intermediate transition dropped
- **Where:** `barriers.rs:106-108` (`HashMap::insert` replaces), `:139-141` (OR-ed masks
  never recomputed)
- If one pass declares the same texture with two access modes, the second barrier replaces
  the first but tracked `set_layout` sequencing keeps the first's — surviving barrier's
  `old_layout` mismatches actual layout. Unusual declaration required, hence low.

### [ ] VK-L2. Dead duplicate `generate_barriers` keyed by raw `vk::Image`
- **Where:** `barriers.rs:248-294`
- Contradicts the handle-reuse warning in `layout.rs:413-418`. If ever wired up, recycled
  handles inherit stale layouts. Delete it.

### [ ] VK-L3. `LoadOp::Clear` with mismatched `ClearValue` silently degrades to `LOAD`
- **Where:** `conversion.rs:247-249`
- First frame then loads an UNDEFINED-layout image's garbage. Should be an error.

### [ ] VK-L4. Silent `_ => R8G8B8A8_UNORM` catch-all in `convert_texture_format`
- **Where:** `conversion.rs:137`
- Currently unreachable (all 76 variants covered — verified), but `TextureFormat` is
  `#[non_exhaustive]`: the first new variant silently renders as RGBA8. Make it exhaustive
  or return an error. Same issue in wgpu (WG-M2).

### [ ] VK-L5. No push-constant support in pipeline layouts
- **Where:** `pipeline.rs:295-311`
- Any shader declaring push constants fails pipeline validation with a confusing error.

### [ ] VK-L6. `spirv_entry_point_name` returns the first `OpEntryPoint`
- **Where:** `pipeline.rs:663-695`
- A module with multiple entry points (vertex+fragment compiled together) yields the wrong
  `pName` for the second stage.

### [ ] VK-L7. Infinite waits (`u64::MAX`) on fences and acquire
- **Where:** `swapchain.rs:323`, `:348`; `mod.rs:1704`
- A device hang freezes the app permanently; no device-lost recovery path.

### [ ] VK-L8. Queue family / physical device selection has no fallback
- **Where:** `device.rs:74-90` (first GRAPHICS family chosen without present-support check),
  `device.rs:12-71` (required extensions / API version never verified during scoring)
- The highest-scored device can fail `vkCreateDevice` while a lesser device would work;
  `ensure_compatible_with_surface` hard-fails later with no family/device fallback search.

### [ ] VK-L9. Binding-group/layout `zip` silently drops mismatched trailing groups
- **Where:** `mod.rs:2044-2047`, `:2140-2146` (missing lookups default to `UNIFORM_BUFFER`)
- A material/instance mismatch renders with unbound sets (UB) instead of erroring.

### [ ] VK-L10. API version 1.3 requested unconditionally (non-macOS)
- **Where:** `instance.rs:14-18`
- No `try_enumerate_instance_version` check; a 1.2-only loader is unhandled.

### [ ] VK-L11. Debug messenger enables INFO severity; flag match mislabels combined types
- **Where:** `debug.rs:14-18`, `:60-65`
- Loader/driver INFO chatter on every validation run; exact-equality match on
  `message_type` labels combined flags "Unknown".

### [ ] VK-L12. Minimized-window (0×0) extent not guarded
- **Where:** `swapchain.rs:84-97`
- `vkCreateSwapchainKHR` with zero extent violates the spec; `configure()` on a minimize
  event fails/UB instead of being skipped.

### [ ] VK-L13. `create_fence` panics on failure
- **Where:** `mod.rs:1048-1049` (`.expect(...)`; trait signature forces it)
- `VK_ERROR_OUT_OF_HOST_MEMORY` aborts the process. Change the trait to return `Result`.

### [ ] VK-L14. `read_buffer` violates its documented blocking contract; zeros for GPU-only buffers
- **Where:** `mod.rs:1394-1407` (contract at `backend/mod.rs:912-915`)
- No GPU wait despite the "blocking" doc; unmapped buffers silently return zeros (wgpu
  returns real data via a staging copy — divergence, see XB-M6).
