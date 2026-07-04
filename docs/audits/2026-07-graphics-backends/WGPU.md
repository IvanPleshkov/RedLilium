# wgpu Backend Findings

Files: `graphics/src/backend/wgpu_impl/` — `mod.rs`, `pass_encoding.rs`, `resources.rs`,
`conversion.rs`, `swapchain.rs`; plus `graphics/src/instance.rs` where noted.
Line numbers refer to the tree at commit `1d6ee28`.

Cross-cutting: no `device.on_uncaptured_error` handler is installed anywhere, so every
validation error listed below is a process abort rather than a recoverable `GraphicsError`.
There is also no staging-belt path — `queue.write_buffer` is used everywhere, including
per-frame uniform streams.

## Critical

### [ ] WG-C1. `pollster::block_on` in backend creation cannot work on wasm32
- **Where:** `wgpu_impl/mod.rs:104`, `:116`, `:182`, `:200`
- **Category:** bug (web)
- `request_adapter`/`request_device` are driven by parking the calling thread — impossible on
  `wasm32-unknown-unknown`, where the browser event loop must run for the future to resolve.
  There is no `cfg(target_arch = "wasm32")` async path anywhere in `wgpu_impl`.
- **Failure:** the wasm-pack web build panics or hangs the tab permanently at
  `WgpuBackend::with_params()`.

### [ ] WG-C2. Device creation hardcodes a native-only feature and default limits
- **Where:** `wgpu_impl/mod.rs:118-119` (duplicated at `:202-204`)
- **Category:** dangerous hardcode
- `required_features: POLYGON_MODE_LINE` (unavailable on WebGPU and WebGL) and
  `required_limits: Limits::default()` (exceeds WebGL2 downlevel limits) with no adapter
  capability check.
- **Failure:** on the web target `request_device` always errors — the wgpu backend can never
  be created in a browser; also fails on native GL / older adapters.
- **Fix:** intersect with `adapter.features()`; use
  `Limits::downlevel_webgl2_defaults().using_resolution(adapter.limits())` where applicable.

### [ ] WG-C3. Blocking `device.poll(Wait)` + blocking channel `recv` unsupported on wasm
- **Where:** `wgpu_impl/mod.rs:300-303` (`execute_graph` no-fence path);
  `resources.rs:443-447`, `:499-504`, `:600`, `:630-633`, `:642-643` (`wait_fence`,
  `wait_fence_timeout`, `read_buffer`)
- **Category:** bug (web)
- On wasm, `poll` cannot block and map callbacks only fire when control returns to the
  browser event loop.
- **Failure:** on web, `execute_graph(None)` silently doesn't wait (breaking the documented
  "GPU is idle" invariant `FrameSchedule::finish` relies on for recycling), and `read_buffer`
  deadlocks the tab (`map_async` callback can never fire while `rx.recv()` blocks).

## High

### [x] WG-H1. `BindingType::StorageBuffer` mapped as read-only — diverges from Vulkan
- **Where:** `conversion.rs:501-505` (vs Vulkan `pipeline.rs:261`, read-write capable)
- **Category:** bug / divergence (see XB-H2)
- **Failure:** any `var<storage, read_write>` shader fails pipeline creation on wgpu with a
  binding-type mismatch while the identical material works on Vulkan — breaks all writable-
  storage compute on wgpu.

### [ ] WG-H2. Compressed texture formats converted but their device features never requested
- **Where:** `conversion.rs:83-222` (BC/ETC2/ASTC in the format table) + `mod.rs:118` (only
  `POLYGON_MODE_LINE` requested)
- **Category:** bug
- **Failure:** `create_texture` with e.g. `Bc7RgbaUnorm` raises "format requires feature",
  which without an uncaptured-error handler panics the process — even on desktop adapters
  that support BC.

### [ ] WG-H3. Surface acquire errors collapsed into a generic failure; no reconfigure-and-retry
- **Where:** `wgpu_impl/swapchain.rs:36-38`
- **Category:** bug
- `SurfaceError::Lost/Outdated/Timeout/OutOfMemory` all become
  `GraphicsError::ResourceCreationFailed`; `Timeout` (recoverable, skip-frame) is not
  distinguished; nothing reconfigures.
- **Failure:** window resize / alt-tab / monitor sleep → frame errors out and rendering
  stops or error-spams instead of transparently recreating the surface.

### [x] WG-H4. Mid-graph `WriteBuffer` transfer op executes before *all* passes in the submission
- **Where:** `pass_encoding.rs:400-419` (`queue.write_buffer` during encoding)
- **Category:** bug / divergence (see XB-H3)
- wgpu schedules queue writes at the start of the next submission, not at the transfer
  pass's position in the graph.
- **Failure:** graph [pass A reads uniform] → [transfer pass writes it]: pass A incorrectly
  sees the *new* data on wgpu; Vulkan encodes the write at the correct point — backends
  diverge.

### [ ] WG-H5. Per-draw bind group creation under a global mutex
- **Where:** `pass_encoding.rs:156`, `:182-183`, `:283-289` (graphics); `:615`, `:637-638`,
  `:738-745` (compute)
- **Category:** performance
- Every draw/dispatch re-creates all bind groups via `device.create_bind_group` (plus clones
  every `BindGroupLayout` into scratch), all while holding the backend-wide `encoder_scratch`
  mutex.
- **Impact:** thousands of bind-group allocations per frame — the dominant-CPU-cost pattern
  wgpu docs warn about, far worse on WebGL. A `(MaterialInstance, resource-version) →
  BindGroup` cache eliminates nearly all of it; the mutex also blocks future multi-threaded
  encoding.

### [x] WG-H6. `wait_fence_timeout` reports failure when the wait succeeded
- **Where:** `resources.rs:499-504`
- **Category:** bug
- After `PollType::Wait { submission_index }` returns `Ok(status)`, the code returns
  `status.is_queue_empty()` — a successful wait for a specific submission returns
  `WaitSucceeded`, not `QueueEmpty`, when other submissions are in flight.
- **Failure:** exactly the multi-frame-in-flight case fences exist for: the target submission
  completes in time but the function returns `false` → spurious timeout handling every frame.
- **Fix:** `status.wait_finished()`.

## Medium

### [x] WG-M1. `Bgra10a2Unorm` mapped with swapped channel order vs Vulkan
- **Where:** `conversion.rs:66` (`→ Rgb10a2Unorm`, RGBA order) vs Vulkan
  `A2R10G10B10_UNORM_PACK32` (BGRA order)
- **Category:** divergence (see XB-M1)
- **Failure:** HDR10-style target renders with R/B swapped on one backend;
  `from_wgpu_texture_format` can never round-trip it.

### [ ] WG-M2. Silent `_ => Rgba8Unorm` fallback in `convert_texture_format`
- **Where:** `conversion.rs:223`
- **Category:** dangerous hardcode (same as VK-L4)
- A new engine `TextureFormat` variant compiles without warning and produces an 8-bit RGBA
  texture; failures surface far from the cause.

### [ ] WG-M3. `configure_surface`: no capability validation; usage/latency/alpha hardcoded
- **Where:** `wgpu_impl/swapchain.rs:13-30`
- **Category:** dangerous hardcode
- Format and present mode unchecked against `surface.get_capabilities()`; width/height not
  guarded against 0; `usage: RENDER_ATTACHMENT` (no `COPY_SRC`), `alpha_mode: Auto`,
  `desired_maximum_frame_latency: 2` fixed.
- **Failure:** `PresentMode::Mailbox` on Metal/GL/web (Fifo-only) or configuring a minimized
  window → validation error → panic. No `COPY_SRC` blocks final-frame screenshots.

### [ ] WG-M4. `write_texture`: `bytes_per_row` wrong for compressed formats; only mip 0 written
- **Where:** `resources.rs:553-585`
- **Category:** bug
- `bytes_per_row = width * block_copy_size` multiplies bytes-per-*block* by width in
  *texels* (4× too large for BC/ETC/ASTC); `rows_per_image = height` should be block rows;
  `mip_level` hardcoded 0 while `mip_level_count` may be > 1.
- **Failure:** BC7 upload rejected ("data size too small for layout"); mips 1..N garbage.
  The same texel-vs-block confusion exists in `pass_encoding.rs:444`, `:508` and is mirrored
  in the Vulkan transfer ops (VK-M14).

### [ ] WG-M5. `read_buffer` staging path ignores mapping errors and can panic
- **Where:** `resources.rs:591-653`
- **Category:** bug
- Fallback path does `let _ = rx.recv();` then unconditionally `get_mapped_range()`. The
  first-attempt direct map also uses a validation error as control flow (device error spam
  on every read of a non-`MAP_READ` buffer). Also allocates a fresh staging buffer +
  submission per call (perf).
- **Failure:** reading back a buffer lacking `COPY_SRC` panics inside `get_mapped_range()`
  instead of returning an error.

### [ ] WG-M6. `ensure_compatible_with_surface` swaps the device, orphaning every existing resource
- **Where:** `wgpu_impl/mod.rs:166-221`
- **Category:** bug
- On adapter mismatch `self.device`/`self.queue` are replaced, but all previously created
  buffers/textures/pipelines/fences belong to the old device; nothing enforces recreation,
  and `GpuFence::Wgpu` objects keep polling the dead device.
- **Failure:** multi-GPU laptop: first draw using any pre-existing resource triggers a
  cross-device validation panic.

### [ ] WG-M7. Layout/bind-group contract mismatch for `CombinedTextureSampler`
- **Where:** `conversion.rs:524-531` (layout: single texture entry at N) vs
  `pass_encoding.rs:201-220`, `:656-675` (bind group: texture at N + sampler at N+1)
- **Category:** bug
- Only correct when Slang reflection pre-split the combined sampler; a hand-authored layout
  using `CombinedTextureSampler` fails `create_bind_group` ("binding N+1 not in layout").
  One side must own the split.

### [ ] WG-M8. Depth attachment with non-wgpu handle panics instead of erroring
- **Where:** `pass_encoding.rs:100-102` (`panic!`), vs the graceful
  `GraphicsError::InvalidParameter` pattern at `:227-233`
- **Category:** bug
- A `GpuTexture::Dummy`/Vulkan depth texture aborts the process; every equivalent color/
  buffer mismatch in the same file returns an error. Also can't handle
  `RenderTarget::Surface` depth, which Vulkan handles.

### [ ] WG-M9. `execute_graph` sync path ignores the poll result, including timeout
- **Where:** `wgpu_impl/mod.rs:300-303` (`let _ = device.poll(Wait { 10s })`)
- **Category:** bug (wgpu counterpart of VK-C2)
- **Failure:** a > 10 s workload (first-use shader compile on WebGL) times out silently;
  `FrameSchedule::finish` recycles per-frame resources still in use.

### [ ] WG-M10. Texture binding layouts hardcode filterable-float / filtering-sampler
- **Where:** `conversion.rs:506-523`
- **Category:** dangerous hardcode
- All texture bindings become `Float { filterable: true }`, all samplers `Filtering`; no
  path for `TextureSampleType::Depth`, comparison samplers, non-filterable float
  (`R32Float` without `FLOAT32_FILTERABLE`), integer, or multisampled textures.
- **Failure:** shadow-mapping material sampling `Depth32Float` with a comparison sampler
  (the sampler descriptor supports `compare`, `resources.rs:100`) fails layout validation
  on wgpu; works on Vulkan — divergence.

### [ ] WG-M11. `D1Array` textures get a `D1` view; array-layer count from `size.depth`
- **Where:** `resources.rs:43`, `:67-69`
- **Category:** bug
- **Failure:** a `D1Array` texture with depth > 1 bound as an array fails bind-group
  validation (auto view is `D1`); arrayed-1D textures don't exist on WebGPU at all.

## Low

### [ ] WG-L1. `LoadOp::DontCare` mapped to `Load` instead of a discard-equivalent
- **Where:** `conversion.rs:294`, `:314`, `:331`
- **Category:** perf / divergence (see XB-H4)
- Every "don't care" attachment forces a full tile load on tiler GPUs (all mobile/web) —
  bandwidth cost on exactly the platforms this backend targets; semantics also differ from
  Vulkan's true `DONT_CARE`.

### [ ] WG-L2. More than 8 color attachments silently truncated in release
- **Where:** `pass_encoding.rs:49-55` (`.min(8)` + `debug_assert!`)
- 9-MRT pass renders 8 and drops the ninth without a log in release; panics in debug.

### [ ] WG-L3. Scissor rects cast and forwarded unclamped
- **Where:** `pass_encoding.rs:149`, `:315-320`, `:349`
- `scissor.x as u32` turns a negative origin into ~4 billion; no clamping against the target
  extent (wgpu requires scissor ⊆ target). Clip rect 1px past the window edge → validation
  panic. (Vulkan counterpart: VK-M2.)

### [ ] WG-L4. `encoder_scratch.lock().unwrap()` poison panic
- **Where:** `pass_encoding.rs:156`, `:615`
- One panicking encode (e.g. WG-M8) poisons the mutex; every later frame then panics even
  though the original cause was transient.

### [ ] WG-L5. Copy alignment unvalidated; multi-layer 1-row copies rejected
- **Where:** `pass_encoding.rs:390-397`, `:418`; `resources.rs:525`
- `write_buffer`/`copy_buffer_to_buffer` need 4-byte (`COPY_BUFFER_ALIGNMENT`) offsets/sizes —
  nothing checks or pads; the 256-byte row alignment is a literal `& !255` instead of
  `COPY_BYTES_PER_ROW_ALIGNMENT`; `bytes_per_row` stays `None` when `height == 1 && depth > 1`,
  which wgpu rejects.

### [ ] WG-L6. Pipeline state hardcodes: no MSAA, fixed depth state, fixed front-face/cull
- **Where:** `resources.rs:269-288`
- `multisample: default()` (count 1), `depth_write_enabled: true` + `LessEqual`,
  `front_face: Ccw`, `cull_mode: None`, `write_mask: ALL`. Consistent with Vulkan's hardcodes
  today (no divergence), but none are expressible via `MaterialDescriptor`, and any
  `sample_count > 1` texture fails pipeline-vs-attachment validation.

### [ ] WG-L7. `WgpuBackendType::WebGpu` maps to `Backends::GL`
- **Where:** `graphics/src/instance.rs:111-112` (`BROWSER_WEBGPU` never used anywhere)
- Explicitly requesting WebGPU selects WebGL instead — compounds WG-C2's limits failure and
  silently loses compute-shader support (WebGL has none).
