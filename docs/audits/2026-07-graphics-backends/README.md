# Graphics Backend Audit — July 2026

Audit of the Vulkan (`graphics/src/backend/vulkan/`) and wgpu (`graphics/src/backend/wgpu_impl/`)
backends plus the backend abstraction layer (`graphics/src/backend/mod.rs`). Scope: correctness
bugs, performance tuning, dangerous hardcode. ~10k lines reviewed in full by four independent
audit passes (Vulkan core, Vulkan sync/pipeline, wgpu, cross-backend consistency); findings
below are deduplicated. Several critical findings were independently confirmed by 2–3 passes.

## Findings by file

| File | Contents |
|------|----------|
| [VULKAN.md](VULKAN.md) | Vulkan backend: sync, swapchain, descriptors, pipelines, uploads |
| [WGPU.md](WGPU.md) | wgpu backend: web/wasm viability, conversions, encoding, resources |
| [CROSS_BACKEND.md](CROSS_BACKEND.md) | Semantic divergences between backends + abstraction-layer issues |

Finding IDs are stable (`VK-*`, `WG-*`, `XB-*`) and each has an unchecked box for tracking fixes.

## Severity summary

| Severity | Vulkan | wgpu | Cross-backend |
|----------|--------|------|---------------|
| Critical | 3 | 3 | 1 |
| High | 9 | 6 | 5 |
| Medium | 17 | 11 | 7 |
| Performance | 9 | (in WG-H5, WG-L1) | — |
| Low | 14 | 7 | 6 |

## Top issues (fix first)

1. **VK-C1** — `is_fence_signaled` returns `true` for unsignaled fences (`.is_ok()` on a
   status that is `Ok(false)` for `VK_NOT_READY`). One-line fix, prevents use-after-free of
   in-flight resources.
2. **VK-C3** — buffer read barriers hardcode `TransferWrite` as the source scope; compute
   shader writes are never synchronized against subsequent reads. Needs real per-buffer
   last-writer tracking (analogous to the texture layout tracker).
3. **WG-C1..C3** — the wgpu backend cannot run on wasm at all: blocking `pollster::block_on`
   / `device.poll(Wait)` / `mpsc::recv()`, plus a native-only `POLYGON_MODE_LINE` feature and
   default limits requested unconditionally.
4. **XB-C1** — vertex attribute `shader_location` conventions differ between backends
   (sequential on wgpu vs semantic-index on Vulkan): the same mesh layout misbinds on exactly
   one backend.
5. **VK-C2 / WG-M9** — fence-wait timeouts are swallowed on both backends; frame resources
   are then recycled while the GPU is still executing.

## Recommended fix order

1. **One-liners with outsized impact:** VK-C1 (`matches!(status, Ok(true))`), WG-H6
   (`wait_finished()` instead of `is_queue_empty()`), VK-H6 (add `UNIFORM_BUFFER_DYNAMIC` to
   the descriptor pool), VK-H8 (set `stencil_attachment_format`), VK-H9 (derive `input_rate`
   from `step_mode`), WG-H1 (`read_only: false` for storage buffers), VK-L4/WG-M2 (turn the
   `_ => Rgba8Unorm` format catch-alls into errors).
2. **Buffer barrier system (VK-C3):** track the last writer per buffer instead of the
   hardcoded `TransferWrite` source; also removes the per-read redundant barriers (VK-P7).
3. **Swapchain lifecycle (VK-H5, VK-H7, VK-M3, VK-M4):** don't reset the in-flight fence
   before acquire; propagate `OUT_OF_DATE`/`SUBOPTIMAL` to the caller; chain the acquire
   semaphore into the layout transition (`srcStage = COLOR_ATTACHMENT_OUTPUT`); make the
   `UNDEFINED` transition conditional on load op / first pass.
4. **`write_texture` on both backends (VK-H4, WG-M4):** all mips/layers, block-compressed
   row pitch in blocks, correct aspect, and register the resulting layout in the tracker.
5. **Vertex location convention (XB-C1):** pick one (semantic indices are the better fit
   since shaders compile once) and align the wgpu backend.
6. **Performance package:** device-local buffers with a staging path (VK-P1), descriptor
   set / bind group caching keyed by material instance + resource version (VK-P2, WG-H5),
   `VkPipelineCache` (VK-P5).
7. **wgpu web path (WG-C1..C3, WG-L7):** async init under `cfg(target_arch = "wasm32")`,
   intersect features/limits with the adapter, non-blocking readbacks, use `BROWSER_WEBGPU`.
