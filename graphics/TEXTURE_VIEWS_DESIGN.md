# Texture views and subresource tracking — proposal

Status: awaiting API review. This document does not describe implemented API.

The goal is to sample one mip or layer range while writing a disjoint range of
that same texture through the render graph. Device methods only create resources;
transfers, mip generation and IBL filtering remain graph operations.

## Resource API

Add an immutable `TextureView` owning an `Arc<Texture>` and a validated, normalized
view descriptor. Native image views remain internal. Create it with:

```rust,ignore
let view = device.create_texture_view(
    &texture,
    &TextureViewDescriptor::new(TextureViewDimension::D2)
        .with_mip_levels(2, 1)
        .with_array_layers(0, 1),
)?;
```

`TextureViewDescriptor` contains dimension, aspect (`All`, `DepthOnly`,
`StencilOnly`), base mip/layer and optional counts (`None` means remaining).
Creation resolves counts, rejects empty/out-of-bounds ranges, checks device
ownership and dimension compatibility. `TextureView` exposes the parent texture,
resolved range, dimension, format and size. Initial views retain the texture's
format; format reinterpretation is a separate extension.

Cube views require six faces and a six-aligned base layer; cube arrays require
multiples of six. A cube or array may expose one face/layer as a D2 view. D3 views
remain D3: Z slices are not array layers or independently tracked image layouts.
MSAA views have one mip and retain the texture's sample count.

Add `BindingGroupDescriptor::with_texture_view` and
`RenderTarget::from_view`. Attachment views must select one mip and one layer;
layered rendering and D3 slice attachments remain separate features. Existing
`with_texture` and `RenderTarget::from_texture_*` keep working, resolving to the
same validated view representation internally. Cache native views by normalized
descriptor; avoid an owning Texture -> TextureView -> Texture cycle.

Generalize sampled bindings with `BindingType::SampledTexture { dimension,
sample_type, multisampled }`. Sample type is Float (filterable or unfilterable),
Sint, Uint, or Depth. Existing Texture/TextureCube/Texture2DArray/DepthTexture
variants remain conveniences normalized to this description for layout comparison
and backend conversion. Storage texture bindings are a separate API decision;
views alone do not introduce them.

## Graph and synchronization

Use one normalized `TextureSubresourceRange` (aspect, mip range, layer range)
throughout usage inference, dependency analysis and backend barriers. Infer ranges
from bound views, render targets and transfer regions. A whole-texture binding
means **all** mips/layers, not the current declaration default of one mip/layer.
Keep parent textures alive through the existing fence retirement path.

Two accesses conflict only when texture identity and ranges overlap and either
access writes. Preserve the current read-only depth attachment/sampling exception.
A same-pass conflicting overlap is an error; disjoint ranges do not create a
false dependency. Strict mode still rejects ambiguous overlapping writers.

Vulkan starts with a compact whole-image state and splits it only on partial
access. Track layout/access/stage and submission history per overlapping range;
merge adjacent equal states and barriers. Preserve cross-submit and cross-queue
synchronization. Without separate depth/stencil layout support, transition both
aspects conservatively. D3 state is per mip, never per Z slice. Existing private
mip-generation barriers must update this state, including scratch-copy paths.

wgpu uses the same graph dependencies and native views; its internal barriers
remain wgpu's responsibility. Backend limitations are checked before recording.
Whole-image tracking remains a fast path for existing users.

## Implementation and acceptance

1. Add views, canonical sampled binding descriptions, creation validation and
   native view caching. Preserve the existing convenience API.
2. Carry exact ranges through every inferred use and graph dependency; convert
   Vulkan state tracking, including across submissions and async queues.
3. Enable disjoint same-texture copies and read/write passes only after tracking
   is complete. Keep overlapping copies rejected. Adapt private mip/IBL paths.
4. Test mip N-1 sampling while rendering mip N, array/cube face isolation,
   overlapping-range rejection, cross-submit state, read-only depth and unchanged
   whole-texture paths on Dummy, Vulkan with validation, and wgpu. Verify generated
   pixels through graph readback and retain native + wasm build coverage.

The decision for review is the API above and this scope. Probe capture, probe
selection/blending, roughness conventions and lighting integration stay with the
renderer/ECS, outside the graphics library.
