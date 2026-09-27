# Texture views and subresource tracking

Status: implemented after API review. The API and restrictions below describe
the supported scope.

The goal is to sample one mip or layer range while writing a disjoint range of
that same texture through the render graph. Device methods only create resources;
transfers, mip generation and IBL filtering remain graph operations.

## Resource API

An immutable `TextureView` owns an `Arc<Texture>` and a validated, normalized
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

Use `BindingGroupDescriptor::with_texture_view` and
`RenderTarget::from_view`. Attachments select one 2D mip/layer and all format aspects;
layered rendering and D3 slice attachments remain separate features. Existing
`with_texture` and `RenderTarget::from_texture_*` keep working, resolving to the
same validated view representation internally. Native views are cached by normalized
descriptor; public views are cached weakly to avoid a Texture -> TextureView -> Texture cycle.

Sampled bindings use `BindingType::SampledTexture { dimension,
sample_type, multisampled }`. Sample type is Float (filterable or unfilterable),
Sint, Uint, or Depth. Existing Texture/TextureCube/Texture2DArray/DepthTexture
variants remain conveniences normalized to this description for layout comparison
and backend conversion. Storage texture bindings are a separate API decision;
views alone do not introduce them.

## Graph and synchronization

One normalized `TextureSubresourceRange` (aspect, mip range, layer range)
is used throughout usage inference, dependency analysis and backend barriers. Ranges
come from bound views, render targets and transfer regions. Whole-texture bindings
include **all** mips/layers. Parent textures survive through the fence retirement path.

Two accesses conflict only when texture identity and ranges overlap and either
access writes. Read-only depth attachment/sampling remains supported.
A same-pass conflicting overlap is an error; disjoint ranges do not create a
false dependency. Strict mode still rejects ambiguous overlapping writers.

Vulkan starts with a compact whole-image state and splits it only on partial
access. It tracks layout/access and submission history per overlapping range,
merging adjacent equal states and barriers across submissions and queues.
Without separate depth/stencil layout support, both aspects transition
conservatively. D3 state is per mip, never per Z slice. Private mip-generation
barriers restore the declared final layout, including scratch-copy paths.

wgpu uses the same graph dependencies and native views; its internal barriers
remain wgpu's responsibility. Backend limitations are checked before recording.
Whole-image tracking remains a fast path for existing users.

## Validation coverage

Unit tests cover descriptor validation, native view reuse, range subtraction,
range-aware dependencies and cross-queue layout history with rollback. GPU tests
cover sampling mip N-1 while rendering mip N, disjoint array layers, same-texture
copies and overlapping-access rejection, with pixel verification through graph
readback. Existing depth, mip-generation and IBL tests cover compatibility with
whole-texture paths. Native workspace and wasm rendering builds check the API
migration in consumers.

Probe capture, probe selection/blending, roughness conventions and lighting integration stay with the
renderer/ECS, outside the graphics library.
