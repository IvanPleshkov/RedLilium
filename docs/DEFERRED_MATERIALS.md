# Application materials in the stock deferred temporal pipeline

`GBufferContract::V1` lets a game use its own surface shader with the stock
G-buffer, resolve, TAA and motion blur. The game owns texture evaluation, cutout,
deformation and previous vertex data. The engine owns camera/model transforms,
render targets, ordering, reprojection and filtering. This does not add TAA to
the forward path or make forward materials deferred-compatible.

## Registration

Before material loading/rendering:

1. Register the game's `ShadingModel` with `ShadingRegistry::register`. Its
   ordered property schema must match the shader's static material block, as
   described in [MATERIAL_ASSETS.md](MATERIAL_ASSETS.md). Duplicate ids and
   duplicate/empty property names are errors. Publish/load the Slang source as
   an ordinary shader asset with the model's shader GUID.
2. Call `world.resource_mut::<PipelineCache>().register_deferred_shader(guid,
   GBufferContract::V1)`. This declares the shader's semantic contract. Built-in
   `pbr` and `pbr_textured` are registered automatically.
3. Put `RenderPath::named(DEFERRED_PIPELINE)` on the camera. Add `TemporalJitter`
   for TAA and/or `MotionBlur { shutter, samples }` for motion blur. Velocity is
   produced even without those effects.

Registration alone does not authorize arbitrary shader outputs. Each encountered
source version, variant and vertex-layout specialization is checked against the
executable WGSL interface: exact fragment output locations/types, no custom
fragment depth, camera/model uniform sizes and matrix offsets, and the reflected
binding-set classes/layouts. Pipeline creation also validates vertex inputs and
resource bindings. A rejected specialization logs an error and does not draw;
on hot reload the cache retains a previously validated pipeline, if any. Define
variants are checked separately; an unvalidated forward specialization cannot
satisfy a deferred cache lookup.

This validation proves the binary interface, not the meaning of shader arithmetic.
The registered shader promises the value conventions below (including finite
values, velocity sign and alpha rejection). The engine cannot infer whether an
application supplied the right previous vertices.

## V1 attachments and depth

| Location | Fragment type | Target format | Meaning |
| --- | --- | --- | --- |
| 0 | `float4` | `Rgba8UnormSrgb` | RGB: linear surface color, encoded by the target. A: geometry/AO marker; zero means background, geometry writes `max(AO, 1/64)` (usually 1 for unlit). |
| 1 | `float4` | `Rgba16Float` | RGB: normalized world normal encoded as `normal * 0.5 + 0.5`. A: metallic in [0,1] for PBR; **-1 for unlit**. |
| 2 | `float` | `R8Unorm` | Roughness in [0,1] (use 1 for unlit). |
| 3 | `float2` | `Rg16Float` | Current minus previous **unjittered NDC** position, per rendered frame. |

Depth is the stock reversed-Z raster depth: clear 0, near 1, greater comparison,
writes enabled. Use the supplied jittered camera matrix for `SV_Position`; do
not write `SV_Depth`. The stock specialization uses triangle topology, back-face
culling and no blending. This contract supports opaque and alpha-tested surfaces,
not partial transparency. Discard transparent texels before writing any outputs;
RT0 alpha is a geometry marker, **not alpha coverage**. Normals must remain valid
also for unlit geometry because optional SSAO/depth consumers can inspect it.

For negative RT1 alpha the deferred resolve writes RT0 RGB directly to
`scene_color`, bypassing lighting, IBL, shadowing and AO. This is an LDR unlit
route, not additive HDR emission: RT0 cannot carry values outside [0,1]. TAA,
motion blur, bloom, exposure and display tone mapping/encoding still operate
downstream. Unlit therefore guarantees authored linear color at scene resolve,
not byte-identical final display pixels. Color textures normally use sRGB
sampling so their decoded linear RGB is written to RT0 without a manual encode.

## Vertex data and bindings

Use Slang `vs_main` / `fs_main` entry points. Declare these parameter blocks in
order, using `import engine` for the attributes:

| Set/binding | Class | Layout |
| --- | --- | --- |
| 0 / 0 | `[UpdateRate("external")]` | Three `column_major float4x4`: jittered raster VP, current unjittered VP, previous unjittered VP; offsets 0/64/128, size 192. |
| 1 / 0 | `[UpdateRate("dynamic")]` | Two `column_major float4x4`: current model, previous model; offsets 0/64, size 128. |
| 2 | `[UpdateRate("static")]` | Optional application material block; normal schema packing, texture/sampler pairs and storage-buffer bindings. |

Sets 0 and 1 each contain only their uniform binding. Their dynamic offsets come
from the engine's fence-protected `FrameRing`. The game must not supply its own
camera/model groups. Additional arbitrary sets are outside V1.

Vertex attributes are application-defined. Shader locations follow the
`VertexLayout.attributes` declaration order, **not** `VertexAttributeSemantic`
indices. Several vertex buffers are supported. A previous position can be an
extra float3 attribute (the example uses the `Normal` semantic at location 1 for previous positions),
or can come from a material `StorageBufferSource::Ref` published with
`MaterialInstanceManager::publish_buffer`. The engine does not generate or retain
that data. Buffer uploads/copies must go through `RenderGraph` transfer passes;
keep a safe frame-ring/suballocation strategy when streaming in-flight data.

For current local position `p` and previous local position `p_prev`:

```text
raster_clip = jittered_VP * model * float4(p, 1)
current_clip = current_unjittered_VP * model * float4(p, 1)
previous_clip = previous_unjittered_VP * previous_model * float4(p_prev, 1)
velocity = current_clip.xy / current_clip.w
         - previous_clip.xy / previous_clip.w
```

Do not add jitter or multiply velocity by a time delta. NDC is +X right, +Y up;
the temporal passes convert this to texture-UV displacement `(vx/2, -vy/2)` and
sample previous history at `current_uv - displacement` (with their jitter
compensation). For perspective discontinuities/invalid previous projections,
reset history rather than writing non-finite velocity.

Store the previous **rendered** deformation, not the previous physics tick.
Several fixed ticks, interpolation, zero fixed ticks during pause, or slow
motion can all occur between renders. Current geometry must match the geometry
actually submitted that frame; update its history once for the rendered frame,
shared by that frame's camera views. New/reappearing objects without meaningful
vertex correspondence should initialize previous vertices from current ones.
Changing topology requires rebuilding that correspondence or resetting it.

The stock shadow depth override does not evaluate a custom cutout/deformation
shader. Set `MeshRenderer::with_casts_shadows(false)` for these surfaces unless
the ordinary depth shader correctly represents their silhouette. V1 does not
provide custom shadow-material registration.

## History reset and ownership

Call `world.resource_mut::<TemporalState>().reset_history()` **before the Render
schedule** after a camera cut, restart, teleport, skin/topology change or another
application-detected discontinuity. The resource exists after renderer setup;
standalone hosts can insert `TemporalState::default()` earlier if needed. A reset:

- Clears previous and pending camera/model transforms for all cameras/entities.
- Invalidates each camera's TAA on its next recorded frame and takes current
  scene color wholesale, even if that camera did not render during the reset.
- Disables motion blur for that frame. Auto-exposure adaptation is retained.
- Leaves the render frame index/jitter sequence monotonic.

The game must also set its previous deformation data to current data. Engine
reset does not rewrite vertex/storage buffers. Calls before Render take effect
in that render; calls after Render affect the next one. Resizing/recreating a
camera's targets and re-enabling temporal effects invalidate their target history
automatically. Custom pipelines can consume `TemporalState::history_generation()`
as an invalidation token. Reset is deliberately global; per-object history masks
and partial-camera invalidation are not part of this API.

Real-time UI remains in the existing overlay path after scene effects. Neither
registration nor reset changes fixed simulation or the scaled/unscaled clocks.

## Shipping game shaders on WASM and native without Slang

The browser cannot run the native Slang compiler. The game must bake every
selected define permutation and register its artifacts before material loading:

```sh
cargo run -p xtask --features slang -- bake-material \
  /path/to/game/cutout.slang /path/to/game/src/cutout_baked.rs
# For a variant, add NAME=VALUE arguments and use a separate output module.
```

The tool needs the Slang SDK (`scripts/fetch-slang.sh`), uses the standard engine
shader library, and emits a Rust module containing source, per-stage WGSL and
binding reflection. Include it as a module and call `cutout_baked::register()?`
at startup on native and WASM. The loaded shader asset must contain the same
source; a source or define change requires rebaking. Runtime Slang builds compile
the original source; default native and WASM builds use the registered artifacts.

The underlying public API is
`redlilium_graphics::shader::baked::register_shader_set`: effective Slang stage
sources/defines, matching WGSL strings, and the pre-reclassification layouts/rates
from `SlangCompiler::reflect_all_bindings`. This supports game-owned bake tooling
without changing engine tables. WGSL is parsed/validated at registration; the
baker owns semantic equivalence to Slang. Entries are immutable for the process
lifetime, identical registration is idempotent, and conflicting code/reflection
is rejected atomically. No GPU objects are allocated by registration. Imported
application modules are not supported by this small `bake-material` command;
custom tooling can provide its own compiler modules and register the result.

## Executable reference

- `ecs/tests/fixtures/deforming_cutout.slang`: application shader with a textured
  alpha test, previous positions, fixed model matrix and unlit output.
- `ecs/tests/deferred_custom_material.rs`: public material schema/asset setup,
  ordinary Render schedule, orthographic camera, GPU readback of cutout,
  deformation velocity, unlit resolve, TAA, motion blur, reset and resize.

```sh
cargo test -p redlilium-ecs --features rendering --test deferred_custom_material
```

The same assertions run in Chrome/WebGPU (use a `wasm-bindgen` CLI version
matching Cargo.lock):

```sh
cargo build -p redlilium-ecs --features rendering \
  --example deferred_material_probe --target wasm32-unknown-unknown
wasm-bindgen target/wasm32-unknown-unknown/debug/examples/deferred_material_probe.wasm \
  --target web --out-dir target/deferred-material-web/pkg
cp ecs/examples/deferred_material_probe.html target/deferred-material-web/index.html
python3 -m http.server 8087 --bind 127.0.0.1 --directory target/deferred-material-web
```

Open `http://127.0.0.1:8087` in a browser with WebGPU. The probe reports PASS after
GPU readbacks, yielding the event loop while fences/maps are pending. It covers
pause/slow-motion and different render intervals as well as the cases above.
The fullscreen resolve/TAA/blur shaders use explicit LOD for single-level
render targets, so per-pixel rejection does not rely on implicit derivatives in
non-uniform control flow. No WGSL validation diagnostic is disabled.
