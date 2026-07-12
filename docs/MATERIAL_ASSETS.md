# Material Assets & Shading Architecture (design)

Status: **Proposed** (architecture agreed; concrete forks open — see end).
Date: 2026-06-29.

This is the design for materials as assets. It turned out to be a *shading
architecture* decision, not just an asset type — so it's captured here before
implementation. The asset side (mesh, vertex layout, shader) it builds on is in
[`ASSETS.md`](ASSETS.md).

## The problem

A material must be **settings-agnostic**: one chair prefab has to render at *any*
graphics settings. But the *compiled* pipeline depends on many runtime axes:

- the **mesh's vertex layout** (the shader reads a compatible subset),
- the **pass** (forward / deferred / depth / shadow …),
- **quality preset + feature toggles** (PBR vs cheap, normal-mapping on/off …),
- **view settings** (MSAA, …),
- the **render target's formats** (LDR vs HDR),
- **hardware capabilities**.

If a material were tied to a concrete shader/pipeline, the chair prefab would
need a copy per setting. It must not. So the authored surface and the compiled
pipeline are different layers, bridged by a permutation/specialization system.

## Decision 1 — the asset / code boundary

- **Assets = content** the game/user authors: `Material` (surface), `Shader`
  (source), `Texture`, `Mesh`, `VertexLayout`.
- **Render pipeline = code/config** the engine owns: passes (forward/deferred),
  the `shading-model × quality → shader + defines` mapping, **binding contracts**
  (which group is camera / per-entity / lighting / material), quality presets,
  hardware-capability handling.

The render pipeline *references shader assets by guid* but *declares bindings in
code*. Bindings are a small fixed engine-defined set (camera, per-entity,
lighting, material) — an explicit code contract, **not** a per-binding asset
(that would be over-engineering). Reasoning: this is a self-engine; the owner of
the graphics stack wants the render architecture explicit in code and the content
data-driven. No magic — the render system puts the camera in group 0 *by
contract*, not by discovery.

## Decision 2 — a `Material` is a SURFACE, not a shader recipe

A `Material` asset declares:

- a **shading model** (e.g. `opaque_pbr`, `unlit`, `transparent`),
- **property values** for that model's schema (albedo, metallic, roughness,
  emissive, texture refs …) — the *full* set; a cheap variant shader just reads a
  subset (exactly like a mesh provides a rich vertex layout and a shader reads a
  subset).

A material has **no shader reference, no vertex-layout reference, no formats** —
those are settings/mesh/target dependent and resolved by the render system. This
is the Unreal/Godot abstraction (material = surface attributes; the renderer owns
shader selection), *not* the Unity/Bevy one (material references a shader).

## Decision 3 — the shading model is the linchpin (code/registry)

Each shading model is engine-defined (code, like the bindings) and declares two
things:

1. the **material property schema** = the layout of the material binding group
   (what the material fills),
2. which **shader(s)** implement it (e.g. forward vs deferred) and the
   **render-side define selection** (quality/pass/hardware → defines).

(The full define story — who declares vs who selects — is its own section below;
the shading model owns only the *render-side* selection, not the define space.)

So the shading model is the *contract* between a material (fills the schema) and
all of that model's shaders (any preset reads the same material binding group).
Materials reference a shading model by name/id.

Open: is the shading-model set fixed, or game-extensible (a registry the game adds
to in code)? Leaning extensible-by-code (flexible, still no magic).

## Decision 4 — the pipeline is derived & cached (Bevy-style)

The compiled GPU pipeline (graphics `Material`) is **not** an asset and **not**
1:1 with a material asset. It is derived **on demand** and cached, keyed by:

```
(shader + defines + vertex_layout + render_state + formats)
```

resolved at render time from `(material.shading_model + current preset + material
features + hardware + mesh.layout + target.formats)`. This is Bevy's
`SpecializedMeshPipeline` pattern — compile only what is actually encountered, no
offline permutation explosion.

## Decision 5 — shader variants & defines: the proven path

"Who owns the shader defines?" was the hard one. The answer from every engine:
**no single owner — the shader declares the space, selection is split, the cache
keys by the variant.** We follow this proven path rather than inventing an owner.

How the engines do it:

- **Unity** — keywords declared in the shader via `#pragma`: `multi_compile`
  (global, set by the render pipeline / global state — quality, lighting) vs
  `shader_feature` (set by the **material**); built with variant stripping.
- **Unreal** — permutation dimensions in the shader's C++ (`FShaderPermutationDomain`)
  + **material static switches**; the renderer + `ShouldCompilePermutation` decide
  what to cook.
- **Godot** — engine shaders declare variant defines (`#ifdef` + a C++ list);
  `StandardMaterial3D` **feature flags** feed in; the renderer selects.
- **Bevy** — `#ifdef` shader-defs + `specialize(key) -> shader_defs` (code); the
  key is `(material flags × mesh layout × view)`; compiled on demand in the
  `PipelineCache`.

The universal pattern (what we adopt):

1. **The shader (`.slang`) declares its define space** (`#ifdef`). The variant
   space travels with the source; the `ShaderManager` loads it but does *not*
   select.
2. **Selection is split:**
   - the **material** contributes its **feature defines** (e.g. has-normal-map),
   - the **render pipeline** contributes **quality / pass / mesh-layout / view /
     hardware** defines (the shading model's render-side selection from Decision 3).
3. **Compile + cache by the full variant** — the pipeline cache keyed by
   `(shader + all defines + layout + state + formats)`, on demand (Bevy-style).

So: the `ShaderManager` owns the **source** (and, later, optionally a compiled-
*module* cache keyed *by* defines — a perf refactor that separates module compile
from pipeline build). It does **not** own define *selection*. Defines are declared
by the shader and selected jointly by the material (features) and the render
pipeline (everything else).

**Implemented (#6)** in `graphics/src/shader/variants.rs`: the shader declares
its space with `//#pragma variant NAME [values…] [default V]` (material-selected,
has a default) and `//#pragma variant_system NAME [values…]` (pipeline-selected,
deliberately default-free — every call site sets it explicitly).
`ShaderVariantSpace::parse` → `.select().feature(…).system(…).build()` →
`VariantKey` → `MaterialDescriptor::with_variant`; typos and missing system axes
fail at key-build time. The offline bake enumerates each shader's full cartesian
product from the same pragmas (capped at `MAX_VARIANTS_PER_SHADER = 64`), so a
variant can no longer be forgotten in the registry. Bool axes emit value-less
`#ifdef` defines when on and nothing when off; enum axes always emit `NAME=value`.

## Decision 6 — keep the Material / MaterialInstance split

Mirrors the graphics types (`Material` pipeline + `MaterialInstance` bindings) and
the Unreal Material / Material-Instance model:

- **`Material` asset** = shading model + **default** property values + feature
  config (the template / surface type),
- **`MaterialInstance` asset** = a parent `Material` ref + property value
  **overrides** (this specific chair's wood albedo/textures).

A `Primitive` references a `MaterialInstance` asset (resolved asynchronously, like
the mesh binding). The explicit split was chosen deliberately: a self-engine owner
who controls the graphics stack prefers explicit over implicit.

## Decision 7 — binding sets classified by update frequency (self-describing)

"Should instance properties be static or ring-buffered?" was a false choice. It's
not one-or-the-other: **each binding set (descriptor set / slang `ParameterBlock`)
carries an update-frequency class**, and the three coexist:

- **external** — owned by the render system (camera, lights); bound per view/frame,
- **dynamic** — ring-buffered, per draw (transform),
- **static** — a per-instance buffer uploaded once (material properties).

The class is **declared in the shader** (self-describing), via a slang user
attribute on each `ParameterBlock`, e.g. `[UpdateRate("static")]`. The render
system reads it via reflection and binds each set accordingly — the same
"declare-in-shader, render reads" philosophy as defines (Decision 5). This makes
Decision 1's binding contract *self-describing* rather than a hidden set-index
convention: the meaning lives next to the binding, in the shader, not in render
code.

Confirmed viable on our stack:

- `shader-slang 0.1` exposes user-attribute reflection — `UserAttribute` with
  `name()` + `argument_value_{string,int,float}()`, and `Variable`/`Type`
  `find_user_attribute_by_name(...)`; `reflection.parameters()` yields
  `VariableLayout`, whose `.variable()` gives the attributes.
- the engine already reflects per binding **space** (`binding_space()`) and already
  recognizes `TypeKind::ParameterBlock` (`graphics/src/shader/slang_compiler.rs`).
- graphics already supports all three classes: `MaterialInstance` holds static
  `binding_groups`; the `FrameRing` handles dynamic; the render system owns
  external.

Requires reorganizing shaders into frequency sets. The current opaque shader mixes
camera (`view_projection`, per view) and `model` (per draw) in one cbuffer (set 0);
split into camera→external, model→dynamic, material→static, each a
`ParameterBlock` carrying its `[UpdateRate]`.

**Implemented.** The attribute boilerplate lives in the `engine` shader
library module (`shaders/library/engine.slang`, auto-written next to the other
modules) — shaders `import engine;` and declare
`[UpdateRate("...")] ParameterBlock<T> gBlock;`. Reflection facts (pinned by
tests in `slang_compiler.rs` + `ecs/tests/std_shaders_reflect.rs`):

- a block's register space = its offset in the `SubElementRegisterSpace`
  category (`binding_space()` stays 0 for blocks);
- inside a block, uniform fields become the implicit constant buffer at
  binding 0 and each opaque field (texture/sampler) takes the next
  `DescriptorTableSlot` — matching the instance manager's static group
  convention (props buffer @0, texture/sampler pairs after);
- the attribute reads back per parameter via `VariableLayout::variable()` →
  `user_attributes()`; rates merge across stages (conflict = compile error).

`create_material` stores the per-set rates on the `Material`
(`set_update_rates()`) and auto-promotes a `dynamic` block's uniform buffer to
a dynamic-offset binding. `ForwardRender` assembles bind groups purely from
the rates: external → the camera block pushed once per view into the shared
ring (bound at fixed offset), dynamic → the model block (per-draw ring
offset), static → the instance's props group. The std opaque shaders declare
the canonical order camera/model/material; legacy `[[vk::binding]]` shaders
(entity_index, debug) still reflect through the old path with no rate class.

## How other engines escape the same corner

Everyone separates an authored, settings-agnostic surface from a compiled
pipeline (a permutation over the axes above) via a permutation/specialization
system. Two philosophies for the abstraction:

- **Material = surface; renderer owns the shader** — **Unreal** (material =
  attribute graph → cooked permutations over `vertex-factory × pass × quality ×
  static-switches`; Material Instance Dynamic = runtime params, Constant = static
  switches → new permutation), **Godot** (`StandardMaterial3D` = surface + feature
  flags → generated shader; renderer Forward+/Mobile/Compatibility is project
  level).
- **Material references a multi-variant shader** — **Unity** (Shader + `keywords`:
  `multi_compile` global / `shader_feature` material; SRP = passes; variant
  stripping), **Bevy** (material provides a shader + bind group;
  `SpecializedMeshPipeline` specializes per `material flags × mesh vertex layout ×
  view` → runtime cache; `#ifdef` shader-defs).

Compile timing: **cook-everything** (Unreal — permutation explosion, long cooks,
static-switch discipline) vs **on-demand + cache** (Bevy, Godot — only what's
used).

**Where we land:** the **Unreal/Godot abstraction** (material = surface, renderer
owns the shader) with the **Bevy mechanism** (on-demand pipeline cache keyed by
the axes). Best combination for a self-engine: settings-agnostic
materials/prefabs without Unreal's offline explosion. Combinatorics stay bounded
because shading models are few and pipelines compile on demand. The cost of
surface-only: a material can't carry its own custom shader (Unity/Bevy can) — a
new effect is a new shading model (code). Acceptable when you own the stack.

## Render flow (target end state)

```
Primitive ── MaterialInstance asset ── Material asset (surface: shading model + props)
        └─ Mesh asset ── VertexLayout asset

per draw:
  (material.shading_model + preset + features + hardware + mesh.layout + target.formats)
      → resolve shader asset + defines (shading-model variant table)
      → get/compile pipeline  (cache keyed by shader+defines+layout+formats)
      → draw: fill material binding from the instance's values,
              render bindings (camera/per-entity/lighting) from the render system.
```

## Resolved decisions (were forks)

- **Instance properties → GPU**: not a choice — Decision 7 classifies each binding
  set by update frequency (static buffer for material props, ring for per-draw,
  render-owned for external), declared in-shader.
- **Material / instance data location**: **DB record `settings`** (like
  VertexLayout) — small structured data (shading-model id + values + refs),
  editable via `ComponentField`, refs via `AssetRef`; the `.material`/`.matinst`
  file stays empty. (A file would only pay off for large authored data like a node
  graph — far off.)
- **Shading-model registry**: **start simple** — a fixed engine-defined set (one
  model, `opaque`, for the first slice). Make it game-extensible later if needed;
  cheap to change.

The **define ownership** is settled (Decision 5: declare-in-shader, select split,
cache-by-variant). What's still only sketched is the concrete *content* of the
variant system — the quality-preset and hardware-capability representations + the
shading models' render-side selection logic — but the first slice doesn't need it
(one preset, no variants).

## Build plan — minimal first, grow later

Build the architecture skeleton with **one shading model (`opaque`, the current
shader) and no variants**, then grow the variant/preset/hardware system on top
without rework:

1. **Material asset** (surface: shading model + `base_color`) + **MaterialInstance
   asset** (override) + their loaders/managers.
2. **Shading-model registry** (one entry) + **pipeline cache** keyed by
   `(shading-model + layout + formats)` — structure ready, the `defines` axis
   grows later.
3. **Primitive → MaterialInstance asset** migration (async resolve, like the mesh
   binding) + **demo migration**.

Later (separate efforts): the variant/preset/hardware/define system; deferred
path; textures as assets; per-material custom effects via new shading models.

## glTF is not a runtime source

glTF is **not** a runtime mesh/material source. There is no "insert this glTF as a
mesh" path. Instead glTF becomes an offline **glTF → prefab conversion** that
explicitly maps glTF materials onto **engine** materials (shading models +
instances), so the result is a correct, engine-native prefab. Consequently the
old name-based material path (`PrimitiveMaterial`, the name-keyed `MaterialManager`,
`MaterialBundle`, the ring `create_opaque_color_*` builders) can be removed wholesale
during the swap — nothing runtime depends on it once the demo uses asset instances.
The glTF importer is a separate future effort.
