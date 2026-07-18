# Procedural Levels — roads, terrain, buildings (design draft)

Status: **draft for review** (2026-07-17). Not yet tied to issues — issues get
cut from this doc once the design is approved.

Consumer context: the parallel **tyroxine** project will produce procedural
pieces (road segments, buildings, blocks) as engine-native `CpuMesh` at
runtime, including stitching metadata. The engine currently has nowhere to put
them. This doc designs that place.

## 1. Motivation

Levels are authored as **semantics, not geometry**: a graph of road nodes and
connections, terrain shape hints, building assembly graphs. Geometry is a
derived artifact, generated at runtime around cameras and thrown away.

Why semantics-first (and not "place meshes by hand"):

- The graph is small, cheap to edit, cheap to undo, cheap to serialize.
- Everything downstream is *derivable*: geometry today; navmesh, occluders,
  interior/exterior volumes, spawn points, traffic lanes tomorrow — without
  reverse-engineering triangle soup. (Navmesh is explicitly **out of scope
  now**; it is listed as the canonical example of what level semantics buys
  us later.)
- LOD baking later is just another consumer of the same deterministic
  generator.

## 2. Principles

- **P1 — Graph is the source of truth.** Scenes store only the authoring
  graph. Generated geometry is a runtime cache, never serialized.
- **P2 — Hard crate boundary.** All of this lives in one new workspace crate
  (working name `redlilium-levels`; bikesheddable). **No domain abstraction
  leaks into `ecs` or `editor`.** When the editor is not flexible enough, we
  extend it with *generic* capabilities (see §6), never with road/terrain
  types. Rationale: the crate retains the right to make design mistakes and
  redesign freely without the editor/ecs paying for it.
- **P3 — Determinism.** Same graph revision + same cell ⇒ same geometry.
  Required for stable rebuild under camera movement and for future LOD baking.
- **P4 — Stub-first generation.** The crate ships its own placeholder
  generators (Bézier tessellation, flat fills). tyroxine plugs in later
  per piece kind. "No generator assigned for this piece" is a permanently
  supported state, answered by the stub — a deliberate fallback, not a
  temporary hack. Direction of travel: **push all procedural detail work to
  tyroxine as soon as it can take it**; the crate keeps only topology,
  parameterization and stubs.
- **P5 — Geometry never meets geometry.** Junctions, attachments and
  crossings are resolved in the **parameter space** of the surfaces
  involved; the generator materializes the result. Mesh×mesh CSG
  ("intersect in projection, reconstruct z at new vertices") is **rejected**:
  it is the buggy option *and* the only one that destroys semantics — after
  a boolean you have triangles from which "this is a driveway" can no longer
  be recovered, killing P1 at that exact point. On a parametric patch, z is
  never reconstructed — it is *evaluated* at (u,v).

## 3. Authoring layer (entities in the scene)

Authoring elements are ordinary entities with `Transform` + components from
the plugin crate. This is deliberate: the editor already knows how to select,
move, undo (EditAction) and serialize entities with transforms — placing
intersections *is* transform editing, and the editor stays ignorant of what
the entities mean.

- **Road node** — a **straight segment** (a cross-section, "срез"), not a
  point. Entity with `Transform` + `RoadNode` (segment half-length; the
  segment runs along a local axis, centered at the entity origin). A node is
  where roads attach: for each attached road, the node contributes **4 patch
  control points distributed uniformly along its segment**.
- **Road** — connects two nodes. Entity with
  `RoadSegment { start: Entity, end: Entity, control: [Vec3; 8], params }`.
  The road surface is a **bicubic Bézier patch (4×4 = 16 control points)**:
  rows 0 and 3 come from the two node segments (8 points, derived, not
  stored), rows 1 and 2 are the 8 free control points stored on the road
  (defaulted from node directions, then hand-editable). `params` carries
  surface semantics (road vs path, material class, …) interpreted by the
  generator.
- **Junction** — N connector nodes closed into one boundary loop:
  `Junction { connectors: Vec<Entity>, corner_tangent }`. Connectors are
  *ordinary road nodes* (the same entity is a road's endpoint and the
  junction's socket); their order is never stored — the loop re-derives by
  angle around the centroid on every evaluation, so dragging connectors
  cannot corrupt authored data. The boundary alternates cross-sections with
  **corner curves**: cubic Béziers leaving a connector along its inward −Z
  and arriving along the next connector's outward +Z — G1-continuous with
  the roads' side edges (curbs flow road → corner → road, watertight per
  P5). The three authoring cases are one continuum, distinguished only at
  evaluation: a "convex quad" 4-way is the degenerate loop whose corner
  curves collapse to points (the *generator* may special-case it into a
  single patch); connectors that don't touch are the general case (corners
  span the gaps); a 3-way T/Y is the same loop at N = 3 — as is any N.
  Interior fill for previews/stubs: fan from the centroid; quality N-sided
  patches are the generator's concern.
  **Convention:** a connector's **+Z faces outward** (the road side); the
  junction fills the −Z side. Roads should attach to a connector as their
  `a` end (departing along +Z).
- **Terrain control point** — entity with `Transform` +
  `TerrainControlPoint`. Terrain fills the regions *between* roads (faces of
  the planar road graph, projected to ground plane); control points bend the
  interpolated surface where plain filling is too flat.
- **Edge attachment** (P5) — the third connection type in the graph, next to
  node–node roads: a connection that lands on **part of a road's boundary
  curve** rather than on a node. The canonical case: a driveway/building
  exit crossing the sidewalk to meet the road. Authored by snapping the
  exit's cross-section to a road edge; the tool projects it and stores
  `Attachment { road: Entity, side, u_interval: [u0, u1], params }` as an
  explicit, editable object. The attached patch samples its end row directly
  from the road's boundary curve on that interval — the seam is **watertight
  by construction** (shared curve, not two meshes stitched after the fact).
  The attachment is passed to the road's generator in its `PieceDesc`, and
  the generator resolves the junction itself (curb break, sidewalk ramp, …).
- **Intersections are hand-managed.** The author places nodes and owns the
  result. A validator flags *unsanctioned* crossings — two roads
  intersecting in projection with neither a shared node nor an attachment —
  as authoring errors; it never tries to auto-resolve them.
- **Building** — entity with `Transform` + a reference to an **assembly
  graph asset** (a reusable recipe: one "землянка" recipe, ten placements).
  Its footprint cuts a hole in the terrain region it sits in and the assembly
  graph generates the structure. From the same graph we later extract
  interior/exterior volumes, occluders, and gameplay metadata. Details get
  their own design round (phase 3, §7).

Graph edits go through the standard `EditAction`/`ActionQueue` path like any
other entity/component edit — the plugin's editor tools produce actions, never
mutate the world directly (HARD RULE 1).

## 4. Runtime layer (transient chunk entities)

- Cameras that drive generation carry a `GenerationTracker` component
  (radius, budget). A plugin system spawns/despawns **chunk entities** for
  cells within radius.
- A **chunk** is one unit of generation — one road patch, one intersection
  fill, one terrain region, one building — not one polygon. Each chunk entity
  carries the generated mesh (renderer components) plus **metadata
  components** (interior/exterior, occluder, gameplay tags). Metadata as ECS
  components is the point: game logic and graphics query it with ordinary
  queries instead of going through a plugin API.
- Chunk entities are **transient**: never serialized (needs the generic
  editor/ecs capability in §6.1).
- **Rebuild policy (now):** one global graph revision counter; any authoring
  edit bumps it and all live chunks regenerate. Full rebuild is deliberately
  chosen for simplicity; incremental (cell ← node dependency) invalidation is
  a later optimization and requires generation locality from the generator.
  The camera radius decides *which* cells exist; the revision decides *when*
  they regenerate.
- Generation runs on the engine's cooperative async compute
  (`redlilium_core::compute` + ecs `ComputePool`); results land as `CpuMesh`
  and reach the GPU **only** via `create_mesh_deferred()` + frame-graph
  transfer uploads (HARD RULE 2). Scheduling model:
  - **One piece = one task** at `Priority::Low` (gap-filler, may span
    frames; `Critical` stays reserved for ECS systems). Generating a single
    piece is single-threaded; parallelism comes from many pieces in flight
    saturating free worker slots.
  - **Task identity = `(cell id, graph revision)`.** Results carrying a
    stale revision are dropped on receipt; cancellation exists to stop
    wasting CPU earlier, correctness never depends on it.
  - The plugin keeps its own **pending queue**, scored by distance to the
    nearest `GenerationTracker` camera, and feeds the pool only a small
    in-flight window (~2× worker count). Each frame a scheduler system:
    drains completions (`TaskHandle::try_recv`), **cancels** tasks whose
    cell left the radius or whose revision went stale
    (`CancellationToken::cancel()` — the task stops cooperatively at its
    next `checkpoint()`), re-scores the pending queue against current
    camera positions, and tops the window back up. The window is what makes
    per-frame re-prioritization possible: work already handed to the pool
    cannot be reordered, so we hand over little at a time.
  - **Replace-on-ready:** on a revision bump, live chunk entities stay
    visible until the replacement piece for the same cell completes — a
    full rebuild must not blank the world. Leaving the camera radius, by
    contrast, despawns immediately.
  - Completed meshes are budgeted onto the frame graph (N uploads/frame)
    to avoid transfer spikes.
- Generation also runs in **edit mode** — you see the landscape while editing
  the graph. Same systems, same chunks; play/stop does not own this pipeline.

## 5. Generator contract

Inside the plugin crate:

```rust
// sketch, not final signatures
trait PieceGenerator {
    async fn generate<C: ComputeContext>(
        &self,
        ctx: &C,
        piece: &PieceDesc,
    ) -> Result<GeneratedPiece, Cancelled>;
}
struct GeneratedPiece {
    mesh: CpuMesh,          // engine-native, ready for create_mesh_deferred
    metadata: PieceMetadata, // stitching, semantic tags, …
}
```

The contract is **async and cooperative**: implementations must call
`ctx.checkpoint().await?` at reasonable intervals (per tessellation row,
per assembly step) so a piece that is no longer needed — camera moved on,
revision went stale — stops promptly instead of burning a worker slot.
`ComputeContext` lives in `redlilium-core` precisely so standalone
generator libraries can be generic over it without linking the ECS.

- `PieceDesc` carries the resolved inputs: the 16 patch points for a road,
  region boundary + control points for terrain, assembly graph for a
  building, plus neighbor stitching info. Per P5 it also carries the piece's
  **boundary attachments** (driveways landing on this road's edge, with
  their u-intervals and cross-sections) and **through-features** for
  intersections (e.g. tram tracks crossing the junction: a feature curve
  with its own profile and params — data, not code). A procedural
  intersection with tram tracks is therefore a *rich descriptor handed to
  tyroxine*, not junction code in this crate.
- Roads normally stay one piece per graph edge, with attachments described
  in the descriptor. When semantic granularity genuinely requires splitting
  a road into parts (different params per stretch), the tool subdivides the
  Bézier patch via **de Casteljau** — subdivision of a Bézier patch at a
  parameter is *exact* (a few lerps on control points, zero geometric
  error), so "road parts" are cheap at the parametric level even though
  they would be scary at the mesh level. Internal tool, not the default
  model.
- A registry maps piece kind → generator. Lookup miss ⇒ **stub generator**
  (P4): road = direct tessellation of the Bézier patch, terrain = planar
  fill, building = box massing.
- **tyroxine integration** = implementing this trait (or an adapter over its
  API). tyroxine already speaks `CpuMesh`; stitching metadata is in its
  project scope. Determinism (P3) is a contract requirement on
  implementations.

## 6. Editor/ecs: generic capabilities we will need

Everything below is domain-agnostic editor/ecs flexibility. This is the
"если упираемся — прокачиваем гибкость редактора" list, ordered by certainty:

1. **Transient entities** — a marker/flag the serializer skips, so the chunk
   layer never leaks into scene files.
2. **Plugin viewport tools** — an extension point to register interactive
   tools (click-to-place node, drag a connection between nodes). Until it
   exists, phase 1 authoring works through ordinary entity creation + the
   inspector.
3. **Entity-reference fields in the inspector** — editing
   `RoadSegment::start/end` (overlaps issue #73's AssetRef editing work).
4. **Plugin debug-draw overlay** — draw the graph itself (segments, patch
   control cages) as lines over the viewport.

## 7. Phasing

1. **Phase 1 — pipeline end-to-end with stubs.** New crate; `RoadNode` +
   `RoadSegment` components; stub Bézier tessellation; `GenerationTracker` +
   chunk spawner; transient entities (§6.1). Authoring via plain entity
   creation + inspector. Exit criterion: place two nodes, connect a road,
   see a generated surface in the viewport, undo works, scene file contains
   only the graph.
2. **Phase 2 — terrain + tools.** Region extraction between roads (planar
   graph faces), terrain fill + control points, intersection surfaces,
   viewport tools (§6.2), debug-draw (§6.4).
3. **Phase 3 — buildings.** Assembly-graph asset, terrain cut-outs,
   interior/exterior + occluder metadata. Gets its own design doc/round.
4. **Later** (explicitly deferred): tyroxine as the real generator, navmesh,
   LOD baking (LOD1 and below), incremental invalidation, streaming budgets.

## 8. Open questions

- Control-point space for the 8 free road points: world space vs local to
  the road entity (leaning local — survives moving the whole road).
- Terrain region extraction: exact algorithm for faces of the planar road
  graph. Unsanctioned crossings are validator errors (§3), which bounds the
  problem; still a phase 2 decision.
- **Interior trims** — an attachment landing in the *interior* of a surface
  (a trim loop inside the (u,v) domain) rather than on its boundary curve.
  Deliberately deferred until a real need appears; edges and cross-sections
  cover everything described so far.
- Chunk cell identity for terrain regions (roads/intersections have natural
  ids; regions appear/disappear as the graph changes).
- Road attachment side: today a road always departs its `a` node along +Z
  and arrives at its `b` node from −Z, so a junction connector must be the
  road's `a` end (the junction owns −Z). Whether `RoadSegment` should grow
  an explicit per-end side instead of this convention — decide when it
  first bites.
- Crate name.
