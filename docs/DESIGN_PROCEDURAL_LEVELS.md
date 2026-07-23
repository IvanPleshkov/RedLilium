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
  **Cross-sections stay straight by design** (decided 2026-07-18): the
  authoring patch is a *reference surface*, not the final one. Crown,
  ditches, curbs — profile shape — is the mesh generator's responsibility,
  driven by semantic params (road class, profile id), with cross-seam
  profile continuity part of the generator's stitching metadata. Keeping
  the reference rows straight is what keeps `∂P/∂v`, edge picking, and
  seam math trivial; what the graph *does* express is width (it affects
  topology: landing intervals, junction loops) and bank/pitch via the
  node's rotation. Per-node profile geometry would also break network
  uniformity — the procedural win is that changing a class's profile in
  the generator updates the whole map.
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
  `TerrainControlPoint`. Terrain fills the regions *between* roads (faces
  of the planar graph formed by road edges, projected to ground plane);
  control points bend the interpolated surface where plain filling is too
  flat.
  **Terrain is the final fill** (decided 2026-07-19, refined 2026-07-22):
  it conforms to roads and architecture — road edges and strokes (open
  curves *with heights*) are boundary conditions of the fill, never
  constraints on them. The fill flows **continuously everywhere it is not
  told otherwise** — and the entity that tells it otherwise is a **`Cut`**
  (decided 2026-07-23, split out of the stroke design): strokes stay bare
  lines with no terrain obligations, a cut *breaks* the fill's continuity
  along its path (C1 crease or C0 step; the transition geometry — wall,
  slope, retaining — is generated, never authored as meshes). Closed
  regions with their own interior fill (the flat parking-pad case) are a
  **future level assembled from stroke/cut pieces**, not a primitive. In particular, nothing in the
  authoring layer assumes a ground plane: strokes and buildings live in
  full 3D, and edge-anchored elements inherit height from the edge they
  sit on.
- **Stroke** — the boundary primitive of the architecture chapter
  (reworked 2026-07-22, replacing the closed "parcel"): an **open
  polyline drawn point-wise onto the landscape** — a fence line, a scarp,
  a curb, a plot edge. `Stroke { points: Vec<Entity> }` lists child
  `StrokeVertex` entities in **explicit path order** (never re-derived);
  vertex local translations carry heights — a stroke rides the world in
  full 3D. Segments follow the **pen model**: each vertex carries two
  vertex-local Bézier handles (`handle_out` / `handle_in`); both adjacent
  handles zero → a straight segment, mirrored collinear handles → a **C1
  joint**, arbitrary handles → curves meeting at a corner (both requested
  cases, one mechanism). Handles being vertex-local means rotating a
  vertex with the gizmo steers its curve.
  **A stroke is bare geometry.** What it *means* is deliberately not
  encoded yet: stroke geometry will be handed to the generator alongside
  road geometry through a **single semantic mechanism designed later**,
  once architecture semantics can be specified. Because strokes are open
  lines, plots never need stitching: a border shared by two plots is
  *one* stroke. **Closure is the next level up** — closed contours (with
  a different interior fill) will be assembled from stroke pieces; the
  stroke itself is never closed.
  **Gates** (`Gate { segment, t, flip }`): connection sockets **glued to
  the stroke's path parametrically**, the same shape as an `EdgeAnchor`
  on a road edge — child `RoadNode`s whose local transform is *derived
  data*, recomputed from the parameter every frame, so a gate (and every
  road into it) follows any reshape of the stroke's points. `flip` picks
  which side of the line +Z faces (an open line has no interior; the
  side is authored at drop time). Dragging a gate with the gizmo
  *slides* it along the path (projection recovers the parameter — same
  undo story as anchored nodes). A gate is **two-sided**: a road is met
  from whichever side it comes from — network roads from the front
  (`b_from_front`), roads from behind connect from behind (other socket
  kinds stay front-only).
  **Grouping is plain hierarchy — there is no container component.** A
  root entity holding strokes, buildings and roads as its subtree IS the
  prefab ("villa" = fences + buildings + a driveway under one root).
  Concretely today: *Duplicate subtree* clones any selected entity's
  subtree through the generic `extract_prefab`/`instantiate` machinery
  with every internal reference (point lists, hierarchy, road endpoints)
  remapped; the copy drops its edge anchor and starts free. Prefab
  *assets* (a villa recipe on disk) ride the existing
  prefab-serialization machinery in a later chapter.
  A stroke may carry an optional single `EdgeAnchor` gluing it to a road
  edge with **inverted derivation**: the rigid stroke dictates its
  frontage length (the local distance between its first two vertices),
  the edge interval's *width* derives from it, and only the interval's
  center is authored — it slides along the road under the gizmo. The
  anchored transform faces the road (+Z into it, the tail extending
  outward). Two anchors would over-constrain a rigid stroke and are
  rejected.
- **Cut** — the terrain-discontinuity line (decided 2026-07-23): where a
  stroke is bare geometry with no obligations, a cut **obliges the fill**
  — along its path the landscape creases (C1 break, C0 kept) or steps (C0
  break — a height jump): pedestal rims, pool walls, moats, small cliffs,
  embankments. Kept a **separate entity from the stroke** so the contract
  is visible in the type — every stroke consumer would otherwise have to
  remember "might also cut". The split is by *component*, not by code:
  cuts reuse the shared vertex machinery (ordered `StrokeVertex` children,
  pen handles, tessellation, two-tier picking).
  **Master + profile** (the ruled-out alternative was two independently
  authored curves — correspondence problems, desync, double editing): the
  authored path is the **upper lip**; each vertex adds
  `CutVertex { drop }` and the **lower lip is derived** — sunk `drop`
  meters straight down in world Y, the drop interpolating along segments.
  `drop = 0` collapses the step into a pure crease; positive drop lowers
  the path's **right-hand side** (of travel direction). One master
  parameterization keeps attachments and the future closure level on a
  single locus per piece. This slice: vertical faces only; the planned
  extension is a per-vertex **plan offset** of the lower lip for battered
  slopes/embankments (with the offset-curve self-intersection caveat on
  concave bends — clamp by curvature or leave it to the generator).
  **Division of labor for crossings** (a road gate through a cut = stairs,
  a ramp): the engine owns the *intent* and the boundary conditions — the
  semantic tag and the two full-3D chords (road edge + cut lip), plus the
  **claimed interval on the cut face** the crossing consumes (so the
  generator doesn't also build wall there — symmetric to how an edge
  anchor claims `[u_min, u_max]`). The generated volume (steps, railings,
  collision) is tyroxine's. Cut faces themselves are generator geometry:
  the cut owns only the boundary description.
- **Edge anchor** (P5) — the way a connection lands on **part of a road's
  boundary curve** rather than on a node. The canonical case: a
  driveway/building exit crossing the sidewalk to meet the road. Not a
  separate patch kind: `EdgeAnchor { parent_road, right_edge, u_min, u_max }`
  is a component on an ordinary `RoadNode`, gluing it to the parent's edge.
  The node's `Transform` and `half_width` are **derived data** (like
  `GlobalTransform`, recomputed by a bounded fixed-point pass each frame):
  position/orientation come from the **chord** between the edge points at
  `u_min`/`u_max` — local X along the chord, +Z outward from the road —
  and `half_width` is half the chord. The chord is taken **in full 3D**
  (decided 2026-07-23): its ends land exactly on the contour points, tilt
  included — the connecting segment is *defined by two points on the
  contour it attaches to*, never projected onto a ground plane. The same
  rule holds for gates on strokes: a gate's cross-section is the chord
  between two curve points spanning its width. The seam is that
  *straight* chord, not the curved edge: **a road piece's geometry is
  always the span between two straight segments**, and closing the sliver
  between chord and true edge curve is the mesh generator's job — it
  receives the parametric interval in the `PieceDesc`, the same division
  of labor as cut profiles.
  Because the interval is parametric, the anchored node follows every
  parent-road edit. The driveway itself is then an **ordinary
  `RoadSegment`** out of the anchored node — its own edges are pickable
  and anchorable like any road's, so chains (road → driveway → parking
  lot) need no special support. **Unified socket convention** (same as
  junction connectors): the anchored node's **+Z faces outward, into the
  road network** — a chain grows out of it as its `a` end; a road arriving
  at it from the field meets it from the front (`b_from_front`). One rule
  everywhere: *a socket node is an `a`-end; +Z points into the road
  network; the structure behind it owns −Z.*
- **Intersections are hand-managed.** The author places nodes and owns the
  result. A validator flags *unsanctioned* crossings — two roads
  intersecting in projection with neither a shared node nor an edge anchor —
  as authoring errors; it never tries to auto-resolve them.
- **Building** — free-standing architecture content: an **ordinary
  entity** with its own transform and its own footprint
  (`half_width × half_depth` local rectangle); group any number under a
  root entity via plain hierarchy (a villa, or a factory full of
  structures — the group subtree is the prefab). The component carries
  flat box-massing recipe parameters (floors, floor height, footprint
  extents, seed) — the P4 stub of the eventual **assembly graph asset**
  (a reusable recipe: one "землянка" recipe, ten placements); the fields
  are already the asset's fields, promotion is mechanical once AssetRef
  inspector editing lands. Connections to the road network are gates on
  strokes or edge anchors — never building fields. How terrain meets a
  building footprint is the generator's decision later; there is no
  footprint cut-out in the authoring layer. Interior/exterior volumes,
  occluders and gameplay metadata still get their own design round
  (phase 3, §7).

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
  **edge anchors** (driveways landing on this road's edge, with their
  u-intervals and chord cross-sections) and **through-features** for
  intersections (e.g. tram tracks crossing the junction: a feature curve
  with its own profile and params — data, not code). A procedural
  intersection with tram tracks is therefore a *rich descriptor handed to
  tyroxine*, not junction code in this crate.
- Roads normally stay one piece per graph edge, with edge anchors described
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
3. **Phase 3 — buildings.** Assembly-graph asset, cut-driven terrain
   seams (the `Cut` faces excavated/embanked by the generator),
   interior/exterior + occluder metadata. Gets its own design doc/round.
4. **Later** (explicitly deferred): tyroxine as the real generator, navmesh,
   LOD baking (LOD1 and below), incremental invalidation, streaming budgets.

## 8. Open questions

- Control-point space for the 8 free road points: world space vs local to
  the road entity (leaning local — survives moving the whole road).
- Terrain region extraction: exact algorithm for faces of the planar road
  graph. Unsanctioned crossings are validator errors (§3), which bounds the
  problem; still a phase 2 decision.
- **Interior trims** — a connection landing in the *interior* of a surface
  (a trim loop inside the (u,v) domain) rather than on its boundary curve.
  Deliberately deferred until a real need appears; edges and cross-sections
  cover everything described so far.
- Chunk cell identity for terrain regions (roads/intersections have natural
  ids; regions appear/disappear as the graph changes).
- **Stroke semantics & terrain coupling** — how a stroke's meaning
  (fence, lamp line, pipe run, curb, plot edge) is declared and handed to
  tyroxine alongside road geometry: one mechanism for all architecture
  semantics, designed once it can be specified (§3: strokes are bare
  geometry today). Also open: how stroke/cut lips parameterize the
  terrain boundary condition, how a cut face's treatment (bare wall vs
  retaining vs slope, once the plan offset lands) is selected, the
  crossing semantics (stairs/ramp through a cut, with its claimed face
  interval), and how **closed contours assembled from stroke/cut pieces**
  (the next level up) get their own interior fill.
- ~~Road attachment side~~ — **resolved** (2026-07-18): `RoadSegment` grew
  `b_from_front` — the road meets `b` on its +Z side instead of the default
  −Z. The connect tool sets it automatically when the clicked target is a
  socket (junction connector / edge-anchored node), which also makes
  socket↔socket roads representable: depart `a` along +Z, enter `b` from
  the front. The `a` end needs no flag — an anchor socket departs along
  its +Z by construction.
- Crate name.
