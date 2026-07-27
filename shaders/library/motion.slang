// RedLilium Shader Library - Motion Module
// Screen-space motion / reprojection helpers shared by the temporal passes
// (TAA #148, motion blur #149). All conventions match the deferred G-buffer:
//   - UV is top-left origin, y-down.
//   - NDC is y-up, in [-1, 1].
//   - Depth is reversed-Z: the far plane is depth 0.
//   - Screen velocity is stored in NDC as (curr - prev) — the
//     `GBUFFER_VELOCITY` convention (#147).

// UV (top-left, y-down) -> NDC (y-up, [-1, 1]).
float2 uv_to_ndc(float2 uv) {
    return (uv - 0.5) * float2(2.0, -2.0);
}

// NDC (y-up, [-1, 1]) -> UV (top-left, y-down).
float2 ndc_to_uv(float2 ndc) {
    return ndc * float2(0.5, -0.5) + 0.5;
}

// World-space position of the sample at `uv` with raster depth `depth`
// (reversed-Z), through the inverse of the view-projection that rasterized it
// (the jittered inverse reconstructs the raster sample exactly; the unjittered
// one is off by the sub-pixel jitter — fine for reprojection/depth proxies).
// This replaces storing world position in the G-buffer: the Depth32Float
// buffer holds full precision everywhere, where an f16 position target loses
// ~0.5 units of precision at |coord| ~ 1000.
float3 reconstruct_world(float4x4 inv_view_proj, float2 uv, float depth) {
    float4 w = mul(inv_view_proj, float4(uv_to_ndc(uv), depth, 1.0));
    return w.xyz / w.w;
}

// World-space point on the far plane along the ray through `uv`
// (reversed-Z: far = depth 0). Used to reproject background/sky pixels, which
// have no geometry of their own.
float3 far_plane_world(float4x4 inv_view_proj, float2 uv) {
    return reconstruct_world(inv_view_proj, uv, 0.0);
}

// NDC of a world-space point under `view_proj` (w guarded against 0).
float2 project_ndc(float4x4 view_proj, float3 world_pos) {
    float4 c = mul(view_proj, float4(world_pos, 1.0));
    return c.xy / max(c.w, 1e-6);
}

// UV of a world-space point under `view_proj`.
float2 project_uv(float4x4 view_proj, float3 world_pos) {
    return ndc_to_uv(project_ndc(view_proj, world_pos));
}

// Camera-only screen velocity (NDC, curr - prev) for a pixel whose current NDC
// is `curr_ndc` and world position is `world_pos`, given the previous
// unjittered view-projection. Object-independent: it measures only how the
// camera moved this world point across the screen. Matches the
// `GBUFFER_VELOCITY` sign convention (#147), so it can be written straight into
// that buffer for background pixels or compared against per-pixel velocity.
float2 camera_velocity_ndc(float4x4 prev_view_proj, float2 curr_ndc, float3 world_pos) {
    return curr_ndc - project_ndc(prev_view_proj, world_pos);
}

// Previous-frame UV of `uv` given its NDC velocity (the value stored in /
// matching GBUFFER_VELOCITY: curr - prev).
float2 reproject_prev_uv(float2 uv, float2 velocity_ndc) {
    return uv - velocity_ndc * float2(0.5, -0.5);
}
