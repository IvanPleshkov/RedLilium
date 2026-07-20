// Motion blur — TileMax (#149, McGuire et al. I3D 2012).
//
// Reduces the full-resolution GBUFFER_VELOCITY to one dominant blur vector per
// KxK tile: the per-tile maximum-magnitude screen-space velocity, already
// converted to pixels, scaled by the shutter, and clamped to the tile radius
// (the filter's max blur length). NeighborMax then spreads it to neighbours so
// a fast tile blurs into the slow ones around it.
//
// Runs at tile resolution (ceil(W/K) x ceil(H/K)); each fragment is one tile.

import motion; // ndc/px velocity conventions (#149)

[[vk::binding(0, 0)]]
cbuffer MbTileUniforms {
    // 1 / full-resolution (velocity texel size).
    float2 inv_resolution;
    // Full-resolution extent, pixels.
    float2 resolution;
    // Shutter fraction (per-frame velocity -> blur length).
    float shutter;
    // Tile size K, pixels — also the max blur radius (velocity clamp).
    float tile_size;
    float2 _pad;
};

[[vk::binding(1, 0)]]
Texture2D gbuffer_velocity;
[[vk::binding(2, 0)]]
SamplerState nearest_sampler;

struct VsOutput {
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

[shader("vertex")]
VsOutput vs_main(uint vertex_id : SV_VertexID) {
    VsOutput output;
    float2 positions[3] = { float2(-1.0, -1.0), float2(3.0, -1.0), float2(-1.0, 3.0) };
    float2 uvs[3] = { float2(0.0, 1.0), float2(2.0, 1.0), float2(0.0, -1.0) };
    output.position = float4(positions[vertex_id], 0.0, 1.0);
    output.uv = uvs[vertex_id];
    return output;
}

[shader("fragment")]
float2 fs_main(VsOutput input) : SV_Target {
    int k = int(tile_size);
    // This fragment is one tile; its top-left full-res pixel.
    int2 base = int2(input.position.xy) * k;

    float2 best = float2(0.0, 0.0);
    float best_len = 0.0;
    for (int j = 0; j < k; j++) {
        for (int i = 0; i < k; i++) {
            float2 uv = (float2(base + int2(i, j)) + 0.5) * inv_resolution;
            float2 v_ndc = gbuffer_velocity.Sample(nearest_sampler, uv).xy;
            // NDC (curr - prev) -> screen pixels (y flips), scaled by shutter.
            float2 v_px = v_ndc * float2(0.5, -0.5) * resolution * shutter;
            // Clamp to the tile radius so no tile advertises more blur than the
            // filter can gather.
            float len = length(v_px);
            if (len > tile_size) {
                v_px *= tile_size / len;
                len = tile_size;
            }
            if (len > best_len) {
                best_len = len;
                best = v_px;
            }
        }
    }
    return best;
}
