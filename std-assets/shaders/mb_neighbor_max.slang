// Motion blur — NeighborMax (#149; McGuire 2012 + Guertin 2014 directional).
//
// For each tile, the dominant blur among its 3x3 tile neighbourhood: a fast
// tile smears into adjacent tiles, so every tile must know the largest motion
// that can *reach* it. McGuire's original takes the plain max magnitude;
// Guertin's refinement only lets a neighbour contribute when its velocity
// actually streaks back toward this tile (`dot(v, offset) < 0`), which avoids
// over-blurring static tiles next to a fast one.
//
// Runs at tile resolution, one fragment per tile; reads TileMax.

[[vk::binding(0, 0)]]
cbuffer MbNeighborUniforms {
    // 1 / tile-resolution extent (tile texel size).
    float2 inv_tile_resolution;
    float2 _pad;
};

[[vk::binding(1, 0)]]
Texture2D tile_max;
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
    int2 tile = int2(input.position.xy);

    // The centre tile always participates (its own blur covers itself).
    float2 center = tile_max.Sample(
        nearest_sampler, (float2(tile) + 0.5) * inv_tile_resolution).xy;
    float2 best = center;
    float best_len = length(center);

    for (int y = -1; y <= 1; y++) {
        for (int x = -1; x <= 1; x++) {
            if (x == 0 && y == 0) {
                continue;
            }
            float2 uv = (float2(tile + int2(x, y)) + 0.5) * inv_tile_resolution;
            float2 v = tile_max.Sample(nearest_sampler, uv).xy;
            float len = length(v);
            // Only a neighbour whose velocity points back toward this tile
            // (opposite the offset that points away from us) can smear over it.
            if (len > best_len && dot(v, float2(x, y)) < 0.0) {
                best_len = len;
                best = v;
            }
        }
    }
    return best;
}
