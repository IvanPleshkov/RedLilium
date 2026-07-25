// Selection outline: fullscreen triangle compositing a contour around the
// selected entities' visible silhouette onto the scene camera's color target.
//
// Reads the R8 selection mask written by entity_index.slang. The mask is
// window-sized while the target is panel-sized; mask_offset maps target
// pixels to mask texels. A pixel is outlined when it lies OUTSIDE the mask
// but within `thickness` pixels of a masked texel, so the contour never
// covers the object itself. Every other pixel discards, leaving the scene
// untouched.

[[vk::binding(0, 0)]]
cbuffer Uniforms {
    float2 mask_offset;  // panel origin inside the mask texture, pixels
    float2 mask_size;    // mask texture dimensions, pixels
    float4 color;        // outline color
    float thickness;     // outline width, pixels
};

[[vk::binding(1, 0)]]
Texture2D<float> mask_texture;

[shader("vertex")]
float4 vs_main(uint vertex_id : SV_VertexID) : SV_Position {
    float2 positions[3] = {
        float2(-1.0, -1.0),
        float2(3.0, -1.0),
        float2(-1.0, 3.0)
    };
    return float4(positions[vertex_id], 0.0, 1.0);
}

[shader("fragment")]
float4 fs_main(float4 frag_coord : SV_Position) : SV_Target {
    int2 center = int2(frag_coord.xy + mask_offset);
    if (mask_texture.Load(int3(center, 0)) > 0.5) {
        discard;
    }
    int radius = int(thickness);
    int2 max_texel = int2(mask_size) - int2(1, 1);
    float hit = 0.0;
    for (int dy = -radius; dy <= radius; dy++) {
        for (int dx = -radius; dx <= radius; dx++) {
            int2 texel = clamp(center + int2(dx, dy), int2(0, 0), max_texel);
            hit = max(hit, mask_texture.Load(int3(texel, 0)));
        }
    }
    if (hit < 0.5) {
        discard;
    }
    return color;
}
