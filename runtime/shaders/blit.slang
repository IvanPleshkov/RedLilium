// Present blit: fullscreen triangle sampling the scene texture.

struct VsOutput {
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

[shader("vertex")]
VsOutput vs_main(uint vertex_id : SV_VertexID) {
    float2 positions[3] = {
        float2(-1.0, -1.0),
        float2(3.0, -1.0),
        float2(-1.0, 3.0)
    };
    // V flipped: NDC -1 is the bottom of the swapchain, texel v=0 is the top
    // of the scene texture.
    float2 uvs[3] = {
        float2(0.0, 1.0),
        float2(2.0, 1.0),
        float2(0.0, -1.0)
    };
    VsOutput output;
    output.position = float4(positions[vertex_id], 0.0, 1.0);
    output.uv = uvs[vertex_id];
    return output;
}

[[vk::binding(0, 0)]]
Texture2D source_texture;
[[vk::binding(1, 0)]]
SamplerState source_sampler;

[shader("fragment")]
float4 fs_main(VsOutput input) : SV_Target {
    return source_texture.Sample(source_sampler, input.uv);
}
