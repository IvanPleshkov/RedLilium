// Debug draw shader — simple unlit line/triangle shader.
//
// Transforms position by a view-projection matrix and passes through vertex color.
// Used by the debug drawer for rendering lines, wireframes, and debug overlays.
//
// Binding group 0: Debug uniforms (VP matrix).

[[vk::binding(0, 0)]]
cbuffer DebugUniforms {
    column_major float4x4 view_proj;
};

struct VsInput {
    [[vk::location(0)]] float3 position : POSITION;
    [[vk::location(5)]] float4 color : COLOR;
};

struct VsOutput {
    float4 position : SV_Position;
    float4 color : COLOR;
};

[shader("vertex")]
VsOutput vs_main(VsInput input) {
    VsOutput output;
    output.position = mul(view_proj, float4(input.position, 1.0));
    output.color = input.color;
    return output;
}

[shader("fragment")]
float4 fs_main(VsOutput input) : SV_Target {
    return input.color;
}
