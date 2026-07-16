// Depth-only shader — writes SV_Position only, for zero-color-attachment
// passes (shadow maps, depth prepass; ADR-035/#129). Vertex stage only: with
// no color attachments the fragment stage is omitted entirely and depth is
// written by the fixed-function path.
//
// Shares the standard position + normal vertex layout with the opaque
// shaders (the normal attribute is declared but unused) so any mesh drawn by
// the forward path can be re-drawn depth-only with the same vertex buffers.
// Binding sets follow the [UpdateRate] convention (docs/MATERIAL_ASSETS.md
// Decision 7): 0 camera (external), 1 model (dynamic); no static set.
import engine;

struct CameraParams {
    column_major float4x4 view_projection;
};

struct ModelParams {
    column_major float4x4 model;
};

[UpdateRate("external")]
ParameterBlock<CameraParams> gCamera;

[UpdateRate("dynamic")]
ParameterBlock<ModelParams> gModel;

struct VsInput {
    [[vk::location(0)]] float3 position : POSITION;
    [[vk::location(1)]] float3 normal : NORMAL;
};

struct VsOutput {
    float4 clip_position : SV_Position;
};

[shader("vertex")]
VsOutput vs_main(VsInput input) {
    VsOutput output;
    float4 world_pos = mul(gModel.model, float4(input.position, 1.0));
    output.clip_position = mul(gCamera.view_projection, world_pos);
    return output;
}
