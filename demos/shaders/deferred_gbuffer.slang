// Deferred Rendering - G-Buffer Pass
// Outputs material properties to multiple render targets for later lighting calculation.

import math;

// =============================================================================
// Shared Types
// =============================================================================

struct InstanceData {
    column_major float4x4 model;
    float4 base_color;
    float4 metallic_roughness;
};

// =============================================================================
// Bindings
// =============================================================================

[[vk::binding(0, 0)]]
cbuffer CameraUniforms {
    column_major float4x4 view_proj;
    column_major float4x4 view;
    column_major float4x4 proj;
    float4 camera_pos;
};

[[vk::binding(1, 0)]]
StructuredBuffer<InstanceData> instances;

// =============================================================================
// Vertex Shader
// =============================================================================

struct VsInput {
    [[vk::location(0)]] float3 position : POSITION;
    [[vk::location(1)]] float3 normal : NORMAL;
    [[vk::location(3)]] float2 uv : TEXCOORD0;
    uint instance_id : SV_InstanceID;
};

struct VsOutput {
    float4 position : SV_Position;
    float3 world_position : TEXCOORD0;
    float3 world_normal : TEXCOORD1;
    float2 uv : TEXCOORD2;
    float4 base_color : TEXCOORD3;
    float metallic : TEXCOORD4;
    float roughness : TEXCOORD5;
};

[shader("vertex")]
VsOutput vs_main(VsInput input) {
    VsOutput output;

    InstanceData instance = instances[input.instance_id];
    float4 world_pos = mul(instance.model, float4(input.position, 1.0));
    float3x3 normal_matrix = float3x3(
        instance.model[0].xyz,
        instance.model[1].xyz,
        instance.model[2].xyz
    );

    output.position = mul(view_proj, world_pos);
    output.world_position = world_pos.xyz;
    output.world_normal = normalize(mul(normal_matrix, input.normal));
    output.uv = input.uv;
    output.base_color = instance.base_color;
    output.metallic = instance.metallic_roughness.x;
    output.roughness = instance.metallic_roughness.y;
    return output;
}

// =============================================================================
// Fragment Shader
// =============================================================================

struct FsOutput {
    // RT0: Albedo (RGB) - sRGB color space
    float4 albedo : SV_Target0;
    // RT1: World Normal (RGB) + Metallic (A) - linear, high precision
    float4 normal_metallic : SV_Target1;
    // RT2: World Position (RGB) + Roughness (A) - linear, high precision
    float4 position_roughness : SV_Target2;
};

[shader("fragment")]
FsOutput fs_main(VsOutput input) {
    FsOutput output;

    float3 albedo = input.base_color.rgb;
    float metallic = input.metallic;
    float roughness = max(input.roughness, 0.04);
    float3 normal = normalize(input.world_normal);

    // RT0: Albedo
    output.albedo = float4(albedo, 1.0);

    // RT1: Normal (encoded to [0,1] range) + Metallic
    float3 encoded_normal = normal * 0.5 + 0.5;
    output.normal_metallic = float4(encoded_normal, metallic);

    // RT2: World Position + Roughness
    output.position_roughness = float4(input.world_position, roughness);

    return output;
}
