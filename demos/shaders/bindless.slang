// Bindless heap demo (#117).
//
// Set 0 is ordinary material data (scene uniforms + per-instance data); set 1
// is the device-owned bindless heap: a runtime-sized sampled-texture array
// and a sampler array, indexed per instance with NonUniformResourceIndex.
// There is no per-texture descriptor anywhere — every instance picks its
// texture by a plain integer.

// =============================================================================
// Set 0: material data
// =============================================================================

struct InstanceData {
    // xy = NDC center, z = half-size (aspect-corrected on the CPU), w unused.
    float4 center_size;
    uint texture_index;
    uint sampler_index;
    uint _pad0;
    uint _pad1;
};

[[vk::binding(0, 0)]]
cbuffer SceneUniforms {
    // x = time in seconds (drives a gentle pulse so frames differ),
    // y = aspect ratio (width / height), z-w unused.
    float4 time_aspect;
};

[[vk::binding(1, 0)]]
StructuredBuffer<InstanceData> instances;

// =============================================================================
// Set 1: the bindless heap (matches GraphicsDevice::bindless_heap_layout)
// =============================================================================

[[vk::binding(0, 1)]]
Texture2D bindless_textures[];

[[vk::binding(1, 1)]]
SamplerState bindless_samplers[];

// =============================================================================
// Stages
// =============================================================================

struct VsInput {
    [[vk::location(0)]] float3 position : POSITION;
    [[vk::location(1)]] float2 uv : TEXCOORD0;
    uint instance_id : SV_InstanceID;
};

struct VsOutput {
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
    nointerpolation uint texture_index : TEXCOORD1;
    nointerpolation uint sampler_index : TEXCOORD2;
};

[shader("vertex")]
VsOutput vs_main(VsInput input) {
    InstanceData inst = instances[input.instance_id];

    // A subtle per-instance pulse, so motion proves the frame loop is live.
    float pulse = 1.0 + 0.06 * sin(time_aspect.x * 2.0 + float(input.instance_id) * 0.7);
    // center_size.z is the NDC half-height; x divides by the aspect ratio so
    // the quads stay square on screen.
    float2 half_size = float2(inst.center_size.z / time_aspect.y, inst.center_size.z) * pulse;
    float2 ndc = inst.center_size.xy + input.position.xy * half_size;

    VsOutput output;
    output.position = float4(ndc, 0.0, 1.0);
    output.uv = input.uv;
    output.texture_index = inst.texture_index;
    output.sampler_index = inst.sampler_index;
    return output;
}

[shader("fragment")]
float4 fs_main(VsOutput input) : SV_Target {
    Texture2D tex = bindless_textures[NonUniformResourceIndex(input.texture_index)];
    SamplerState smp = bindless_samplers[NonUniformResourceIndex(input.sampler_index)];
    // Linear output — the swapchain surface is sRGB and encodes on write.
    return float4(tex.Sample(smp, input.uv).rgb, 1.0);
}
