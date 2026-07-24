// SSAO denoise (#150) — a joint-bilateral blur over the raw horizon AO.
//
// The `ssao` pass trades undersampling banding for high-frequency noise (per-
// pixel kernel rotation). This 4x4 bilateral averages that noise away while
// stopping at geometry edges: neighbor weights fall off with world-space
// distance (G-buffer RT2), so occlusion does not bleed across silhouettes.
// Runs regardless of TAA — it is the baseline denoise; TAA (when present)
// only cleans the residual. Output feeds the resolve's ambient term.

struct VsOutput {
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

[shader("vertex")]
VsOutput vs_main(uint vertex_id : SV_VertexID) {
    VsOutput output;
    float2 positions[3] = {
        float2(-1.0, -1.0),
        float2(3.0, -1.0),
        float2(-1.0, 3.0)
    };
    float2 uvs[3] = {
        float2(0.0, 1.0),
        float2(2.0, 1.0),
        float2(0.0, -1.0)
    };
    output.position = float4(positions[vertex_id], 0.0, 1.0);
    output.uv = uvs[vertex_id];
    return output;
}

[[vk::binding(0, 0)]]
cbuffer SsaoBlurUniforms {
    // xy = texel size (1 / extent); z = world-space edge-stop sigma; w unused.
    float4 params;
};

[[vk::binding(1, 0)]]
Texture2D ssao_raw; // R8: raw horizon AO
[[vk::binding(2, 0)]]
Texture2D gbuffer_position_roughness; // RT2: .rgb = world position (edge stop)
[[vk::binding(3, 0)]]
SamplerState point_sampler;

// Symmetric 4x4 tap offsets (in texels), centered on the pixel.
static const float OFFSETS[4] = { -1.5, -0.5, 0.5, 1.5 };

[shader("fragment")]
float4 fs_main(VsOutput input) : SV_Target {
    float2 texel = params.xy;
    float3 center = gbuffer_position_roughness.Sample(point_sampler, input.uv).rgb;
    float sigma = max(params.z, 1e-4);
    float inv_two_sigma_sq = 1.0 / (2.0 * sigma * sigma);

    float sum = 0.0;
    float wsum = 0.0;
    [unroll]
    for (int i = 0; i < 4; i++) {
        [unroll]
        for (int j = 0; j < 4; j++) {
            float2 suv = input.uv + float2(OFFSETS[j], OFFSETS[i]) * texel;
            float a = ssao_raw.Sample(point_sampler, suv).r;
            float3 sp = gbuffer_position_roughness.Sample(point_sampler, suv).rgb;
            float3 dp = sp - center;
            float w = exp(-dot(dp, dp) * inv_two_sigma_sq);
            sum += a * w;
            wsum += w;
        }
    }
    return float4(sum / max(wsum, 1e-4), 0.0, 0.0, 1.0);
}
