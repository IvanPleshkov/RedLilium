// Velocity completion (#149) — fills camera-only screen motion into the
// background of GBUFFER_VELOCITY.
//
// The G-buffer pass writes per-pixel velocity for geometry only (#147) and
// clears the rest to zero, so sky/background pixels read as "still" even while
// the camera pans. Motion blur's TileMax (and, in future, a unified TAA) need
// the sky to move with the camera, so this fullscreen pass runs right after
// the G-buffer: geometry fragments `discard` (LoadOp::Load keeps their
// object+camera velocity), background fragments are filled with the camera-only
// reprojection through the unjittered camera pair — the same helper TAA uses,
// so the two agree by construction.

import motion; // shared reprojection helpers (#149)

[[vk::binding(0, 0)]]
cbuffer VelocityCompleteUniforms {
    // Inverse of the current unjittered view-projection (background ray).
    column_major float4x4 inv_view_proj;
    // Previous frame's unjittered view-projection.
    column_major float4x4 prev_view_proj;
};

[[vk::binding(1, 0)]]
Texture2D gbuffer_albedo;
[[vk::binding(2, 0)]]
SamplerState nearest_sampler;

struct VsOutput {
    float4 position : SV_Position;
    float2 uv : TEXCOORD0;
};

[shader("vertex")]
VsOutput vs_main(uint vertex_id : SV_VertexID) {
    VsOutput output;

    // Fullscreen triangle.
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

[shader("fragment")]
float2 fs_main(VsOutput input) : SV_Target {
    // Geometry already carries object+camera velocity from the G-buffer; leave
    // it untouched (the pass loads, does not clear, the velocity target). The
    // background marker is the G-buffer albedo alpha (ADR-039).
    if (gbuffer_albedo.Sample(nearest_sampler, input.uv).a >= 1.0 / 128.0) {
        discard;
    }
    // Background: reproject the far-plane ray through the camera pair. Matches
    // the GBUFFER_VELOCITY convention (NDC, curr - prev, unjittered).
    float3 world = far_plane_world(inv_view_proj, input.uv);
    return camera_velocity_ndc(prev_view_proj, uv_to_ndc(input.uv), world);
}
