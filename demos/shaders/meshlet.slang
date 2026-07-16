// Meshlet rendering via task + mesh shaders (#111).
//
// The task stage culls whole meshlets before rasterization — frustum test on
// the meshlet's bounding sphere plus a backface-cone test (a meshlet whose
// every triangle faces away from the camera is dropped) — and forwards the
// survivors' indices to the mesh stage through the payload. The mesh stage
// fetches the meshlet's vertices/triangles from storage buffers (there is no
// vertex input at all) and emits them.
//
// Culling uses the *cull* camera from the uniforms, which the demo can freeze
// independently of the render camera to make the culling visible.

// Threads per task workgroup; one thread tests one meshlet.
static const uint TASK_GROUP_SIZE = 32;
// Mesh-stage output limits. The demo's generator packs 5x5-vertex sphere
// patches (32 triangles), comfortably under both.
static const uint MAX_VERTICES = 64;
static const uint MAX_TRIANGLES = 124;

// =============================================================================
// Shared Types
// =============================================================================

struct MeshletVertex {
    // xyz = object-space position (w unused).
    float4 position;
    // xyz = object-space normal (w unused).
    float4 normal;
};

struct Meshlet {
    // Offsets/counts into `vertex_indices` / `triangles`.
    uint vertex_offset;
    uint vertex_count;
    uint triangle_offset;
    uint triangle_count;
    // Object-space bounding sphere: xyz = center, w = radius.
    float4 sphere;
    // Object-space backface cone: xyz = axis, w = cutoff (meshoptimizer
    // convention: cull when dot(center - cam, axis) >= cutoff * |center - cam|
    // + radius). A cutoff of 1 disables the test for that meshlet.
    float4 cone;
};

struct InstanceData {
    // xyz = world translation, w = uniform scale.
    float4 position_scale;
    // rgb tint (a unused).
    float4 color;
};

// =============================================================================
// Bindings
// =============================================================================

[[vk::binding(0, 0)]]
cbuffer SceneUniforms {
    // Render camera (always live).
    column_major float4x4 view_proj;
    // Cull camera: world-space frustum planes (xyz = normal, w = distance;
    // inside when dot(n, p) + w >= 0) and position. Frozen on demand.
    float4 cull_planes[6];
    float4 cull_camera_pos;
    // -Z-ish directional light, normalized, pointing *from* the light.
    float4 light_dir;
    uint meshlet_count;
    uint instance_count;
    uint _pad0;
    uint _pad1;
};

[[vk::binding(1, 0)]]
StructuredBuffer<MeshletVertex> vertices;

// Per-meshlet vertex slots -> global vertex indices.
[[vk::binding(2, 0)]]
StructuredBuffer<uint> vertex_indices;

// One packed triangle per uint: local vertex indices i0 | i1<<8 | i2<<16.
[[vk::binding(3, 0)]]
StructuredBuffer<uint> triangles;

[[vk::binding(4, 0)]]
StructuredBuffer<Meshlet> meshlets;

[[vk::binding(5, 0)]]
StructuredBuffer<InstanceData> instances;

// =============================================================================
// Task (amplification) stage: one thread culls one meshlet
// =============================================================================

struct Payload {
    uint instance_index;
    uint meshlet_indices[TASK_GROUP_SIZE];
};

groupshared Payload s_payload;
groupshared uint s_survivor_count;

bool meshlet_visible(Meshlet m, InstanceData inst) {
    // Object -> world for a translate+uniform-scale instance.
    float scale = inst.position_scale.w;
    float3 center = m.sphere.xyz * scale + inst.position_scale.xyz;
    float radius = m.sphere.w * scale;

    // Frustum: outside any plane by more than the radius -> culled.
    for (uint i = 0; i < 6; i++) {
        if (dot(cull_planes[i].xyz, center) + cull_planes[i].w < -radius) {
            return false;
        }
    }

    // Backface cone (axis is rotation-free here — instances only translate
    // and scale): every triangle faces away -> culled.
    float3 to_meshlet = center - cull_camera_pos.xyz;
    if (dot(to_meshlet, m.cone.xyz) >= m.cone.w * length(to_meshlet) + radius) {
        return false;
    }

    return true;
}

[numthreads(TASK_GROUP_SIZE, 1, 1)]
[shader("amplification")]
void ts_main(uint group_thread : SV_GroupIndex, uint3 group_id : SV_GroupID) {
    if (group_thread == 0) {
        s_survivor_count = 0;
        s_payload.instance_index = group_id.y;
    }
    GroupMemoryBarrierWithGroupSync();

    uint meshlet_index = group_id.x * TASK_GROUP_SIZE + group_thread;
    if (meshlet_index < meshlet_count) {
        InstanceData inst = instances[group_id.y];
        if (meshlet_visible(meshlets[meshlet_index], inst)) {
            uint slot;
            InterlockedAdd(s_survivor_count, 1, slot);
            s_payload.meshlet_indices[slot] = meshlet_index;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    // Uniform after the barrier: one mesh workgroup per surviving meshlet.
    DispatchMesh(s_survivor_count, 1, 1, s_payload);
}

// =============================================================================
// Mesh stage: one workgroup emits one meshlet
// =============================================================================

struct MeshVertexOut {
    float4 position : SV_Position;
    float3 world_normal : TEXCOORD0;
    float3 color : TEXCOORD1;
};

// Deterministic bright-ish color from the meshlet index, so the partition is
// visible.
float3 meshlet_debug_color(uint index) {
    uint h = index * 2654435761u;
    float3 c = float3(
        float((h >> 0) & 0xFF),
        float((h >> 8) & 0xFF),
        float((h >> 16) & 0xFF)) / 255.0;
    return 0.25 + 0.75 * c;
}

[outputtopology("triangle")]
[numthreads(MAX_VERTICES, 1, 1)]
[shader("mesh")]
void ms_main(
    uint group_thread : SV_GroupIndex,
    uint3 group_id : SV_GroupID,
    in payload Payload payload,
    out vertices MeshVertexOut out_vertices[MAX_VERTICES],
    out indices uint3 out_triangles[MAX_TRIANGLES])
{
    uint meshlet_index = payload.meshlet_indices[group_id.x];
    Meshlet m = meshlets[meshlet_index];
    InstanceData inst = instances[payload.instance_index];

    SetMeshOutputCounts(m.vertex_count, m.triangle_count);

    if (group_thread < m.vertex_count) {
        MeshletVertex v = vertices[vertex_indices[m.vertex_offset + group_thread]];
        float3 world_pos =
            v.position.xyz * inst.position_scale.w + inst.position_scale.xyz;

        MeshVertexOut vout;
        vout.position = mul(view_proj, float4(world_pos, 1.0));
        vout.world_normal = v.normal.xyz;
        vout.color = meshlet_debug_color(meshlet_index) * inst.color.rgb;
        out_vertices[group_thread] = vout;
    }

    // MAX_TRIANGLES can exceed the workgroup size, so stride the loop.
    for (uint t = group_thread; t < m.triangle_count; t += MAX_VERTICES) {
        uint packed = triangles[m.triangle_offset + t];
        out_triangles[t] = uint3(packed & 0xFF, (packed >> 8) & 0xFF, (packed >> 16) & 0xFF);
    }
}

// =============================================================================
// Fragment stage
// =============================================================================

[shader("fragment")]
float4 fs_main(MeshVertexOut input) : SV_Target {
    float3 n = normalize(input.world_normal);
    float ndotl = max(dot(n, -light_dir.xyz), 0.0);
    // Linear output — the swapchain surface is sRGB and encodes on write.
    float3 color = input.color * (0.15 + 0.85 * ndotl);
    return float4(color, 1.0);
}
