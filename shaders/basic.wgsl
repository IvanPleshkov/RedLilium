// Basic shader for rendering meshes with simple lighting

struct CameraUniform {
    view: mat4x4<f32>,
    proj: mat4x4<f32>,
    view_proj: mat4x4<f32>,
    inv_view: mat4x4<f32>,
    inv_proj: mat4x4<f32>,
    position: vec4<f32>,
    near_far: vec4<f32>,
}

struct ObjectUniform {
    model: mat4x4<f32>,
    normal_matrix: mat4x4<f32>,
}

struct MaterialUniform {
    base_color: vec4<f32>,
    metallic: f32,
    roughness: f32,
    _padding: vec2<f32>,
}

@group(0) @binding(0) var<uniform> camera: CameraUniform;
@group(1) @binding(0) var<uniform> object: ObjectUniform;
@group(2) @binding(0) var<uniform> material: MaterialUniform;

struct VertexInput {
    @location(0) position: vec3<f32>,
    @location(1) normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
    @location(3) tangent: vec4<f32>,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) world_position: vec3<f32>,
    @location(1) world_normal: vec3<f32>,
    @location(2) uv: vec2<f32>,
}

@vertex
fn vs_main(in: VertexInput) -> VertexOutput {
    var out: VertexOutput;

    let world_pos = object.model * vec4<f32>(in.position, 1.0);
    out.world_position = world_pos.xyz;
    out.clip_position = camera.view_proj * world_pos;
    out.world_normal = normalize((object.normal_matrix * vec4<f32>(in.normal, 0.0)).xyz);
    out.uv = in.uv;

    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    // Simple directional light
    let light_dir = normalize(vec3<f32>(0.5, 1.0, 0.3));
    let light_color = vec3<f32>(1.0, 0.95, 0.9);
    let ambient = vec3<f32>(0.1, 0.1, 0.15);

    let normal = normalize(in.world_normal);
    let ndotl = max(dot(normal, light_dir), 0.0);

    // View direction for specular
    let view_dir = normalize(camera.position.xyz - in.world_position);
    let half_dir = normalize(light_dir + view_dir);
    let ndoth = max(dot(normal, half_dir), 0.0);

    // Simple Blinn-Phong specular
    let shininess = mix(8.0, 128.0, 1.0 - material.roughness);
    let specular = pow(ndoth, shininess) * (1.0 - material.roughness);

    // Metallic affects specular color
    let spec_color = mix(vec3<f32>(0.04), material.base_color.rgb, material.metallic);

    // Final color
    let diffuse = material.base_color.rgb * (1.0 - material.metallic);
    let color = ambient * material.base_color.rgb
              + diffuse * light_color * ndotl
              + spec_color * light_color * specular;

    return vec4<f32>(color, material.base_color.a);
}
