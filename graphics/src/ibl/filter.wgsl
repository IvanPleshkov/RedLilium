// GGX split-sum prefilter, N = V = R, perceptual roughness -> alpha = r².
// Source LOD follows the sample PDF (filtered importance sampling).
// See xtask/src/ibl.rs and https://google.github.io/filament/main/filament.html
@group(0) @binding(0) var environment: texture_cube<f32>;
@group(0) @binding(1) var env_sampler: sampler;
struct Parameters { size: u32, samples: u32, roughness: f32, padding: u32 }
@group(0) @binding(2) var<uniform> params: Parameters;
const PI: f32 = 3.141592653589793;
struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) @interpolate(flat) face: u32,
}
@vertex fn vs(@builtin(vertex_index) index: u32, @builtin(instance_index) face: u32) -> VertexOutput {
    let positions = array(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
    return VertexOutput(vec4f(positions[index], 0.0, 1.0), face);
}
fn direction(input: VertexOutput) -> vec3f {
    let uv = input.position.xy / f32(params.size) * 2.0 - 1.0;
    var ray: vec3f;
    switch input.face {
        case 0u: { ray = vec3f(1.0, -uv.y, -uv.x); }
        case 1u: { ray = vec3f(-1.0, -uv.y, uv.x); }
        case 2u: { ray = vec3f(uv.x, 1.0, uv.y); }
        case 3u: { ray = vec3f(uv.x, -1.0, -uv.y); }
        case 4u: { ray = vec3f(uv.x, -uv.y, 1.0); }
        default: { ray = vec3f(-uv.x, -uv.y, -1.0); }
    }
    return normalize(ray);
}
fn frame(n: vec3f) -> mat3x3f {
    let up = select(vec3f(1.0, 0.0, 0.0), vec3f(0.0, 0.0, 1.0), abs(n.z) < 0.999);
    let t = normalize(cross(n, up));
    return mat3x3f(t, cross(n, t), n);
}
fn hammersley(i: u32) -> vec2f {
    return vec2f(f32(i) / f32(params.samples), f32(reverseBits(i)) * 2.3283064365386963e-10);
}
fn source_lod(pdf: f32) -> f32 {
    let size = f32(textureDimensions(environment).x);
    let texel_angle = 4.0 * PI / (6.0 * size * size);
    let sample_angle = 1.0 / (f32(params.samples) * max(pdf, 1e-6));
    return clamp(0.5 * log2(sample_angle / texel_angle), 0.0, f32(textureNumLevels(environment) - 1u));
}
fn radiance(dir: vec3f, lod: f32) -> vec3f {
    return textureSampleLevel(environment, env_sampler, dir, lod).rgb;
}
@fragment fn specular(input: VertexOutput) -> @location(0) vec4f {
    let n = direction(input);
    if params.roughness == 0.0 {
        let lod = max(log2(f32(textureDimensions(environment).x) / f32(params.size)), 0.0);
        return vec4f(radiance(n, lod), 1.0);
    }
    let basis = frame(n);
    let a = params.roughness * params.roughness;
    let a2 = a * a;
    var sum = vec3f(0.0);
    var weight = 0.0;
    for (var i = 0u; i < params.samples; i++) {
        let xi = hammersley(i);
        let phi = 2.0 * PI * xi.x;
        let cos_theta = sqrt((1.0 - xi.y) / (1.0 + (a2 - 1.0) * xi.y));
        let sin_theta = sqrt(max(1.0 - cos_theta * cos_theta, 0.0));
        let h = normalize(basis * vec3f(cos(phi) * sin_theta, sin(phi) * sin_theta, cos_theta));
        let l = normalize(2.0 * dot(n, h) * h - n);
        let no_l = max(dot(n, l), 0.0);
        if no_l > 0.0 {
            let no_h = max(dot(n, h), 0.0);
            let denom = no_h * no_h * (a2 - 1.0) + 1.0;
            let pdf = a2 / (4.0 * PI * denom * denom);
            sum += radiance(l, source_lod(pdf)) * no_l;
            weight += no_l;
        }
    }
    return vec4f(sum / max(weight, 1e-6), 1.0);
}
@fragment fn diffuse(input: VertexOutput) -> @location(0) vec4f {
    let basis = frame(direction(input));
    var sum = vec3f(0.0);
    for (var i = 0u; i < params.samples; i++) {
        let xi = hammersley(i);
        let phi = 2.0 * PI * xi.x;
        let radius = sqrt(xi.y);
        let cos_theta = sqrt(1.0 - xi.y);
        let l = basis * vec3f(cos(phi) * radius, sin(phi) * radius, cos_theta);
        // Cosine PDF cancels N.L / PI: result is irradiance / PI.
        sum += radiance(l, source_lod(cos_theta / PI));
    }
    return vec4f(sum / f32(params.samples), 1.0);
}
