// Matches core::texture::mipmaps: KTX/Vulkan +X,-X,+Y,-Y,+Z,-Z.
// Each six-layer group is independent. No hardware filtering is required.
@group(0) @binding(0) var source: texture_2d_array<f32>;
struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) @interpolate(flat) layer: u32,
}
@vertex fn vs(@builtin(vertex_index) vertex: u32, @builtin(instance_index) layer: u32) -> VertexOutput {
    let positions = array(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
    return VertexOutput(vec4f(positions[vertex], 0.0, 1.0), layer);
}
fn direction(face: u32, uv: vec2i, size: i32) -> vec3i {
    switch face {
        case 0u: { return vec3i(size, -uv.y, -uv.x); }
        case 1u: { return vec3i(-size, -uv.y, uv.x); }
        case 2u: { return vec3i(uv.x, size, uv.y); }
        case 3u: { return vec3i(uv.x, -size, -uv.y); }
        case 4u: { return vec3i(uv.x, -uv.y, size); }
        default: { return vec3i(-uv.x, -uv.y, -size); }
    }
}
fn load_cube(layer: u32, pixel: vec2i, size: u32) -> vec4f {
    // Integer center rays keep exact boundary taps consistent with CPU import.
    let d = direction(layer % 6u, 2 * pixel + 1 - i32(size), i32(size));
    let a = abs(d);
    var face: u32; var st: vec2i; var major: i32;
    if a.x >= a.y && a.x >= a.z {
        major = a.x;
        if d.x > 0 { face = 0u; st = vec2i(-d.z, -d.y); }
        else { face = 1u; st = vec2i(d.z, -d.y); }
    } else if a.y >= a.z {
        major = a.y;
        if d.y > 0 { face = 2u; st = vec2i(d.x, d.z); }
        else { face = 3u; st = vec2i(d.x, -d.z); }
    } else {
        major = a.z;
        if d.z > 0 { face = 4u; st = vec2i(d.x, -d.y); }
        else { face = 5u; st = vec2i(-d.x, -d.y); }
    }
    let xy = min(vec2u(st + major) * size / u32(2 * major), vec2u(size - 1u));
    return textureLoad(source, vec2i(xy), i32(layer / 6u * 6u + face), 0);
}
@fragment fn fs(input: VertexOutput) -> @location(0) vec4f {
    let size = textureDimensions(source).x;
    let scale = f32(size) / f32(max(size / 2u, 1u));
    let center = (floor(input.position.xy) + 0.5) * scale;
    var sum = vec4f(0.0); var total = 0.0;
    for (var y = i32(floor(center.y - scale)); y < i32(ceil(center.y + scale)); y++) {
        for (var x = i32(floor(center.x - scale)); x < i32(ceil(center.x + scale)); x++) {
            let weights = max(vec2f(0.0), 1.0 - abs((vec2f(f32(x), f32(y)) + 0.5 - center) / scale));
            let weight = weights.x * weights.y;
            sum += load_cube(input.layer, vec2i(x, y), size) * weight;
            total += weight;
        }
    }
    return sum / total;
}
