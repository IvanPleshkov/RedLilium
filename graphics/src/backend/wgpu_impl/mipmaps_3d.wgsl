@group(0) @binding(0) var source: texture_3d<f32>;

struct VertexOutput {
    @builtin(position) position: vec4f,
    @location(0) @interpolate(flat) slice: u32,
}

@vertex fn vs(@builtin(vertex_index) index: u32, @builtin(instance_index) slice: u32) -> VertexOutput {
    let positions = array(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
    return VertexOutput(vec4f(positions[index], 0.0, 1.0), slice);
}

// Volume-weighted box reduction includes edge voxels on all three axes.
// Explicit loads support non-filterable float formats; sRGB views perform
// linear-light decoding and encoding, just as for 2D mip reduction.
@fragment fn fs(input: VertexOutput) -> @location(0) vec4f {
    let size = textureDimensions(source);
    let destination_size = max(size / 2u, vec3u(1u));
    let scale = vec3f(size) / vec3f(destination_size);
    let lo = vec3f(floor(input.position.xy), f32(input.slice)) * scale;
    let hi = min(lo + scale, vec3f(size));
    var sum = vec4f(0.0);
    for (var z = i32(floor(lo.z)); z < i32(ceil(hi.z)); z++) {
        for (var y = i32(floor(lo.y)); y < i32(ceil(hi.y)); y++) {
            for (var x = i32(floor(lo.x)); x < i32(ceil(hi.x)); x++) {
                let p = vec3f(f32(x), f32(y), f32(z));
                let overlap = max(vec3f(0.0), min(hi, p + vec3f(1.0)) - max(lo, p));
                sum += textureLoad(source, vec3i(x, y, z), 0) * overlap.x * overlap.y * overlap.z;
            }
        }
    }
    return sum / (scale.x * scale.y * scale.z);
}
