@group(0) @binding(0) var source: texture_2d<f32>;

@vertex fn vs(@builtin(vertex_index) index: u32) -> @builtin(position) vec4f {
    let positions = array(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
    return vec4f(positions[index], 0.0, 1.0);
}

// Area-weighted box reduction includes the entire source for odd dimensions.
// textureLoad also works for non-filterable float formats. sRGB views decode
// before averaging; the sRGB render target encodes the result on store.
@fragment fn fs(@builtin(position) position: vec4f) -> @location(0) vec4f {
    let size = textureDimensions(source);
    let destination_size = max(size / 2u, vec2u(1u));
    let scale = vec2f(size) / vec2f(destination_size);
    let lo = floor(position.xy) * scale;
    let hi = lo + scale;
    var sum = vec4f(0.0);
    for (var y = i32(floor(lo.y)); y < i32(ceil(hi.y)); y++) {
        for (var x = i32(floor(lo.x)); x < i32(ceil(hi.x)); x++) {
            let overlap = max(vec2f(0.0), min(hi, vec2f(f32(x + 1), f32(y + 1)))
                - max(lo, vec2f(f32(x), f32(y))));
            sum += textureLoad(source, vec2i(x, y), 0) * overlap.x * overlap.y;
        }
    }
    return sum / (scale.x * scale.y);
}
