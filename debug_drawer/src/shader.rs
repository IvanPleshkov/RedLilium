/// Debug draw shader source (GLSL).
///
/// Simple unlit line shader: transforms position by a view-projection matrix
/// and passes through vertex color.
pub const DEBUG_DRAW_SHADER_SOURCE: &str = r#"#version 450

#ifdef VERTEX

layout(set = 0, binding = 0) uniform DebugUniforms {
    mat4 view_proj;
};

layout(location = 0) in vec3 position;
layout(location = 5) in vec4 color;

layout(location = 0) out vec4 v_color;

void main() {
    gl_Position = view_proj * vec4(position, 1.0);
    v_color = color;
}

#endif

#ifdef FRAGMENT

layout(location = 0) in vec4 v_color;
layout(location = 0) out vec4 out_color;

void main() {
    out_color = v_color;
}

#endif
"#;
