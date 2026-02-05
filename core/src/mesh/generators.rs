//! Mesh generators for common shapes.
//!
//! These generators produce [`CpuMesh`] values that can be uploaded
//! to the GPU via `GraphicsDevice::create_mesh_from_cpu`.

use std::f32::consts::PI;

use super::data::CpuMesh;
use super::layout::VertexLayout;

/// Internal vertex type for sphere generation (position + normal + uv).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct PnuVertex {
    position: [f32; 3],
    normal: [f32; 3],
    uv: [f32; 2],
}

/// Internal vertex type for quad generation (position + uv).
#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct PuVertex {
    position: [f32; 3],
    uv: [f32; 2],
}

/// Generate a UV sphere mesh.
///
/// Creates a sphere with the given radius, number of longitudinal segments,
/// and number of latitudinal rings. The mesh uses the `position_normal_uv`
/// layout (32 bytes per vertex) with u32 indices.
///
/// # Arguments
///
/// * `radius` - Sphere radius
/// * `segments` - Number of longitudinal segments (around the equator)
/// * `rings` - Number of latitudinal rings (from pole to pole)
pub fn generate_sphere(radius: f32, segments: u32, rings: u32) -> CpuMesh {
    let mut vertices = Vec::new();
    let mut indices = Vec::new();

    for ring in 0..=rings {
        let theta = ring as f32 * PI / rings as f32;
        let sin_theta = theta.sin();
        let cos_theta = theta.cos();

        for segment in 0..=segments {
            let phi = segment as f32 * 2.0 * PI / segments as f32;
            let sin_phi = phi.sin();
            let cos_phi = phi.cos();

            let x = sin_theta * cos_phi;
            let y = cos_theta;
            let z = sin_theta * sin_phi;

            vertices.push(PnuVertex {
                position: [x * radius, y * radius, z * radius],
                normal: [x, y, z],
                uv: [segment as f32 / segments as f32, ring as f32 / rings as f32],
            });
        }
    }

    for ring in 0..rings {
        for segment in 0..segments {
            let current = ring * (segments + 1) + segment;
            let next = current + segments + 1;

            indices.push(current);
            indices.push(next);
            indices.push(current + 1);

            indices.push(current + 1);
            indices.push(next);
            indices.push(next + 1);
        }
    }

    let vertex_bytes = bytemuck::cast_slice(&vertices).to_vec();

    CpuMesh::new(VertexLayout::position_normal_uv())
        .with_vertex_data(0, vertex_bytes)
        .with_indices_u32(&indices)
        .with_label("sphere")
}

/// Generate a quad mesh on the XY plane.
///
/// Creates a quad centered at the origin with the given half-width and
/// half-height. The mesh uses a position + texcoord layout (20 bytes per
/// vertex) with u32 indices.
///
/// UV coordinates go from (0,0) at top-left to (1,1) at bottom-right.
///
/// # Arguments
///
/// * `half_width` - Half the width of the quad along the X axis
/// * `half_height` - Half the height of the quad along the Y axis
pub fn generate_quad(half_width: f32, half_height: f32) -> CpuMesh {
    let vertices = [
        PuVertex {
            position: [-half_width, -half_height, 0.0],
            uv: [0.0, 1.0],
        },
        PuVertex {
            position: [half_width, -half_height, 0.0],
            uv: [1.0, 1.0],
        },
        PuVertex {
            position: [half_width, half_height, 0.0],
            uv: [1.0, 0.0],
        },
        PuVertex {
            position: [-half_width, half_height, 0.0],
            uv: [0.0, 0.0],
        },
    ];

    let indices: [u32; 6] = [0, 1, 2, 2, 3, 0];
    let vertex_bytes = bytemuck::cast_slice(&vertices).to_vec();

    CpuMesh::new(VertexLayout::position_uv())
        .with_vertex_data(0, vertex_bytes)
        .with_indices_u32(&indices)
        .with_label("quad")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_sphere() {
        let mesh = generate_sphere(1.0, 8, 4);
        assert!(mesh.vertex_count() > 0);
        assert!(mesh.is_indexed());
        assert!(mesh.index_count() > 0);
        // (rings+1) * (segments+1) = 5 * 9 = 45 vertices
        assert_eq!(mesh.vertex_count(), 45);
        // rings * segments * 6 = 4 * 8 * 6 = 192 indices
        assert_eq!(mesh.index_count(), 192);
    }

    #[test]
    fn test_generate_quad() {
        let mesh = generate_quad(0.5, 0.5);
        assert_eq!(mesh.vertex_count(), 4);
        assert!(mesh.is_indexed());
        assert_eq!(mesh.index_count(), 6);
    }

    #[test]
    fn test_sphere_vertex_data_size() {
        let mesh = generate_sphere(1.0, 4, 2);
        let data = mesh.vertex_buffer_data(0).unwrap();
        // (2+1) * (4+1) = 15 vertices * 32 bytes = 480
        assert_eq!(data.len(), 15 * 32);
    }

    #[test]
    fn test_quad_vertex_data_size() {
        let mesh = generate_quad(1.0, 1.0);
        let data = mesh.vertex_buffer_data(0).unwrap();
        // 4 vertices * 20 bytes = 80
        assert_eq!(data.len(), 4 * 20);
    }
}
