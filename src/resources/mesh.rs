//! Mesh data structures and generation

use crate::backend::types::Vertex;
use bytemuck::{Pod, Zeroable};
use glam::{Vec2, Vec3, Vec4};

/// A mesh with vertex and index data
#[derive(Debug, Clone)]
pub struct Mesh {
    pub vertices: Vec<Vertex>,
    pub indices: Vec<u32>,
    pub name: String,
}

impl Mesh {
    pub fn new(name: &str) -> Self {
        Self {
            vertices: Vec::new(),
            indices: Vec::new(),
            name: name.to_string(),
        }
    }

    /// Calculate vertex count
    pub fn vertex_count(&self) -> usize {
        self.vertices.len()
    }

    /// Calculate index count
    pub fn index_count(&self) -> usize {
        self.indices.len()
    }

    /// Calculate triangle count
    pub fn triangle_count(&self) -> usize {
        self.indices.len() / 3
    }

    /// Get vertex data as bytes
    pub fn vertex_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.vertices)
    }

    /// Get index data as bytes
    pub fn index_bytes(&self) -> &[u8] {
        bytemuck::cast_slice(&self.indices)
    }

    /// Create a unit cube centered at origin
    pub fn cube() -> Self {
        let mut mesh = Mesh::new("cube");

        // Define cube vertices with normals and UVs
        let positions = [
            // Front face
            (Vec3::new(-0.5, -0.5, 0.5), Vec3::Z, Vec2::new(0.0, 1.0)),
            (Vec3::new(0.5, -0.5, 0.5), Vec3::Z, Vec2::new(1.0, 1.0)),
            (Vec3::new(0.5, 0.5, 0.5), Vec3::Z, Vec2::new(1.0, 0.0)),
            (Vec3::new(-0.5, 0.5, 0.5), Vec3::Z, Vec2::new(0.0, 0.0)),
            // Back face
            (Vec3::new(0.5, -0.5, -0.5), -Vec3::Z, Vec2::new(0.0, 1.0)),
            (Vec3::new(-0.5, -0.5, -0.5), -Vec3::Z, Vec2::new(1.0, 1.0)),
            (Vec3::new(-0.5, 0.5, -0.5), -Vec3::Z, Vec2::new(1.0, 0.0)),
            (Vec3::new(0.5, 0.5, -0.5), -Vec3::Z, Vec2::new(0.0, 0.0)),
            // Right face
            (Vec3::new(0.5, -0.5, 0.5), Vec3::X, Vec2::new(0.0, 1.0)),
            (Vec3::new(0.5, -0.5, -0.5), Vec3::X, Vec2::new(1.0, 1.0)),
            (Vec3::new(0.5, 0.5, -0.5), Vec3::X, Vec2::new(1.0, 0.0)),
            (Vec3::new(0.5, 0.5, 0.5), Vec3::X, Vec2::new(0.0, 0.0)),
            // Left face
            (Vec3::new(-0.5, -0.5, -0.5), -Vec3::X, Vec2::new(0.0, 1.0)),
            (Vec3::new(-0.5, -0.5, 0.5), -Vec3::X, Vec2::new(1.0, 1.0)),
            (Vec3::new(-0.5, 0.5, 0.5), -Vec3::X, Vec2::new(1.0, 0.0)),
            (Vec3::new(-0.5, 0.5, -0.5), -Vec3::X, Vec2::new(0.0, 0.0)),
            // Top face
            (Vec3::new(-0.5, 0.5, 0.5), Vec3::Y, Vec2::new(0.0, 1.0)),
            (Vec3::new(0.5, 0.5, 0.5), Vec3::Y, Vec2::new(1.0, 1.0)),
            (Vec3::new(0.5, 0.5, -0.5), Vec3::Y, Vec2::new(1.0, 0.0)),
            (Vec3::new(-0.5, 0.5, -0.5), Vec3::Y, Vec2::new(0.0, 0.0)),
            // Bottom face
            (Vec3::new(-0.5, -0.5, -0.5), -Vec3::Y, Vec2::new(0.0, 1.0)),
            (Vec3::new(0.5, -0.5, -0.5), -Vec3::Y, Vec2::new(1.0, 1.0)),
            (Vec3::new(0.5, -0.5, 0.5), -Vec3::Y, Vec2::new(1.0, 0.0)),
            (Vec3::new(-0.5, -0.5, 0.5), -Vec3::Y, Vec2::new(0.0, 0.0)),
        ];

        for (position, normal, uv) in positions {
            // Calculate tangent (pointing along U direction)
            let tangent = if normal.abs().y > 0.9 {
                Vec4::new(1.0, 0.0, 0.0, 1.0)
            } else {
                let right = Vec3::Y.cross(normal).normalize();
                right.extend(1.0)
            };

            mesh.vertices.push(Vertex {
                position,
                normal,
                uv,
                tangent,
            });
        }

        // Define indices (two triangles per face)
        for face in 0..6 {
            let base = face * 4;
            mesh.indices.extend_from_slice(&[
                base,
                base + 1,
                base + 2,
                base,
                base + 2,
                base + 3,
            ]);
        }

        mesh
    }

    /// Create a UV sphere
    pub fn sphere(segments: u32, rings: u32) -> Self {
        let mut mesh = Mesh::new("sphere");

        let segment_angle = 2.0 * std::f32::consts::PI / segments as f32;
        let ring_angle = std::f32::consts::PI / rings as f32;

        // Generate vertices
        for ring in 0..=rings {
            let phi = ring as f32 * ring_angle;
            let y = phi.cos();
            let ring_radius = phi.sin();

            for segment in 0..=segments {
                let theta = segment as f32 * segment_angle;
                let x = ring_radius * theta.cos();
                let z = ring_radius * theta.sin();

                let position = Vec3::new(x * 0.5, y * 0.5, z * 0.5);
                let normal = Vec3::new(x, y, z).normalize();
                let uv = Vec2::new(
                    segment as f32 / segments as f32,
                    ring as f32 / rings as f32,
                );

                // Tangent along theta direction
                let tangent = Vec3::new(-theta.sin(), 0.0, theta.cos()).normalize();

                mesh.vertices.push(Vertex {
                    position,
                    normal,
                    uv,
                    tangent: tangent.extend(1.0),
                });
            }
        }

        // Generate indices
        for ring in 0..rings {
            for segment in 0..segments {
                let current = ring * (segments + 1) + segment;
                let next = current + segments + 1;

                mesh.indices.extend_from_slice(&[
                    current,
                    next,
                    current + 1,
                    current + 1,
                    next,
                    next + 1,
                ]);
            }
        }

        mesh
    }

    /// Create a plane on the XZ axis
    pub fn plane(width: f32, depth: f32, subdivisions: u32) -> Self {
        let mut mesh = Mesh::new("plane");

        let half_width = width / 2.0;
        let half_depth = depth / 2.0;
        let step_x = width / subdivisions as f32;
        let step_z = depth / subdivisions as f32;

        // Generate vertices
        for z in 0..=subdivisions {
            for x in 0..=subdivisions {
                let px = -half_width + x as f32 * step_x;
                let pz = -half_depth + z as f32 * step_z;

                mesh.vertices.push(Vertex {
                    position: Vec3::new(px, 0.0, pz),
                    normal: Vec3::Y,
                    uv: Vec2::new(x as f32 / subdivisions as f32, z as f32 / subdivisions as f32),
                    tangent: Vec4::new(1.0, 0.0, 0.0, 1.0),
                });
            }
        }

        // Generate indices
        for z in 0..subdivisions {
            for x in 0..subdivisions {
                let current = z * (subdivisions + 1) + x;
                let next = current + subdivisions + 1;

                mesh.indices.extend_from_slice(&[
                    current,
                    next,
                    current + 1,
                    current + 1,
                    next,
                    next + 1,
                ]);
            }
        }

        mesh
    }

    /// Create a cylinder
    pub fn cylinder(radius: f32, height: f32, segments: u32) -> Self {
        let mut mesh = Mesh::new("cylinder");

        let half_height = height / 2.0;
        let angle_step = 2.0 * std::f32::consts::PI / segments as f32;

        // Side vertices
        for i in 0..=segments {
            let angle = i as f32 * angle_step;
            let x = angle.cos() * radius;
            let z = angle.sin() * radius;
            let normal = Vec3::new(angle.cos(), 0.0, angle.sin());
            let u = i as f32 / segments as f32;

            // Bottom vertex
            mesh.vertices.push(Vertex {
                position: Vec3::new(x, -half_height, z),
                normal,
                uv: Vec2::new(u, 1.0),
                tangent: Vec4::new(-angle.sin(), 0.0, angle.cos(), 1.0),
            });

            // Top vertex
            mesh.vertices.push(Vertex {
                position: Vec3::new(x, half_height, z),
                normal,
                uv: Vec2::new(u, 0.0),
                tangent: Vec4::new(-angle.sin(), 0.0, angle.cos(), 1.0),
            });
        }

        // Side indices
        for i in 0..segments {
            let base = i * 2;
            mesh.indices.extend_from_slice(&[
                base,
                base + 2,
                base + 1,
                base + 1,
                base + 2,
                base + 3,
            ]);
        }

        // Top cap center
        let top_center_idx = mesh.vertices.len() as u32;
        mesh.vertices.push(Vertex {
            position: Vec3::new(0.0, half_height, 0.0),
            normal: Vec3::Y,
            uv: Vec2::new(0.5, 0.5),
            tangent: Vec4::new(1.0, 0.0, 0.0, 1.0),
        });

        // Bottom cap center
        let bottom_center_idx = mesh.vertices.len() as u32;
        mesh.vertices.push(Vertex {
            position: Vec3::new(0.0, -half_height, 0.0),
            normal: -Vec3::Y,
            uv: Vec2::new(0.5, 0.5),
            tangent: Vec4::new(1.0, 0.0, 0.0, 1.0),
        });

        // Cap vertices and indices
        for i in 0..=segments {
            let angle = i as f32 * angle_step;
            let x = angle.cos() * radius;
            let z = angle.sin() * radius;
            let u = 0.5 + angle.cos() * 0.5;
            let v = 0.5 + angle.sin() * 0.5;

            // Top cap vertex
            let top_idx = mesh.vertices.len() as u32;
            mesh.vertices.push(Vertex {
                position: Vec3::new(x, half_height, z),
                normal: Vec3::Y,
                uv: Vec2::new(u, v),
                tangent: Vec4::new(1.0, 0.0, 0.0, 1.0),
            });

            // Bottom cap vertex
            let bottom_idx = mesh.vertices.len() as u32;
            mesh.vertices.push(Vertex {
                position: Vec3::new(x, -half_height, z),
                normal: -Vec3::Y,
                uv: Vec2::new(u, v),
                tangent: Vec4::new(1.0, 0.0, 0.0, 1.0),
            });

            if i > 0 {
                // Top cap triangle
                mesh.indices.extend_from_slice(&[top_center_idx, top_idx - 2, top_idx]);
                // Bottom cap triangle
                mesh.indices.extend_from_slice(&[bottom_center_idx, bottom_idx, bottom_idx - 2]);
            }
        }

        mesh
    }
}
