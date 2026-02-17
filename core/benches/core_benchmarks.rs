use criterion::{Criterion, black_box, criterion_group, criterion_main};

use redlilium_core::mesh::generators::{generate_quad, generate_sphere};
use redlilium_core::mesh::{
    VertexAttribute, VertexAttributeFormat, VertexAttributeSemantic, VertexBufferLayout,
    VertexLayout,
};

// ---------------------------------------------------------------------------
// Mesh generation
// ---------------------------------------------------------------------------

fn bench_generate_sphere_low(c: &mut Criterion) {
    c.bench_function("generate_sphere_16x8", |b| {
        b.iter(|| generate_sphere(black_box(1.0), black_box(16), black_box(8)));
    });
}

fn bench_generate_sphere_medium(c: &mut Criterion) {
    c.bench_function("generate_sphere_64x32", |b| {
        b.iter(|| generate_sphere(black_box(1.0), black_box(64), black_box(32)));
    });
}

fn bench_generate_sphere_high(c: &mut Criterion) {
    c.bench_function("generate_sphere_128x64", |b| {
        b.iter(|| generate_sphere(black_box(1.0), black_box(128), black_box(64)));
    });
}

fn bench_generate_quad(c: &mut Criterion) {
    c.bench_function("generate_quad", |b| {
        b.iter(|| generate_quad(black_box(0.5), black_box(0.5)));
    });
}

// ---------------------------------------------------------------------------
// Vertex layout construction
// ---------------------------------------------------------------------------

fn bench_vertex_layout_prebuilt(c: &mut Criterion) {
    c.bench_function("vertex_layout_position_normal_uv", |b| {
        b.iter(|| black_box(VertexLayout::position_normal_uv()));
    });
}

fn bench_vertex_layout_custom(c: &mut Criterion) {
    c.bench_function("vertex_layout_custom_build", |b| {
        b.iter(|| {
            black_box(
                VertexLayout::new()
                    .with_buffer(VertexBufferLayout::new(48))
                    .with_attribute(VertexAttribute::new(
                        VertexAttributeSemantic::Position,
                        VertexAttributeFormat::Float3,
                        0,
                        0,
                    ))
                    .with_attribute(VertexAttribute::new(
                        VertexAttributeSemantic::Normal,
                        VertexAttributeFormat::Float3,
                        12,
                        0,
                    ))
                    .with_attribute(VertexAttribute::new(
                        VertexAttributeSemantic::Tangent,
                        VertexAttributeFormat::Float4,
                        24,
                        0,
                    ))
                    .with_attribute(VertexAttribute::new(
                        VertexAttributeSemantic::TexCoord0,
                        VertexAttributeFormat::Float2,
                        40,
                        0,
                    )),
            )
        });
    });
}

fn bench_vertex_layout_get_attribute(c: &mut Criterion) {
    let layout = VertexLayout::position_normal_uv();
    c.bench_function("vertex_layout_get_attribute", |b| {
        b.iter(|| {
            black_box(layout.get_attribute(black_box(VertexAttributeSemantic::Normal)));
        });
    });
}

criterion_group!(
    benches,
    bench_generate_sphere_low,
    bench_generate_sphere_medium,
    bench_generate_sphere_high,
    bench_generate_quad,
    bench_vertex_layout_prebuilt,
    bench_vertex_layout_custom,
    bench_vertex_layout_get_attribute,
);
criterion_main!(benches);
