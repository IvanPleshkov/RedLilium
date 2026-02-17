use criterion::{Criterion, black_box, criterion_group, criterion_main};

use redlilium_graphics::{
    BufferDescriptor, BufferUsage, ComputePass, GraphicsInstance, GraphicsPass, RenderGraph,
    RenderGraphCompilationMode, TextureDescriptor, TextureFormat, TextureUsage, TransferPass,
};

// ---------------------------------------------------------------------------
// Render graph construction
// ---------------------------------------------------------------------------

fn bench_graph_build_small(c: &mut Criterion) {
    c.bench_function("render_graph_build_4_passes", |b| {
        b.iter(|| {
            let mut graph = RenderGraph::new();
            let shadow = graph.add_graphics_pass(GraphicsPass::new("shadow".into()));
            let geometry = graph.add_graphics_pass(GraphicsPass::new("geometry".into()));
            let lighting = graph.add_graphics_pass(GraphicsPass::new("lighting".into()));
            let post = graph.add_graphics_pass(GraphicsPass::new("post".into()));
            graph.add_dependency(geometry, shadow);
            graph.add_dependency(lighting, geometry);
            graph.add_dependency(post, lighting);
            black_box(&graph);
        });
    });
}

fn bench_graph_build_large(c: &mut Criterion) {
    c.bench_function("render_graph_build_32_passes_chain", |b| {
        b.iter(|| {
            let mut graph = RenderGraph::new();
            let mut prev = graph.add_graphics_pass(GraphicsPass::new("pass_0".into()));
            for i in 1..32 {
                let handle = graph.add_graphics_pass(GraphicsPass::new(format!("pass_{i}")));
                graph.add_dependency(handle, prev);
                prev = handle;
            }
            black_box(&graph);
        });
    });
}

fn bench_graph_build_mixed(c: &mut Criterion) {
    c.bench_function("render_graph_build_mixed_16_passes", |b| {
        b.iter(|| {
            let mut graph = RenderGraph::new();
            let upload = graph.add_transfer_pass(TransferPass::new("upload".into()));
            let mut prev_gfx = graph.add_graphics_pass(GraphicsPass::new("gbuffer".into()));
            graph.add_dependency(prev_gfx, upload);

            for i in 0..6 {
                let compute = graph.add_compute_pass(ComputePass::new(format!("compute_{i}")));
                graph.add_dependency(compute, prev_gfx);
                let gfx = graph.add_graphics_pass(GraphicsPass::new(format!("pass_{i}")));
                graph.add_dependency(gfx, compute);
                prev_gfx = gfx;
            }

            let download = graph.add_transfer_pass(TransferPass::new("download".into()));
            graph.add_dependency(download, prev_gfx);
            black_box(&graph);
        });
    });
}

// ---------------------------------------------------------------------------
// Render graph compilation
// ---------------------------------------------------------------------------

fn bench_graph_compile_small(c: &mut Criterion) {
    c.bench_function("render_graph_compile_4_passes", |b| {
        b.iter_with_setup(
            || {
                let mut graph = RenderGraph::new();
                let shadow = graph.add_graphics_pass(GraphicsPass::new("shadow".into()));
                let geometry = graph.add_graphics_pass(GraphicsPass::new("geometry".into()));
                let lighting = graph.add_graphics_pass(GraphicsPass::new("lighting".into()));
                let post = graph.add_graphics_pass(GraphicsPass::new("post".into()));
                graph.add_dependency(geometry, shadow);
                graph.add_dependency(lighting, geometry);
                graph.add_dependency(post, lighting);
                graph
            },
            |mut graph| {
                black_box(graph.compile(RenderGraphCompilationMode::Strict).unwrap());
            },
        );
    });
}

fn bench_graph_compile_large(c: &mut Criterion) {
    c.bench_function("render_graph_compile_32_passes_chain", |b| {
        b.iter_with_setup(
            || {
                let mut graph = RenderGraph::new();
                let mut prev = graph.add_graphics_pass(GraphicsPass::new("pass_0".into()));
                for i in 1..32 {
                    let handle = graph.add_graphics_pass(GraphicsPass::new(format!("pass_{i}")));
                    graph.add_dependency(handle, prev);
                    prev = handle;
                }
                graph
            },
            |mut graph| {
                black_box(graph.compile(RenderGraphCompilationMode::Strict).unwrap());
            },
        );
    });
}

// ---------------------------------------------------------------------------
// Dummy backend resource creation
// ---------------------------------------------------------------------------

fn bench_dummy_create_buffer(c: &mut Criterion) {
    let instance = GraphicsInstance::new().unwrap();
    let device = instance.create_device().unwrap();

    c.bench_function("dummy_create_buffer_1kb", |b| {
        b.iter(|| {
            black_box(
                device
                    .create_buffer(&BufferDescriptor::new(1024, BufferUsage::VERTEX))
                    .unwrap(),
            );
        });
    });
}

fn bench_dummy_create_texture(c: &mut Criterion) {
    let instance = GraphicsInstance::new().unwrap();
    let device = instance.create_device().unwrap();

    c.bench_function("dummy_create_texture_256x256", |b| {
        b.iter(|| {
            black_box(
                device
                    .create_texture(&TextureDescriptor::new_2d(
                        256,
                        256,
                        TextureFormat::Rgba8Unorm,
                        TextureUsage::TEXTURE_BINDING,
                    ))
                    .unwrap(),
            );
        });
    });
}

criterion_group!(
    benches,
    bench_graph_build_small,
    bench_graph_build_large,
    bench_graph_build_mixed,
    bench_graph_compile_small,
    bench_graph_compile_large,
    bench_dummy_create_buffer,
    bench_dummy_create_texture,
);
criterion_main!(benches);
