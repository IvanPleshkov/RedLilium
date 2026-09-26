//! Actual GPU convolution and graph-ordering tests.
#[allow(dead_code)]
mod common;
use common::{Backend, TestContext};
use half::f16;
use redlilium_graphics::{
    ibl::{EnvironmentFilter, EnvironmentFilterSettings},
    *,
};
use rstest::rstest;
use std::sync::Arc;

fn cube(ctx: &TestContext, size: u32) -> Arc<Texture> {
    ctx.device
        .create_texture(
            &TextureDescriptor::new_cube(
                size,
                TextureFormat::Rgba16Float,
                TextureUsage::TEXTURE_BINDING | TextureUsage::COPY_SRC | TextureUsage::COPY_DST,
            )
            .with_mip_levels(32 - size.leading_zeros()),
        )
        .unwrap()
}
fn ray(face: u32, x: u32, y: u32, size: u32) -> [f32; 3] {
    let u = 2.0 * (x as f32 + 0.5) / size as f32 - 1.0;
    let v = 2.0 * (y as f32 + 0.5) / size as f32 - 1.0;
    let n = match face {
        0 => [1.0, -v, -u],
        1 => [-1.0, -v, u],
        2 => [u, 1.0, v],
        3 => [u, -1.0, -v],
        4 => [u, -v, 1.0],
        _ => [-u, -v, -1.0],
    };
    let len = n.iter().map(|v| v * v).sum::<f32>().sqrt();
    n.map(|v| v / len)
}
fn upload(
    ctx: &TestContext,
    graph: &mut RenderGraph,
    source: &Arc<Texture>,
    sample: impl Fn([f32; 3]) -> [f32; 3],
) -> PassHandle {
    let size = source.width();
    let mut ops = Vec::new();
    for face in 0..6 {
        let mut bytes = Vec::new();
        for y in 0..size {
            for x in 0..size {
                for c in sample(ray(face, x, y, size)).into_iter().chain([1.0]) {
                    bytes.extend_from_slice(&f16::from_f32(c).to_le_bytes());
                }
            }
        }
        ops.push(
            TransferOperation::upload_texture_level(&ctx.device, source.clone(), 0, face, &bytes)
                .unwrap(),
        );
    }
    ops.push(TransferOperation::generate_mipmaps(source.clone()));
    let mut pass = TransferPass::new("captured environment".into());
    pass.set_transfer_config(TransferConfig::new().with_operations(ops));
    graph.add_transfer_pass(pass)
}
struct Read {
    buffer: Arc<Buffer>,
    size: u32,
    mip: u32,
    face: u32,
    diffuse: bool,
}
fn read_outputs(
    ctx: &TestContext,
    graph: &mut RenderGraph,
    job: &ibl::EnvironmentFilterJob,
    after: PassHandle,
) -> Vec<Read> {
    let mut reads = Vec::new();
    let mut ops = Vec::new();
    for (texture, diffuse) in [(job.specular(), false), (job.diffuse(), true)] {
        for mip in 0..texture.mip_level_count() {
            let size = (texture.width() >> mip).max(1);
            for face in 0..6 {
                let buffer = ctx.create_readback_buffer(u64::from(size * 256));
                ops.push(TransferOperation::readback_texture(
                    texture.clone(),
                    buffer.clone(),
                    vec![BufferTextureCopyRegion::new(
                        BufferTextureLayout::new(0, Some(256), None),
                        TextureCopyLocation::new(mip, TextureOrigin::new(0, 0, face)),
                        Extent3d::new_2d(size, size),
                    )],
                ));
                reads.push(Read {
                    buffer,
                    size,
                    mip,
                    face,
                    diffuse,
                });
            }
        }
    }
    let mut pass = TransferPass::new("filtered environment readback".into());
    pass.set_transfer_config(TransferConfig::new().with_operations(ops));
    let handle = graph.add_transfer_pass(pass);
    graph.add_dependency(handle, after);
    reads
}
fn verify(
    ctx: &TestContext,
    reads: Vec<Read>,
    expected: impl Fn(&Read, u32, u32) -> [f32; 3],
    tolerance: f32,
) {
    for read in reads {
        let bytes = ctx.read_buffer(&read.buffer, u64::from(read.size * 256));
        for y in 0..read.size {
            for x in 0..read.size {
                let offset = (y * 256 + x * 8) as usize;
                let actual: Vec<_> = bytes[offset..offset + 8]
                    .chunks_exact(2)
                    .map(|v| f16::from_le_bytes(v.try_into().unwrap()).to_f32())
                    .collect();
                let reference = expected(&read, x, y);
                for c in 0..3 {
                    assert!(
                        (actual[c] - reference[c]).abs() < tolerance,
                        "{:?}, diffuse {}, mip {}, face {}, ({x},{y}) channel {c}: {actual:?} expected {reference:?}",
                        ctx.backend,
                        read.diffuse,
                        read.mip,
                        read.face
                    );
                }
                assert_eq!(actual[3], 1.0);
            }
        }
    }
}
fn assert_validation(backend: Backend) {
    #[cfg(feature = "vulkan-backend")]
    if backend == Backend::Vulkan {
        assert_eq!(diagnostics::vulkan::validation_error_count(), 0);
    }
}
#[rstest]
#[case::vulkan(Backend::Vulkan)]
#[case::wgpu(Backend::WebGpu)]
fn constant_hdr_survives_updates_and_job_drop(#[case] backend: Backend) {
    let Some(ctx) = TestContext::new_with_validation(backend) else {
        return;
    };
    #[cfg(feature = "vulkan-backend")]
    diagnostics::vulkan::reset_validation_error_count();
    let source = cube(&ctx, 8);
    let filter = EnvironmentFilter::new(ctx.device.clone()).unwrap();
    let job = filter
        .prepare(
            source.clone(),
            EnvironmentFilterSettings {
                specular_size: 8,
                diffuse_size: 4,
                specular_samples: 64,
                diffuse_samples: 64,
            },
        )
        .unwrap();
    assert_eq!(job.max_reflection_lod(), 3.0);
    for color in [[4.0, 2.0, 1.0], [0.5, 8.0, 3.0]] {
        let mut graph = RenderGraph::new();
        let capture = upload(&ctx, &mut graph, &source, |_| color);
        let done = job.add_to_graph(&mut graph, &[capture]);
        let reads = read_outputs(&ctx, &mut graph, &job, done);
        ctx.execute_graph(graph);
        verify(&ctx, reads, |_, _, _| color, 0.02);
    }
    let mut graph = RenderGraph::new();
    let done = job.add_to_graph(&mut graph, &[]);
    let reads = read_outputs(&ctx, &mut graph, &job, done);
    drop(job);
    drop(filter);
    drop(source);
    ctx.execute_graph(graph);
    verify(&ctx, reads, |_, _, _| [0.5, 8.0, 3.0], 0.02);
    drop(ctx);
    assert_validation(backend);
}

#[rstest]
#[case::vulkan(Backend::Vulkan)]
#[case::wgpu(Backend::WebGpu)]
fn directional_environment_preserves_orientation_and_diffuse_normalization(
    #[case] backend: Backend,
) {
    let Some(ctx) = TestContext::new_with_validation(backend) else {
        return;
    };
    #[cfg(feature = "vulkan-backend")]
    diagnostics::vulkan::reset_validation_error_count();
    let source = cube(&ctx, 32);
    let filter = EnvironmentFilter::new(ctx.device.clone()).unwrap();
    let job = filter
        .prepare(
            source.clone(),
            EnvironmentFilterSettings {
                specular_size: 8,
                diffuse_size: 4,
                specular_samples: 1024,
                diffuse_samples: 1024,
            },
        )
        .unwrap();
    let mut graph = RenderGraph::new();
    let capture = upload(&ctx, &mut graph, &source, |n| n.map(|v| 0.5 + 0.25 * v));
    let done = job.add_to_graph(&mut graph, &[capture]);
    let reads = read_outputs(&ctx, &mut graph, &job, done);
    ctx.execute_graph(graph);
    // Analytic diffuse convolution of a linear directional field has factor 2/3.
    // GGX at roughness=1 reduces to the same cosine-weighted hemisphere integral.
    // Intermediate GGX levels use independent deterministic quadrature over
    // cos(theta), rather than duplicating the shader's importance sampler.
    let factors: Vec<_> = (0..4)
        .map(|mip| {
            if mip == 0 {
                return 1.0;
            }
            let a2 = (mip as f64 / 3.0).powi(4);
            let mut numerator = 0.0;
            let mut denominator = 0.0;
            for i in 0..65536 {
                let mu = (i as f64 + 0.5) / 65536.0;
                let d = a2 / (((1.0 + mu) * 0.5 * (a2 - 1.0) + 1.0).powi(2));
                numerator += mu * mu * d;
                denominator += mu * d;
            }
            (numerator / denominator) as f32
        })
        .collect();
    verify(
        &ctx,
        reads,
        |r, x, y| {
            let factor = if r.diffuse {
                2.0 / 3.0
            } else {
                factors[r.mip as usize]
            };
            ray(r.face, x, y, r.size).map(|v| 0.5 + 0.25 * factor * v)
        },
        0.018,
    );
    drop(job);
    drop(filter);
    drop(source);
    drop(ctx);
    assert_validation(backend);
}

#[rstest]
#[case::vulkan(Backend::Vulkan)]
#[case::wgpu(Backend::WebGpu)]
fn tiny_hdr_light_contributes_to_rough_reflections(#[case] backend: Backend) {
    let Some(ctx) = TestContext::new_with_validation(backend) else {
        return;
    };
    #[cfg(feature = "vulkan-backend")]
    diagnostics::vulkan::reset_validation_error_count();
    let source = cube(&ctx, 32);
    let filter = EnvironmentFilter::new(ctx.device.clone()).unwrap();
    let job = filter
        .prepare(
            source.clone(),
            EnvironmentFilterSettings {
                specular_size: 8,
                diffuse_size: 1,
                specular_samples: 1024,
                diffuse_samples: 1024,
            },
        )
        .unwrap();
    let mut graph = RenderGraph::new();
    // Exactly one +X texel contains a light much brighter than display white.
    let capture = upload(&ctx, &mut graph, &source, |n| {
        let lit = n[0] > 0.99 && n[1] < 0.0 && n[2] < 0.0 && n[1].abs() < 0.04 && n[2].abs() < 0.04;
        if lit { [40000.0; 3] } else { [0.0; 3] }
    });
    let done = job.add_to_graph(&mut graph, &[capture]);
    let reads = read_outputs(&ctx, &mut graph, &job, done);
    ctx.execute_graph(graph);
    let area = (0.0625f64 * 0.0625 / (1.0 + 2.0 * 0.0625f64.powi(2)).sqrt()).atan();
    let reference = (40000.0 * area / std::f64::consts::PI) as f32;
    for read in reads {
        let bytes = ctx.read_buffer(&read.buffer, u64::from(read.size * 256));
        for y in 0..read.size {
            for x in 0..read.size {
                for c in 0..3 {
                    let offset = (y * 256 + x * 8 + c * 2) as usize;
                    let value =
                        f16::from_le_bytes(bytes[offset..offset + 2].try_into().unwrap()).to_f32();
                    assert!(value.is_finite() && (0.0..=40000.0).contains(&value));
                    if read.face == 0 && (read.diffuse || read.mip == 3) {
                        // Finite-sample convolution + ordinary source mips is approximate,
                        // but must retain the tiny light's energy rather than miss it.
                        assert!(
                            (value - reference).abs() < reference * 0.35,
                            "{backend:?} diffuse {}: {value} vs {reference}",
                            read.diffuse
                        );
                    }
                }
            }
        }
    }
    drop(job);
    drop(filter);
    drop(source);
    drop(ctx);
    assert_validation(backend);
}
