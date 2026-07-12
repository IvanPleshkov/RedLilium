//! ECS bridge for the asset system.
//!
//! Two **loader-agnostic** systems drive an [`AssetProcessor`] resource: they
//! know nothing about specific loaders or asset types — they only tick the
//! processor (drain async work onto the compute/IO pools, collect results, and
//! flush GPU residency into the frame graph). Consumers (e.g. the rendering
//! managers) request assets and use the delivered results themselves; that
//! wiring lives outside this bridge by design.
//!
//! Register [`AssetPump`] in a regular schedule (e.g. `Update`) and
//! [`AssetGpuFlush`] in the `Render` schedule, and insert an [`AssetProcessor`]
//! as a resource. Both systems no-op when their inputs are absent.

use redlilium_assets::{AssetProcessor, Executor};

use crate::std::rendering::RenderSchedule;
use crate::system::SystemError;
use crate::{IoRunner, Priority, System, SystemContext};

/// Drives async (IO/CPU) asset loading: drains the [`AssetProcessor`]'s ready
/// stage tasks, spawns each on the matching executor (`ctx.io()` for IO,
/// `ctx.compute()` for CPU), then collects finished results to advance requests.
///
/// Runs in a regular schedule (no frame graph needed). No-op without an
/// `AssetProcessor` resource.
pub struct AssetPump;

impl System for AssetPump {
    type Result = ();
    fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) -> Result<Self::Result, SystemError> {
        let world = ctx.raw_world();
        if !world.has_resource::<AssetProcessor>() {
            return Ok(());
        }

        // Drain under the resource lock, then spawn without holding it.
        let tasks = world.resource_mut::<AssetProcessor>().drain_tasks();
        for (executor, fut) in tasks {
            match executor {
                // Fire-and-forget: the work is already scheduled (tokio / the
                // compute pool); each future reports its result through the
                // processor's own channel, so the returned handle is dropped
                // (neither pool cancels a task when its handle is dropped).
                Executor::Io => {
                    drop(ctx.io().run(fut));
                }
                Executor::Cpu => {
                    drop(ctx.compute().spawn(Priority::Low, move |_cctx| fut));
                }
                // `drain_tasks` never yields GPU stages — those run in
                // `AssetGpuFlush` via the frame graph.
                Executor::Gpu => {}
            }
        }

        world.resource_mut::<AssetProcessor>().collect();
        Ok(())
    }
}

/// Flushes ready GPU asset stages into the frame graph held by the
/// [`RenderSchedule`] resource (adding their upload transfer pass), making the
/// freshly loaded assets GPU-resident through the graph.
///
/// Runs in the `Render` schedule (it needs the bound graph). No-op without an
/// `AssetProcessor` resource or a bound graph.
pub struct AssetGpuFlush;

impl System for AssetGpuFlush {
    type Result = ();
    fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) -> Result<Self::Result, SystemError> {
        let world = ctx.raw_world();
        if !world.has_resource::<AssetProcessor>() || !world.has_resource::<RenderSchedule>() {
            return Ok(());
        }
        let mut schedule = world.resource_mut::<RenderSchedule>();
        let Some(graph) = schedule.graph_mut() else {
            return Ok(()); // no frame graph bound (not in the render bracket)
        };
        world.resource_mut::<AssetProcessor>().flush_gpu(graph);
        Ok(())
    }
}
