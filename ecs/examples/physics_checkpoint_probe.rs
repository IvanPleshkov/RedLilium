//! Headless release measurement and native/WASM continuation fingerprint.
//! See book/src/ecs/physics-checkpoints.md for commands and measurement scope.
use redlilium_core::math::Vec3;
use redlilium_ecs::physics::PhysicsEventDelivery;
use redlilium_ecs::{EcsRunner, FixedUpdate, Schedules, Transform, World, register_std_components};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicIsize, Ordering};
#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;
#[cfg(target_arch = "wasm32")]
use web_time::Instant;

// Counts requested live heap bytes, not allocator overhead or shared geometry.
struct CountAlloc;
static LIVE: AtomicIsize = AtomicIsize::new(0);
unsafe impl GlobalAlloc for CountAlloc {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = unsafe { System.alloc(layout) };
        if !ptr.is_null() {
            LIVE.fetch_add(layout.size() as isize, Ordering::Relaxed);
        }
        ptr
    }
    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        LIVE.fetch_sub(layout.size() as isize, Ordering::Relaxed);
        unsafe { System.dealloc(ptr, layout) };
    }
}
#[global_allocator]
static ALLOCATOR: CountAlloc = CountAlloc;

macro_rules! measurement {
    () => {
        pub fn run() -> String {
            let mut w = World::new();
            register_std_components(&mut w);
            let mut p = Physics::default();
            p.gravity.y = -9.81;
            p.event_delivery = PhysicsEventDelivery::TickOnly;
            w.insert_resource(p);
            w.spawn_with((
                Transform::from_translation(Vec3::new(0.0, -1000.5, 0.0)),
                Body::fixed(),
                Collider::ball(1000.0),
            ))
            .unwrap();
            for i in 0..100 {
                w.spawn_with((
                    Transform::from_translation(Vec3::new(
                        (i % 10) as f32 * 1.1,
                        (i / 10) as f32 * 1.02,
                        0.0,
                    )),
                    Body::dynamic(),
                    Collider::ball(0.5),
                ))
                .unwrap();
            }
            let mut s = Schedules::new();
            s.set_fixed_timestep(1.0 / 128.0);
            let fixed = s.get_mut::<FixedUpdate>();
            fixed.add_exclusive(BodySync);
            fixed.add_exclusive(JointSync);
            fixed.add(Step);
            fixed.add(Record);
            fixed.add_edge::<BodySync, JointSync>().unwrap();
            fixed.add_edge::<JointSync, Step>().unwrap();
            fixed.add_edge::<Step, Record>().unwrap();
            let r = EcsRunner::single_thread();
            for _ in 0..128 {
                s.run_fixed_tick(&mut w, &r).unwrap();
            }
            let clock = s.capture_fixed_clock(&w).unwrap();
            let before = LIVE.load(Ordering::Relaxed);
            let checkpoint = Bindings::with(&mut w, |p, b| p.capture_checkpoint(b)).unwrap();
            let bytes = LIVE.load(Ordering::Relaxed) - before + std::mem::size_of_val(&checkpoint) as isize;
            let sample = |w: &World| {
                let p = w.resource::<Physics>();
                p.bodies()
                    .iter()
                    .map(|(h, b)| {
                        format!(
                            "{h:?}:{:?}:{:?}:{:?}:{}",
                            b.position(),
                            b.linvel(),
                            b.angvel(),
                            b.is_sleeping()
                        )
                    })
                    .collect::<String>()
            };
            let mut expected = String::new();
            for _ in 0..64 {
                s.run_fixed_tick(&mut w, &r).unwrap();
                expected.push_str(&sample(&w));
            }
            Bindings::with(&mut w, |p, b| p.restore_checkpoint(&checkpoint, b)).unwrap();
            s.restore_fixed_clock(&mut w, &clock).unwrap();
            let mut actual = String::new();
            for _ in 0..64 {
                s.run_fixed_tick(&mut w, &r).unwrap();
                actual.push_str(&sample(&w));
            }
            assert_eq!(
                actual, expected,
                "continuation must be exact in the same world"
            );
            let hash = expected.bytes().fold(0xcbf29ce484222325u64, |h, b| {
                (h ^ b as u64).wrapping_mul(0x100000001b3)
            });
            Bindings::with(&mut w, |p, b| p.restore_checkpoint(&checkpoint, b)).unwrap();
            let start = Instant::now();
            for _ in 0..100 {
                std::hint::black_box(Bindings::with(&mut w, |p, b| p.capture_checkpoint(b)).unwrap());
            }
            let capture_us = start.elapsed().as_secs_f64() * 10_000.0;
            let start = Instant::now();
            for _ in 0..100 {
                Bindings::with(&mut w, |p, b| p.restore_checkpoint(&checkpoint, b)).unwrap();
            }
            let restore_us = start.elapsed().as_secs_f64() * 10_000.0;
            format!(
                "{} scalar_bytes={} bodies=101 retained_bytes={bytes} capture_drop_us={capture_us:.1} restore_us={restore_us:.1} continuation={hash:016x}",
                module_path!(),
                std::mem::size_of::<Real>()
            )
        }
    };
}
#[cfg(any(feature = "physics-2d", feature = "physics-2d-f32"))]
mod dim2 {
    use super::*;
    use redlilium_ecs::physics::{
        components2d::{Collider2D as Collider, RigidBody2D as Body},
        rapier2d::prelude::Real,
        systems2d::{
            RecordPhysicsPose2D as Record, StepPhysics2D as Step, SyncPhysicsBodies2D as BodySync,
            SyncPhysicsJoints2D as JointSync,
        },
        world2d::{PhysicsBindings2D as Bindings, PhysicsWorld2D as Physics},
    };
    measurement!();
}
#[cfg(any(feature = "physics-3d", feature = "physics-3d-f32"))]
mod dim3 {
    use super::*;
    use redlilium_ecs::physics::{
        components3d::{Collider3D as Collider, RigidBody3D as Body},
        rapier3d::prelude::Real,
        systems3d::{
            RecordPhysicsPose as Record, StepPhysics3D as Step, SyncPhysicsBodies3D as BodySync,
            SyncPhysicsJoints3D as JointSync,
        },
        world3d::{PhysicsBindings3D as Bindings, PhysicsWorld3D as Physics},
    };
    measurement!();
}
#[cfg(target_arch = "wasm32")]
#[path = "../tests/fixed_replay.rs"]
mod clock_contracts;
#[cfg(target_arch = "wasm32")]
#[path = "../tests/physics_checkpoints.rs"]
mod physics_contracts;

#[cfg_attr(target_arch = "wasm32", wasm_bindgen::prelude::wasm_bindgen)]
pub fn probe() -> String {
    #[cfg(target_arch = "wasm32")]
    {
        physics_contracts::run_wasm_contracts();
        clock_contracts::run_wasm_contracts();
    }
    let mut lines = vec!["checkpoint probe: exact local continuation verified".to_owned()];
    #[cfg(any(feature = "physics-2d", feature = "physics-2d-f32"))]
    lines.push(dim2::run());
    #[cfg(any(feature = "physics-3d", feature = "physics-3d-f32"))]
    lines.push(dim3::run());
    lines.join("\n")
}
fn main() {
    println!("{}", probe());
}
