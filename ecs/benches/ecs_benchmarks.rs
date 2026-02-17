#![allow(dead_code)]

use criterion::{BatchSize, Criterion, black_box, criterion_group, criterion_main};

use redlilium_ecs::{SparseSetInner, World};

// ---------------------------------------------------------------------------
// Helper component types
// ---------------------------------------------------------------------------

#[derive(Clone, Copy)]
struct Position {
    x: f32,
    y: f32,
    z: f32,
}

#[derive(Clone, Copy)]
struct Velocity {
    x: f32,
    y: f32,
    z: f32,
}

#[derive(Clone, Copy)]
struct Health(f32);

// ---------------------------------------------------------------------------
// Entity spawning
// ---------------------------------------------------------------------------

fn bench_spawn_entities_1k(c: &mut Criterion) {
    c.bench_function("spawn_1k_entities", |b| {
        b.iter_batched(
            World::new,
            |mut world| {
                for _ in 0..1_000 {
                    black_box(world.spawn());
                }
            },
            BatchSize::SmallInput,
        );
    });
}

fn bench_spawn_entities_10k(c: &mut Criterion) {
    c.bench_function("spawn_10k_entities", |b| {
        b.iter_batched(
            World::new,
            |mut world| {
                for _ in 0..10_000 {
                    black_box(world.spawn());
                }
            },
            BatchSize::SmallInput,
        );
    });
}

fn bench_spawn_and_despawn_recycling(c: &mut Criterion) {
    c.bench_function("spawn_despawn_recycle_1k", |b| {
        b.iter_batched(
            || {
                let mut world = World::new();
                let entities: Vec<_> = (0..1_000).map(|_| world.spawn()).collect();
                for e in &entities {
                    world.despawn(*e);
                }
                world
            },
            |mut world| {
                // Re-spawning should reuse recycled slots
                for _ in 0..1_000 {
                    black_box(world.spawn());
                }
            },
            BatchSize::SmallInput,
        );
    });
}

// ---------------------------------------------------------------------------
// Component insert / remove via World
// ---------------------------------------------------------------------------

fn bench_insert_component_1k(c: &mut Criterion) {
    c.bench_function("insert_component_1k", |b| {
        b.iter_batched(
            || {
                let mut world = World::new();
                world.register_component::<Position>();
                let entities: Vec<_> = (0..1_000).map(|_| world.spawn()).collect();
                (world, entities)
            },
            |(mut world, entities)| {
                for (i, e) in entities.iter().enumerate() {
                    world
                        .insert(
                            *e,
                            Position {
                                x: i as f32,
                                y: 0.0,
                                z: 0.0,
                            },
                        )
                        .unwrap();
                }
            },
            BatchSize::SmallInput,
        );
    });
}

fn bench_remove_component_1k(c: &mut Criterion) {
    c.bench_function("remove_component_1k", |b| {
        b.iter_batched(
            || {
                let mut world = World::new();
                world.register_component::<Position>();
                let entities: Vec<_> = (0..1_000).map(|_| world.spawn()).collect();
                for (i, e) in entities.iter().enumerate() {
                    world
                        .insert(
                            *e,
                            Position {
                                x: i as f32,
                                y: 0.0,
                                z: 0.0,
                            },
                        )
                        .unwrap();
                }
                (world, entities)
            },
            |(mut world, entities)| {
                for e in &entities {
                    black_box(world.remove::<Position>(*e));
                }
            },
            BatchSize::SmallInput,
        );
    });
}

// ---------------------------------------------------------------------------
// Component iteration via World read/write
// ---------------------------------------------------------------------------

fn bench_iterate_single_component_10k(c: &mut Criterion) {
    let mut world = World::new();
    world.register_component::<Position>();
    for i in 0..10_000 {
        let e = world.spawn();
        world
            .insert(
                e,
                Position {
                    x: i as f32,
                    y: 0.0,
                    z: 0.0,
                },
            )
            .unwrap();
    }

    c.bench_function("iterate_single_10k", |b| {
        b.iter(|| {
            let positions = world.read::<Position>().unwrap();
            let mut sum = 0.0f32;
            for (_idx, pos) in positions.iter() {
                sum += pos.x;
            }
            black_box(sum);
        });
    });
}

fn bench_iterate_two_components_10k(c: &mut Criterion) {
    let mut world = World::new();
    world.register_component::<Position>();
    world.register_component::<Velocity>();
    for i in 0..10_000 {
        let e = world.spawn();
        world
            .insert(
                e,
                Position {
                    x: i as f32,
                    y: 0.0,
                    z: 0.0,
                },
            )
            .unwrap();
        world
            .insert(
                e,
                Velocity {
                    x: 1.0,
                    y: 0.0,
                    z: 0.0,
                },
            )
            .unwrap();
    }

    c.bench_function("iterate_two_components_10k", |b| {
        b.iter(|| {
            let positions = world.read::<Position>().unwrap();
            let velocities = world.read::<Velocity>().unwrap();
            let mut sum = 0.0f32;
            for (idx, pos) in positions.iter() {
                if let Some(vel) = velocities.get(idx) {
                    sum += pos.x + vel.x;
                }
            }
            black_box(sum);
        });
    });
}

// ---------------------------------------------------------------------------
// SparseSetInner operations (direct)
// ---------------------------------------------------------------------------

fn bench_sparse_set_insert_10k(c: &mut Criterion) {
    c.bench_function("sparse_set_insert_10k", |b| {
        b.iter_batched(
            SparseSetInner::<Position>::new,
            |mut set| {
                for i in 0..10_000u32 {
                    set.insert(
                        i,
                        Position {
                            x: i as f32,
                            y: 0.0,
                            z: 0.0,
                        },
                    );
                }
            },
            BatchSize::SmallInput,
        );
    });
}

fn bench_sparse_set_get_10k(c: &mut Criterion) {
    let mut set = SparseSetInner::<Position>::new();
    for i in 0..10_000u32 {
        set.insert(
            i,
            Position {
                x: i as f32,
                y: 0.0,
                z: 0.0,
            },
        );
    }

    c.bench_function("sparse_set_get_10k", |b| {
        b.iter(|| {
            let mut sum = 0.0f32;
            for i in 0..10_000u32 {
                if let Some(pos) = set.get(i) {
                    sum += pos.x;
                }
            }
            black_box(sum);
        });
    });
}

fn bench_sparse_set_iterate_10k(c: &mut Criterion) {
    let mut set = SparseSetInner::<Position>::new();
    for i in 0..10_000u32 {
        set.insert(
            i,
            Position {
                x: i as f32,
                y: 0.0,
                z: 0.0,
            },
        );
    }

    c.bench_function("sparse_set_dense_iterate_10k", |b| {
        b.iter(|| {
            let mut sum = 0.0f32;
            for (_idx, pos) in set.iter() {
                sum += pos.x;
            }
            black_box(sum);
        });
    });
}

fn bench_sparse_set_remove_half(c: &mut Criterion) {
    c.bench_function("sparse_set_remove_5k_of_10k", |b| {
        b.iter_batched(
            || {
                let mut set = SparseSetInner::<Position>::new();
                for i in 0..10_000u32 {
                    set.insert(
                        i,
                        Position {
                            x: i as f32,
                            y: 0.0,
                            z: 0.0,
                        },
                    );
                }
                set
            },
            |mut set| {
                // Remove every other entity
                for i in (0..10_000u32).step_by(2) {
                    black_box(set.remove(i));
                }
            },
            BatchSize::SmallInput,
        );
    });
}

// ---------------------------------------------------------------------------
// Fragmented iteration (sparse overlap)
// ---------------------------------------------------------------------------

fn bench_iterate_fragmented_10k(c: &mut Criterion) {
    let mut world = World::new();
    world.register_component::<Position>();
    world.register_component::<Velocity>();
    world.register_component::<Health>();

    // Only half have Velocity, quarter have Health
    for i in 0..10_000 {
        let e = world.spawn();
        world
            .insert(
                e,
                Position {
                    x: i as f32,
                    y: 0.0,
                    z: 0.0,
                },
            )
            .unwrap();
        if i % 2 == 0 {
            world
                .insert(
                    e,
                    Velocity {
                        x: 1.0,
                        y: 0.0,
                        z: 0.0,
                    },
                )
                .unwrap();
        }
        if i % 4 == 0 {
            world.insert(e, Health(100.0)).unwrap();
        }
    }

    c.bench_function("iterate_fragmented_3_components_10k", |b| {
        b.iter(|| {
            let positions = world.read::<Position>().unwrap();
            let velocities = world.read::<Velocity>().unwrap();
            let healths = world.read::<Health>().unwrap();
            let mut sum = 0.0f32;
            for (idx, pos) in positions.iter() {
                if let Some(vel) = velocities.get(idx)
                    && let Some(hp) = healths.get(idx)
                {
                    sum += pos.x + vel.x + hp.0;
                }
            }
            black_box(sum);
        });
    });
}

criterion_group!(
    benches,
    bench_spawn_entities_1k,
    bench_spawn_entities_10k,
    bench_spawn_and_despawn_recycling,
    bench_insert_component_1k,
    bench_remove_component_1k,
    bench_iterate_single_component_10k,
    bench_iterate_two_components_10k,
    bench_sparse_set_insert_10k,
    bench_sparse_set_get_10k,
    bench_sparse_set_iterate_10k,
    bench_sparse_set_remove_half,
    bench_iterate_fragmented_10k,
);
criterion_main!(benches);
