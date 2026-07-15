//! Regenerates the car game's scene assets (`game/assets/scenes/*.scene`)
//! and the mount's `assets.db`.
//!
//! Scenes are authored in code (via the same spawn functions the game uses)
//! until the editor persists scenes (#2). Run from the repo root after
//! changing `spawn_level` / `spawn_menu_backdrop`:
//!
//! ```bash
//! cargo run -p car-game --bin gen_scenes
//! ```
#![recursion_limit = "256"]

use car_game::{LEVEL_SCENE, MENU_SCENE};
use redlilium_assets::{AssetDb, AssetLoader, AssetPath, AssetRecord, Guid};
use redlilium_ecs::{
    SceneLoader, World, register_rendering_components, register_std_components, serialize_scene_ron,
};

/// FNV-1a, matching the asset scanner's content hash (`assets::scan`), so an
/// editor rescan of the mount sees the generated records as up-to-date.
fn fnv1a(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

fn authoring_world() -> World {
    let mut world = World::new();
    register_std_components(&mut world);
    register_rendering_components(&mut world);
    world.register_inspector::<car_game::CarController>();
    world.register_inspector::<car_game::FollowCamera>();
    world
}

type SpawnFn = fn(&mut World);

fn main() {
    let scenes: Vec<(&str, SpawnFn)> = vec![
        (LEVEL_SCENE, car_game::spawn_level),
        (MENU_SCENE, car_game::spawn_menu_backdrop),
    ];

    std::fs::create_dir_all("game/assets/scenes").expect("create scenes dir");

    let mut db = AssetDb::new();
    for (path, spawn) in scenes {
        let mut world = authoring_world();
        spawn(&mut world);
        let ron = serialize_scene_ron(&world).expect("serialize scene");
        std::fs::write(format!("game/assets/{path}"), &ron).expect("write scene");
        db.insert(
            Guid::stable(path),
            AssetRecord {
                path: AssetPath::new("game", path),
                kind: SceneLoader::NAME.to_string(),
                source_hash: fnv1a(ron.as_bytes()),
                settings: None,
                references: Default::default(),
            },
        )
        .expect("db insert");
        println!("wrote game/assets/{path}");
    }

    std::fs::write(
        "game/assets/assets.db",
        db.to_ron_for_mount("game").expect("db -> ron"),
    )
    .expect("write assets.db");
    println!("wrote game/assets/assets.db");
}
