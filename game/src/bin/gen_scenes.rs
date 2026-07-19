//! Regenerates the car game's scene assets (`game/assets/scenes/*.scene`)
//! and the mount's `assets.db` from the code-side spawn functions.
//!
//! **This is a reset tool.** Scenes are normally authored in the *editor*
//! (#2/#105: open via the `game` mount, edit, Cmd+S / `save_scene`) — running
//! this bin OVERWRITES any editor-authored content with the code-generated
//! baseline. Run from the repo root only to bootstrap or reset the pack:
//!
//! ```bash
//! cargo run -p car-game --bin gen_scenes
//! ```
#![recursion_limit = "256"]

use car_game::{LEVEL_SCENE, MENU_SCENE, PLAYGROUND_SCENE};
use redlilium_assets::{AssetDb, AssetLoader, AssetPath, AssetRecord, Guid};
use redlilium_ecs::rendering::{MaterialInstanceData, PropValue};
use redlilium_ecs::{
    MaterialInstanceLoader, SceneLoader, World, register_rendering_components,
    register_std_components, serialize_scene_ron,
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
    use redlilium_runtime::Plugin;
    redlilium_levels::LevelsPlugin.register_types(&mut world);
    world.register_inspector::<car_game::CarController>();
    world.register_inspector::<car_game::FollowCamera>();
    world.register_inspector::<car_game::CarSpawn>();
    world
}

type SpawnFn = fn(&mut World);

fn main() {
    let scenes: Vec<(&str, SpawnFn)> = vec![
        // The level carries a CarSpawn marker, not the car (#105) — the game
        // spawns the chassis at the marker when the scene loads.
        (LEVEL_SCENE, car_game::spawn_level_scene),
        (MENU_SCENE, car_game::spawn_menu_backdrop),
        // Road-graph playground (docs/DESIGN_PROCEDURAL_LEVELS.md prototype).
        (PLAYGROUND_SCENE, car_game::spawn_levels_playground),
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

    // The car's paint (#144): a pbr material instance parented on the std
    // pbr material. The asset is an empty file — the settings live in the
    // db record, like the std material pack.
    let car_matinst = "materials/car.matinst";
    let car_paint = MaterialInstanceData {
        parent: Guid::stable("materials/pbr.material"),
        overrides: vec![
            (
                "base_color".to_owned(),
                PropValue::Vec4([0.72, 0.06, 0.05, 1.0]),
            ),
            (
                "pbr_params".to_owned(),
                PropValue::Vec4([0.9, 0.35, 0.0, 0.0]),
            ),
        ],
    };
    std::fs::create_dir_all("game/assets/materials").expect("create materials dir");
    std::fs::write(format!("game/assets/{car_matinst}"), "").expect("write car.matinst");
    db.insert(
        Guid::stable(car_matinst),
        AssetRecord {
            path: AssetPath::new("game", car_matinst),
            kind: MaterialInstanceLoader::NAME.to_string(),
            source_hash: fnv1a(b""),
            settings: Some(ron::to_string(&car_paint).expect("car paint -> ron")),
            references: Default::default(),
        },
    )
    .expect("db insert");
    println!("wrote game/assets/{car_matinst}");

    std::fs::write(
        "game/assets/assets.db",
        db.to_ron_for_mount("game").expect("db -> ron"),
    )
    .expect("write assets.db");
    println!("wrote game/assets/assets.db");
}
