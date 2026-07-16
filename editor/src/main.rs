//! The engine repo's own editor binary: no statically linked game
//! (`REDLILIUM_GAME=<cdylib>` hosts one dynamically). Game projects own
//! their editor binary instead — see `redlilium_editor::run` (ADR-033).

fn main() {
    redlilium_editor::run_without_game();
}
