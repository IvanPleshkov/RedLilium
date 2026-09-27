mod scenarios;
use redlilium_ecs::SystemsContainer;
use std::sync::Arc;

#[unsafe(no_mangle)]
pub unsafe extern "C" fn install(
    systems: *mut SystemsContainer,
    stats: *const Arc<scenarios::Stats>,
    kind: u32,
    failure: u32,
) {
    // Host and guest are built from identical engine artifacts and fixture types.
    scenarios::install(
        unsafe { &mut *systems },
        unsafe { &*stats }.clone(),
        kind,
        failure,
    );
}
