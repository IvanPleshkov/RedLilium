#[allow(dead_code)]
mod scenarios;
use redlilium_ecs::{EcsRunner, SystemError, SystemsContainer, World};
use std::ffi::{CString, c_char, c_void};
use std::sync::{Arc, atomic::Ordering};
use std::time::Duration;
unsafe extern "C" {
    fn dlopen(path: *const c_char, flags: i32) -> *mut c_void;
    fn dlsym(module: *mut c_void, name: *const c_char) -> *mut c_void;
    fn dlclose(module: *mut c_void) -> i32;
}
fn main() {
    let path = CString::new(std::env::args().nth(1).unwrap()).unwrap();
    for multi in [false, true] {
        for kind in 0..3 {
            for failure in 0..3 {
                let module = unsafe { dlopen(path.as_ptr(), 2) };
                assert!(!module.is_null());
                let symbol = unsafe { dlsym(module, c"install".as_ptr()) };
                assert!(!symbol.is_null());
                let install: unsafe extern "C" fn(
                    *mut SystemsContainer,
                    *const Arc<scenarios::Stats>,
                    u32,
                    u32,
                ) = unsafe { std::mem::transmute(symbol) };
                let stats = Arc::new(scenarios::Stats::default());
                let mut systems = SystemsContainer::new();
                unsafe {
                    install(&mut systems, &stats, kind, failure);
                }
                let runner = if multi {
                    EcsRunner::multi_thread(1)
                } else {
                    EcsRunner::single_thread()
                };
                let mut world = World::new();
                assert!(runner.run(&mut world, &systems).is_empty());
                let errors = runner.run(&mut world, &systems);
                let expected =
                    ["run failure", "reuse failure", "result drop failure"][failure as usize];
                assert!(
                    matches!(errors.as_slice(), [SystemError::Panicked { message, .. }] if message == expected),
                    "{errors:?}"
                );
                assert_eq!(
                    stats.runs.load(Ordering::SeqCst),
                    if failure == 0 { 2 } else { 1 }
                );
                assert_eq!(stats.downstream.load(Ordering::SeqCst), 2);
                assert!(runner.run(&mut world, &systems).is_empty());
                assert_eq!(
                    stats.runs.load(Ordering::SeqCst),
                    if failure == 0 { 3 } else { 2 }
                );
                assert_eq!(stats.downstream.load(Ordering::SeqCst), 3);
                drop(world);
                drop(systems);
                runner.prepare_reload(Duration::from_secs(2)).unwrap();
                assert_eq!(unsafe { dlclose(module) }, 0);
                // Only owned diagnostics survive the image that created them.
                assert!(errors[0].to_string().contains(expected));
            }
        }
    }
    println!(
        "System cdylib: 18 scenarios passed; both runners, three system kinds, run/reuse/default-drop panics, recovery and diagnostics after unload"
    );
}
