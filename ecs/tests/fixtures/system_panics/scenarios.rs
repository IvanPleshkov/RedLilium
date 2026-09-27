use redlilium_ecs::{
    ExclusiveSystem, ReadOnlyExclusiveSystem, System, SystemContext, SystemError, SystemsContainer,
    World,
};
use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

#[derive(Default)]
pub struct Stats {
    pub runs: AtomicUsize,
    pub reuses: AtomicUsize,
    pub downstream: AtomicUsize,
}

pub struct ResultValue(bool);
impl Drop for ResultValue {
    fn drop(&mut self) {
        if self.0 {
            panic!("result drop failure");
        }
    }
}

struct Logic {
    stats: Arc<Stats>,
    failure: u32,
}
impl Logic {
    fn run(&self) -> Result<ResultValue, SystemError> {
        let run = self.stats.runs.fetch_add(1, Ordering::SeqCst) + 1;
        if self.failure == 0 && run == 2 {
            panic!("run failure");
        }
        Ok(ResultValue(self.failure == 2 && run == 1))
    }
    fn reuse(&self, prev: ResultValue) {
        let reuse = self.stats.reuses.fetch_add(1, Ordering::SeqCst) + 1;
        if self.failure == 1 && reuse == 1 {
            panic!("reuse failure");
        }
        drop(prev);
    }
}
struct Regular(Logic);
impl System for Regular {
    type Result = ResultValue;
    fn run<'a>(&'a self, _: &'a SystemContext<'a>) -> Result<ResultValue, SystemError> {
        self.0.run()
    }
    fn reuse_result(&self, prev: ResultValue) {
        self.0.reuse(prev);
    }
}
struct Exclusive(Logic);
impl ExclusiveSystem for Exclusive {
    type Result = ResultValue;
    fn run(&mut self, _: &mut World) -> Result<ResultValue, SystemError> {
        self.0.run()
    }
    fn reuse_result(&mut self, prev: ResultValue) {
        self.0.reuse(prev);
    }
}
struct ReadOnly(Logic);
impl ReadOnlyExclusiveSystem for ReadOnly {
    type Result = ResultValue;
    fn run(&self, _: &World) -> Result<ResultValue, SystemError> {
        self.0.run()
    }
    fn reuse_result(&self, prev: ResultValue) {
        self.0.reuse(prev);
    }
}
// Exercise the actual default hook for all three system traits.
struct DefaultRegular(Logic);
impl System for DefaultRegular {
    type Result = ResultValue;
    fn run<'a>(&'a self, _: &'a SystemContext<'a>) -> Result<ResultValue, SystemError> {
        self.0.run()
    }
}
struct DefaultExclusive(Logic);
impl ExclusiveSystem for DefaultExclusive {
    type Result = ResultValue;
    fn run(&mut self, _: &mut World) -> Result<ResultValue, SystemError> {
        self.0.run()
    }
}
struct DefaultReadOnly(Logic);
impl ReadOnlyExclusiveSystem for DefaultReadOnly {
    type Result = ResultValue;
    fn run(&self, _: &World) -> Result<ResultValue, SystemError> {
        self.0.run()
    }
}
struct Downstream(Arc<Stats>);
impl System for Downstream {
    type Result = ();
    fn run<'a>(&'a self, _: &'a SystemContext<'a>) -> Result<(), SystemError> {
        self.0.downstream.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
}

pub fn install(systems: &mut SystemsContainer, stats: Arc<Stats>, kind: u32, failure: u32) {
    let logic = Logic {
        stats: stats.clone(),
        failure,
    };
    systems.add(Downstream(stats));
    match (kind, failure == 2) {
        (0, false) => {
            systems.add(Regular(logic));
            systems.add_edge::<Regular, Downstream>().unwrap();
        }
        (1, false) => {
            systems.add_exclusive(Exclusive(logic));
            systems.add_edge::<Exclusive, Downstream>().unwrap();
        }
        (2, false) => {
            systems.add_read_only_exclusive(ReadOnly(logic));
            systems.add_edge::<ReadOnly, Downstream>().unwrap();
        }
        (0, true) => {
            systems.add(DefaultRegular(logic));
            systems.add_edge::<DefaultRegular, Downstream>().unwrap();
        }
        (1, true) => {
            systems.add_exclusive(DefaultExclusive(logic));
            systems.add_edge::<DefaultExclusive, Downstream>().unwrap();
        }
        (2, true) => {
            systems.add_read_only_exclusive(DefaultReadOnly(logic));
            systems.add_edge::<DefaultReadOnly, Downstream>().unwrap();
        }
        _ => panic!("invalid scenario"),
    }
}
