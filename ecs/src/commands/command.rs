use std::fmt;
use std::panic::{AssertUnwindSafe, Location, catch_unwind};

use crate::World;
use crate::system::panic_payload_to_string;

/// A panic raised while applying a deferred command.
///
/// The location identifies where the command was queued. All strings are owned
/// so the report can outlive the game module that created the command.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CommandError {
    pub message: String,
    pub file: String,
    pub line: u32,
    pub column: u32,
}

impl fmt::Display for CommandError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "command queued at {}:{}:{} panicked: {}",
            self.file, self.line, self.column, self.message
        )
    }
}

impl std::error::Error for CommandError {}

type ApplyFn = Box<dyn FnOnce(&mut World) -> Result<(), CommandError> + Send>;

/// A drained deferred command with its own panic boundary.
///
/// Consume it with [`apply`](Self::apply). The originating module must remain
/// loaded until the command is applied or dropped.
pub struct DeferredCommand {
    apply: ApplyFn,
}

impl DeferredCommand {
    #[track_caller]
    pub(crate) fn new<F: FnOnce(&mut World) + Send + 'static>(command: F) -> Self {
        let location = Location::caller();
        Self {
            // This generic closure is instantiated in the image that queues
            // F. Catch there, before type erasure: catching a guest's panic in
            // the host's libstd can abort with a foreign-exception error.
            apply: Box::new(move |world| {
                catch_unwind(AssertUnwindSafe(|| command(world))).map_err(|payload| CommandError {
                    message: panic_payload_to_string(&*payload),
                    file: location.file().to_owned(),
                    line: location.line(),
                    column: location.column(),
                })
            }),
        }
    }

    /// Applies this command once, returning a panic as an error.
    /// Mutations made before a panic are retained; there is no rollback.
    pub fn apply(self, world: &mut World) -> Result<(), CommandError> {
        (self.apply)(world)
    }
}

pub(crate) fn apply_batch(commands: Vec<DeferredCommand>, world: &mut World) -> Vec<CommandError> {
    commands
        .into_iter()
        .filter_map(|command| command.apply(world).err())
        .collect()
}
