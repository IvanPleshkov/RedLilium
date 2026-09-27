//! Shared collision-event lifecycle vocabulary.

/// A transition of an observed contact/intersection pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CollisionPhase {
    /// Tracking began, possibly because events were enabled inside an existing contact.
    Started,
    Stopped(CollisionStopReason),
}

/// Why tracking of a pair ended.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CollisionStopReason {
    Separated,
    /// The current collision groups or body-type rules no longer allow the pair.
    FilteredOut,
    Removed,
    TrackingDisabled,
    /// At least one participant changed its sensor role.
    Reconfigured,
}
