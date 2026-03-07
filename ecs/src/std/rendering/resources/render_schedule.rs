//! Frame schedule resource.

use redlilium_graphics::FrameSchedule;

/// Resource wrapping a [`FrameSchedule`] for the current frame.
///
/// The application layer inserts this before running ECS systems and
/// extracts it after, using [`take`](Self::take).
pub struct RenderSchedule {
    schedule: Option<FrameSchedule>,
}

impl RenderSchedule {
    /// Create a new render schedule resource holding the given frame schedule.
    pub fn new(schedule: FrameSchedule) -> Self {
        Self {
            schedule: Some(schedule),
        }
    }

    /// Create an empty render schedule (no active frame).
    pub fn empty() -> Self {
        Self { schedule: None }
    }

    /// Take the frame schedule out, leaving this resource empty.
    pub fn take(&mut self) -> Option<FrameSchedule> {
        self.schedule.take()
    }

    /// Replace the current schedule with a new one.
    pub fn set(&mut self, schedule: FrameSchedule) {
        self.schedule = Some(schedule);
    }

    /// Get a reference to the frame schedule, if present.
    pub fn schedule(&self) -> Option<&FrameSchedule> {
        self.schedule.as_ref()
    }

    /// Get a mutable reference to the frame schedule, if present.
    pub fn schedule_mut(&mut self) -> Option<&mut FrameSchedule> {
        self.schedule.as_mut()
    }

    /// Returns `true` if a frame schedule is currently held.
    pub fn is_active(&self) -> bool {
        self.schedule.is_some()
    }
}
