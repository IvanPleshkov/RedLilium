/// Priority level for task execution.
///
/// Higher priority tasks are executed before lower priority tasks.
/// Sync ECS systems always run at Critical priority.
///
/// # Ordering
///
/// `Critical > High > Low` — derives `Ord` so priorities can be compared
/// and sorted directly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Priority {
    /// Fills gaps when higher-priority work is unavailable.
    /// May span multiple frames.
    Low,
    /// Should complete this frame. Used for important async tasks.
    High,
    /// Must complete this frame. Used for ECS systems.
    Critical,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn priority_ordering() {
        assert!(Priority::Critical > Priority::High);
        assert!(Priority::High > Priority::Low);
        assert!(Priority::Critical > Priority::Low);
    }
}
