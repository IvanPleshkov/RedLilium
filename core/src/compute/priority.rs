/// Priority level for task execution.
///
/// Higher priority tasks are executed before lower priority tasks.
/// These priorities order compute work; ready ECS systems take precedence
/// over background compute regardless of its priority.
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
    /// Important async tasks, ahead of low-priority work.
    High,
    /// Highest-priority compute work. Frame completion is not guaranteed.
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
