use std::any::Any;

/// Result type for condition systems.
///
/// A condition system returns `Condition<T>` to signal whether gated
/// systems should run this tick. [`True(T)`](Condition::True) means the
/// condition is met and carries an optional payload that gated systems
/// can read via [`SystemContext::system_result()`]. [`False`](Condition::False)
/// means the condition is not met and gated systems will be skipped.
///
/// # Example
///
/// ```ignore
/// struct IsPlaying;
///
/// impl System for IsPlaying {
///     type Result = Condition<()>;
///     fn run<'a>(&'a self, ctx: &'a SystemContext<'a>) -> Result<Condition<()>, SystemError> {
///         let playing = ctx.lock::<(Res<GameState>,)>()
///             .execute(|(state,)| *state == GameState::Playing);
///         if playing {
///             Ok(Condition::True(()))
///         } else {
///             Ok(Condition::False)
///         }
///     }
/// }
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Condition<T = ()> {
    /// Condition is met. The payload `T` is accessible to gated systems.
    True(T),
    /// Condition is not met. Gated systems will be skipped.
    False,
}

impl<T> Condition<T> {
    /// Returns `true` if the condition is met.
    pub fn is_true(&self) -> bool {
        matches!(self, Self::True(_))
    }

    /// Returns `true` if the condition is not met.
    pub fn is_false(&self) -> bool {
        matches!(self, Self::False)
    }

    /// Returns a reference to the payload if the condition is met.
    pub fn value(&self) -> Option<&T> {
        match self {
            Self::True(v) => Some(v),
            Self::False => None,
        }
    }

    /// Consumes the condition and returns the payload if met.
    pub fn into_value(self) -> Option<T> {
        match self {
            Self::True(v) => Some(v),
            Self::False => None,
        }
    }
}

/// Trait implemented by condition result types.
///
/// This is automatically implemented for all [`Condition<T>`] types.
/// Used internally by the runner to evaluate whether gated systems should run.
pub trait ConditionResult: Send + Sync + 'static {
    /// Returns whether the condition is met.
    fn passed(&self) -> bool;
}

impl<T: Send + Sync + 'static> ConditionResult for Condition<T> {
    fn passed(&self) -> bool {
        self.is_true()
    }
}

/// How multiple run conditions on a single system are combined.
///
/// When a system has multiple condition predecessors, this mode
/// determines how their results are combined.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConditionMode {
    /// All conditions must pass for the system to run (default).
    All,
    /// At least one condition must pass for the system to run.
    Any,
}

/// Type-erased condition checker function.
///
/// Downcasts the boxed result to `R` and calls [`ConditionResult::passed()`].
/// Returns `false` if the downcast fails (should not happen with correct types).
pub(crate) fn condition_checker<R: ConditionResult + 'static>(
    result: &(dyn Any + Send + Sync),
) -> bool {
    result.downcast_ref::<R>().is_some_and(|r| r.passed())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn condition_true_is_true() {
        let c = Condition::True(42);
        assert!(c.is_true());
        assert!(!c.is_false());
        assert_eq!(c.value(), Some(&42));
    }

    #[test]
    fn condition_false_is_false() {
        let c: Condition<i32> = Condition::False;
        assert!(!c.is_true());
        assert!(c.is_false());
        assert_eq!(c.value(), None);
    }

    #[test]
    fn condition_into_value() {
        assert_eq!(Condition::True("hello").into_value(), Some("hello"));
        assert_eq!(Condition::<&str>::False.into_value(), None);
    }

    #[test]
    fn condition_result_trait() {
        let t: Condition<()> = Condition::True(());
        assert!(t.passed());

        let f: Condition<()> = Condition::False;
        assert!(!f.passed());
    }

    #[test]
    fn condition_checker_fn() {
        let result: Box<dyn Any + Send + Sync> = Box::new(Condition::True(42u32));
        assert!(condition_checker::<Condition<u32>>(result.as_ref()));

        let result: Box<dyn Any + Send + Sync> = Box::new(Condition::<u32>::False);
        assert!(!condition_checker::<Condition<u32>>(result.as_ref()));
    }

    #[test]
    fn condition_checker_wrong_type_returns_false() {
        let result: Box<dyn Any + Send + Sync> = Box::new(42u32);
        assert!(!condition_checker::<Condition<u32>>(result.as_ref()));
    }

    #[test]
    fn condition_unit_default() {
        let c: Condition = Condition::True(());
        assert!(c.is_true());

        let c: Condition = Condition::False;
        assert!(c.is_false());
    }

    #[test]
    fn condition_clone_copy() {
        let c = Condition::True(5u32);
        let c2 = c;
        assert_eq!(c, c2);
    }

    #[test]
    fn condition_debug() {
        let c = Condition::True(42);
        let s = format!("{c:?}");
        assert!(s.contains("True"));
        assert!(s.contains("42"));
    }
}
