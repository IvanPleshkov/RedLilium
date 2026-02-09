use std::future::Future;
use std::pin::Pin;

/// A type-erased system future.
///
/// This is a boxed future returned by [`System::run`](crate::System::run).
/// Each system invocation produces a `SystemFuture` that the runner polls
/// to completion.
pub type SystemFuture<'a> = Pin<Box<dyn Future<Output = ()> + Send + 'a>>;
