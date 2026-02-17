/// Marks an entity as disabled. Disabled entities are automatically
/// filtered from all queries (`Read<T>`, `Write<T>`, `QueryIter`, `ForEach`).
///
/// Use [`disable`](crate::std::hierarchy::disable) / [`enable`](crate::std::hierarchy::enable)
/// to disable/enable entities. These always propagate to children.
///
/// To access disabled entities, use the `_unfiltered` methods on
/// [`Ref<T>`](crate::Ref) / [`RefMut<T>`](crate::RefMut), or query
/// `Read<Disabled>` directly (which is not filtered).
#[derive(Debug, Clone, Copy, PartialEq, Eq, crate::Component)]
pub struct Disabled;

/// Internal marker: entity was disabled by propagation from a parent's
/// [`disable`](crate::std::hierarchy::disable) call, not by the user directly.
///
/// Used by [`enable`](crate::std::hierarchy::enable) to distinguish which
/// children should be re-enabled vs left alone (manually disabled).
///
/// Not part of the public API — users interact with [`Disabled`] only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, crate::Component)]
pub(crate) struct InheritedDisabled;
