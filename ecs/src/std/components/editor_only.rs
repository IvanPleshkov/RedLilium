/// Marks an entity as editor-only — it should not exist during gameplay.
///
/// Editor-only entities are hidden when Play mode starts and restored when
/// Play ends or pauses. They are skipped by `With<EditorOnly>` queries
/// that aren't explicitly interested in editor entities.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, crate::Component)]
pub struct EditorOnly;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn editor_only_is_copy() {
        let e = EditorOnly;
        let _e2 = e;
        let _e3 = e;
    }
}
