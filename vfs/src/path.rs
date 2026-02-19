use crate::VfsError;

/// Normalize a VFS path.
///
/// - Replaces backslashes with forward slashes
/// - Collapses redundant separators (`a///b` → `a/b`)
/// - Drops `.` segments
/// - Rejects `..` segments (path traversal not allowed)
/// - Strips leading and trailing slashes
///
/// Returns `Err(VfsError::InvalidPath)` if the path is empty or contains `..`.
pub fn normalize(path: &str) -> Result<String, VfsError> {
    let replaced = path.replace('\\', "/");
    let mut segments = Vec::new();

    for segment in replaced.split('/') {
        if segment.is_empty() || segment == "." {
            continue;
        }
        if segment == ".." {
            return Err(VfsError::InvalidPath(
                "path traversal (..) not allowed".into(),
            ));
        }
        segments.push(segment);
    }

    if segments.is_empty() {
        return Err(VfsError::InvalidPath("empty path".into()));
    }

    Ok(segments.join("/"))
}

/// Split a normalized path into source name and remainder.
///
/// Returns `(source, rest)` where `source` is the first path segment
/// and `rest` is everything after it. If there is only one segment,
/// `rest` is empty.
///
/// The path must already be normalized (no leading slash, no `..`).
pub(crate) fn split_source(path: &str) -> (&str, &str) {
    match path.find('/') {
        Some(pos) => (&path[..pos], &path[pos + 1..]),
        None => (path, ""),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simple_path() {
        assert_eq!(
            normalize("textures/brick.png").unwrap(),
            "textures/brick.png"
        );
    }

    #[test]
    fn leading_slash() {
        assert_eq!(
            normalize("/textures/brick.png").unwrap(),
            "textures/brick.png"
        );
    }

    #[test]
    fn trailing_slash() {
        assert_eq!(normalize("textures/").unwrap(), "textures");
    }

    #[test]
    fn redundant_slashes() {
        assert_eq!(
            normalize("textures///brick.png").unwrap(),
            "textures/brick.png"
        );
    }

    #[test]
    fn dot_segments() {
        assert_eq!(
            normalize("textures/./brick.png").unwrap(),
            "textures/brick.png"
        );
    }

    #[test]
    fn backslashes() {
        assert_eq!(
            normalize("textures\\brick.png").unwrap(),
            "textures/brick.png"
        );
    }

    #[test]
    fn reject_dotdot() {
        assert!(normalize("textures/../secret.txt").is_err());
    }

    #[test]
    fn reject_empty() {
        assert!(normalize("").is_err());
    }

    #[test]
    fn reject_only_slashes() {
        assert!(normalize("///").is_err());
    }

    #[test]
    fn reject_only_dots() {
        assert!(normalize("././.").is_err());
    }

    #[test]
    fn single_segment() {
        assert_eq!(normalize("file.txt").unwrap(), "file.txt");
    }

    #[test]
    fn deeply_nested() {
        assert_eq!(normalize("a/b/c/d/e.txt").unwrap(), "a/b/c/d/e.txt");
    }

    #[test]
    fn split_source_with_rest() {
        assert_eq!(
            split_source("assets/textures/brick.png"),
            ("assets", "textures/brick.png")
        );
    }

    #[test]
    fn split_source_no_rest() {
        assert_eq!(split_source("file.txt"), ("file.txt", ""));
    }
}
