//! Shared options and errors for read-only spatial queries.

/// Whether a ray starting inside a collider immediately hits it or seeks an exit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RayCastOptions {
    pub solid: bool,
}
impl Default for RayCastOptions {
    fn default() -> Self {
        Self { solid: true }
    }
}

/// Linear shape sweep options. All lengths use physics world units.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ShapeCastOptions {
    /// Finite, nonnegative clearance at which to report a hit.
    pub target_distance: f32,
    /// Report initial contact even when motion separates the shapes. If false,
    /// only separating initial contacts may be skipped, not all initial overlaps.
    pub stop_at_penetration: bool,
}
impl Default for ShapeCastOptions {
    fn default() -> Self {
        Self {
            target_distance: 0.0,
            stop_at_penetration: true,
        }
    }
}

/// How the backend obtained a shape sweep result. Approximate hits remain hits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ShapeCastStatus {
    Converged,
    /// Initially intersecting or within the requested clearance.
    InitialContact,
    /// Conservative approximation after reaching the iteration limit.
    OutOfIterations,
    /// Backend numerical failure with a conservative hit estimate.
    Failed,
}

/// A malformed query or a backend result that cannot be represented by this API.
/// Errors are separate from a valid query that finds no collider.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PhysicsQueryError {
    InvalidOrigin,
    InvalidDisplacement,
    InvalidPose,
    InvalidShape,
    InvalidTargetDistance,
    InvalidResult,
}
impl std::fmt::Display for PhysicsQueryError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            Self::InvalidOrigin => "query origin must be finite",
            Self::InvalidDisplacement => {
                "query displacement must be finite and nonzero, with a finite endpoint"
            }
            Self::InvalidPose => "query pose must be finite with a valid rotation",
            Self::InvalidShape => "query shape dimensions must be finite and valid",
            Self::InvalidTargetDistance => "query target distance must be finite and nonnegative",
            Self::InvalidResult => "physics query returned invalid or unrepresentable geometry",
        })
    }
}
impl std::error::Error for PhysicsQueryError {}
