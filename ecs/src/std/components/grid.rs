/// Configuration for the debug grid display.
///
/// Insert as a resource to enable the [`DrawGrid`](crate::std::systems::DrawGrid) system.
pub struct GridConfig {
    /// Center of the grid in world space.
    pub center: [f32; 3],
    /// Size of each grid cell.
    pub cell_size: f32,
    /// Number of cells from center to edge (total grid is 2*half_count cells wide).
    pub half_count: u32,
    /// Grid line color (RGBA).
    pub color: [f32; 4],
}

impl GridConfig {
    /// Create a default grid configuration.
    pub fn new() -> Self {
        Self {
            center: [0.0, 0.0, 0.0],
            cell_size: 1.0,
            half_count: 10,
            color: [0.4, 0.4, 0.4, 1.0],
        }
    }

    /// Set the center position.
    pub fn with_center(mut self, center: [f32; 3]) -> Self {
        self.center = center;
        self
    }

    /// Set the cell size.
    pub fn with_cell_size(mut self, cell_size: f32) -> Self {
        self.cell_size = cell_size;
        self
    }

    /// Set the number of cells from center to edge.
    pub fn with_half_count(mut self, half_count: u32) -> Self {
        self.half_count = half_count;
        self
    }

    /// Set the grid line color.
    pub fn with_color(mut self, color: [f32; 4]) -> Self {
        self.color = color;
        self
    }
}

impl Default for GridConfig {
    fn default() -> Self {
        Self::new()
    }
}
