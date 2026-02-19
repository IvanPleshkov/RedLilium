use crate::drawer::DebugDrawerContext;

impl DebugDrawerContext<'_> {
    /// Draw a single line segment.
    pub fn draw_line(&mut self, start: [f32; 3], end: [f32; 3], color: [f32; 4]) {
        self.push_line(start, end, color);
    }

    /// Draw an axis-aligned bounding box from min to max corners (12 edges).
    pub fn draw_aabb(&mut self, min: [f32; 3], max: [f32; 3], color: [f32; 4]) {
        let c = [
            [min[0], min[1], min[2]],
            [max[0], min[1], min[2]],
            [max[0], max[1], min[2]],
            [min[0], max[1], min[2]],
            [min[0], min[1], max[2]],
            [max[0], min[1], max[2]],
            [max[0], max[1], max[2]],
            [min[0], max[1], max[2]],
        ];
        // Bottom face
        self.push_line(c[0], c[1], color);
        self.push_line(c[1], c[2], color);
        self.push_line(c[2], c[3], color);
        self.push_line(c[3], c[0], color);
        // Top face
        self.push_line(c[4], c[5], color);
        self.push_line(c[5], c[6], color);
        self.push_line(c[6], c[7], color);
        self.push_line(c[7], c[4], color);
        // Vertical edges
        self.push_line(c[0], c[4], color);
        self.push_line(c[1], c[5], color);
        self.push_line(c[2], c[6], color);
        self.push_line(c[3], c[7], color);
    }

    /// Draw an oriented box given center, half-extents, and rotation columns.
    ///
    /// `rotation` is a 3x3 rotation matrix given as three column vectors.
    pub fn draw_box(
        &mut self,
        center: [f32; 3],
        half_extents: [f32; 3],
        rotation: [[f32; 3]; 3],
        color: [f32; 4],
    ) {
        let corner = |sx: f32, sy: f32, sz: f32| -> [f32; 3] {
            let lx = half_extents[0] * sx;
            let ly = half_extents[1] * sy;
            let lz = half_extents[2] * sz;
            [
                center[0] + rotation[0][0] * lx + rotation[1][0] * ly + rotation[2][0] * lz,
                center[1] + rotation[0][1] * lx + rotation[1][1] * ly + rotation[2][1] * lz,
                center[2] + rotation[0][2] * lx + rotation[1][2] * ly + rotation[2][2] * lz,
            ]
        };

        let c = [
            corner(-1.0, -1.0, -1.0),
            corner(1.0, -1.0, -1.0),
            corner(1.0, 1.0, -1.0),
            corner(-1.0, 1.0, -1.0),
            corner(-1.0, -1.0, 1.0),
            corner(1.0, -1.0, 1.0),
            corner(1.0, 1.0, 1.0),
            corner(-1.0, 1.0, 1.0),
        ];

        // Bottom face
        self.push_line(c[0], c[1], color);
        self.push_line(c[1], c[2], color);
        self.push_line(c[2], c[3], color);
        self.push_line(c[3], c[0], color);
        // Top face
        self.push_line(c[4], c[5], color);
        self.push_line(c[5], c[6], color);
        self.push_line(c[6], c[7], color);
        self.push_line(c[7], c[4], color);
        // Vertical edges
        self.push_line(c[0], c[4], color);
        self.push_line(c[1], c[5], color);
        self.push_line(c[2], c[6], color);
        self.push_line(c[3], c[7], color);
    }

    /// Draw a circle in a plane defined by two orthogonal axes.
    pub fn draw_circle(
        &mut self,
        center: [f32; 3],
        radius: f32,
        axis_a: [f32; 3],
        axis_b: [f32; 3],
        color: [f32; 4],
    ) {
        const SEGMENTS: usize = 32;
        let mut prev = [
            center[0] + axis_a[0] * radius,
            center[1] + axis_a[1] * radius,
            center[2] + axis_a[2] * radius,
        ];

        for i in 1..=SEGMENTS {
            let angle = (i as f32) * std::f32::consts::TAU / SEGMENTS as f32;
            let (sin, cos) = angle.sin_cos();
            let point = [
                center[0] + (axis_a[0] * cos + axis_b[0] * sin) * radius,
                center[1] + (axis_a[1] * cos + axis_b[1] * sin) * radius,
                center[2] + (axis_a[2] * cos + axis_b[2] * sin) * radius,
            ];
            self.push_line(prev, point, color);
            prev = point;
        }
    }

    /// Draw a wireframe sphere using 3 great circles (XY, XZ, YZ planes).
    pub fn draw_sphere(&mut self, center: [f32; 3], radius: f32, color: [f32; 4]) {
        self.draw_circle(center, radius, [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], color);
        self.draw_circle(center, radius, [1.0, 0.0, 0.0], [0.0, 0.0, 1.0], color);
        self.draw_circle(center, radius, [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], color);
    }

    /// Draw a cross marker at a point (3 axis-aligned lines).
    pub fn draw_cross(&mut self, center: [f32; 3], size: f32, color: [f32; 4]) {
        let half = size * 0.5;
        self.push_line(
            [center[0] - half, center[1], center[2]],
            [center[0] + half, center[1], center[2]],
            color,
        );
        self.push_line(
            [center[0], center[1] - half, center[2]],
            [center[0], center[1] + half, center[2]],
            color,
        );
        self.push_line(
            [center[0], center[1], center[2] - half],
            [center[0], center[1], center[2] + half],
            color,
        );
    }

    /// Draw a ray from origin in a direction with a given length.
    pub fn draw_ray(
        &mut self,
        origin: [f32; 3],
        direction: [f32; 3],
        length: f32,
        color: [f32; 4],
    ) {
        let end = [
            origin[0] + direction[0] * length,
            origin[1] + direction[1] * length,
            origin[2] + direction[2] * length,
        ];
        self.push_line(origin, end, color);
    }

    /// Draw coordinate axes gizmo (R=X, G=Y, B=Z).
    pub fn draw_axes(&mut self, center: [f32; 3], size: f32) {
        self.draw_ray(center, [1.0, 0.0, 0.0], size, [1.0, 0.0, 0.0, 1.0]);
        self.draw_ray(center, [0.0, 1.0, 0.0], size, [0.0, 1.0, 0.0, 1.0]);
        self.draw_ray(center, [0.0, 0.0, 1.0], size, [0.0, 0.0, 1.0, 1.0]);
    }

    /// Draw a wireframe frustum from 8 corners.
    ///
    /// Corner order: near plane `[TL, TR, BR, BL]`, far plane `[TL, TR, BR, BL]`.
    pub fn draw_frustum(&mut self, corners: &[[f32; 3]; 8], color: [f32; 4]) {
        // Near face
        self.push_line(corners[0], corners[1], color);
        self.push_line(corners[1], corners[2], color);
        self.push_line(corners[2], corners[3], color);
        self.push_line(corners[3], corners[0], color);
        // Far face
        self.push_line(corners[4], corners[5], color);
        self.push_line(corners[5], corners[6], color);
        self.push_line(corners[6], corners[7], color);
        self.push_line(corners[7], corners[4], color);
        // Connecting edges
        self.push_line(corners[0], corners[4], color);
        self.push_line(corners[1], corners[5], color);
        self.push_line(corners[2], corners[6], color);
        self.push_line(corners[3], corners[7], color);
    }

    /// Draw a grid on the XZ plane.
    pub fn draw_grid(
        &mut self,
        center: [f32; 3],
        cell_size: f32,
        half_count: u32,
        color: [f32; 4],
    ) {
        let extent = cell_size * half_count as f32;
        let n = half_count as i32;

        for i in -n..=n {
            let offset = i as f32 * cell_size;
            // Line along Z
            self.push_line(
                [center[0] + offset, center[1], center[2] - extent],
                [center[0] + offset, center[1], center[2] + extent],
                color,
            );
            // Line along X
            self.push_line(
                [center[0] - extent, center[1], center[2] + offset],
                [center[0] + extent, center[1], center[2] + offset],
                color,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::drawer::DebugDrawer;

    #[test]
    fn test_draw_line() {
        let drawer = DebugDrawer::new();
        {
            let mut ctx = drawer.context();
            ctx.draw_line([0.0; 3], [1.0; 3], [1.0; 4]);
        }
        drawer.advance_tick();
        let data = drawer.take_render_data();
        assert_eq!(data.len(), 2);
    }

    #[test]
    fn test_draw_aabb() {
        let drawer = DebugDrawer::new();
        {
            let mut ctx = drawer.context();
            ctx.draw_aabb([-1.0; 3], [1.0; 3], [1.0; 4]);
        }
        drawer.advance_tick();
        let data = drawer.take_render_data();
        assert_eq!(data.len(), 24); // 12 edges * 2 vertices
    }

    #[test]
    fn test_draw_sphere() {
        let drawer = DebugDrawer::new();
        {
            let mut ctx = drawer.context();
            ctx.draw_sphere([0.0; 3], 1.0, [1.0; 4]);
        }
        drawer.advance_tick();
        let data = drawer.take_render_data();
        // 3 circles * 32 segments * 2 vertices = 192
        assert_eq!(data.len(), 192);
    }

    #[test]
    fn test_draw_grid() {
        let drawer = DebugDrawer::new();
        {
            let mut ctx = drawer.context();
            ctx.draw_grid([0.0; 3], 1.0, 2, [0.5; 4]);
        }
        drawer.advance_tick();
        let data = drawer.take_render_data();
        // half_count=2: -2..=2 = 5 lines per axis * 2 axes = 10 lines * 2 vertices = 20
        assert_eq!(data.len(), 20);
    }
}
