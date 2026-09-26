# Resource transfers

GPU uploads, copies and readbacks go through `RenderGraph` transfer passes.
Do not add direct upload/write/readback commands to the public `GraphicsDevice`
API. The device creates resources; the graph orders data transfers and derives
their synchronization. Preserve the existing fence-protected ring-buffer
streaming path without turning it into a general device upload API.
