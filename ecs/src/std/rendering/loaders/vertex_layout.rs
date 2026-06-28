//! The vertex-layout loader: reads a serialized [`VertexLayout`] (RON) from a
//! file. `Asset = VertexLayout`, so an `AssetHandle<VertexLayout>` resolves to a
//! shared `Arc<VertexLayout>`.
//!
//! A vertex layout is pure data — no GPU residency — so the pipeline is just
//! `File -> [read (IO), deserialize (CPU)]`. Sharing across consumers (so a mesh
//! and a material bind the *same* `Arc<VertexLayout>` for pointer-equality
//! batching) is the job of the
//! [`VertexLayoutManager`](super::super::VertexLayoutManager), which is the sole
//! requester per source.

use redlilium_assets::{
    AnyAsset, AssetError, AssetLoader, AssetPath, AssetSource, AssetStage, Executor, Guid, LoadEnv,
    StageFuture,
};
use redlilium_core::mesh::VertexLayout;
use redlilium_vfs::Vfs;

/// Identity of a vertex-layout asset: a file resolved from `guid` via the DB.
/// Serialized in components / prefabs that bind a layout.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct VertexLayoutSource {
    pub guid: Guid,
}

impl AssetSource for VertexLayoutSource {
    fn file_guid(&self) -> Option<Guid> {
        Some(self.guid)
    }
}

/// Loads a serialized [`VertexLayout`] (RON) from a file.
pub struct VertexLayoutLoader;

impl AssetLoader for VertexLayoutLoader {
    const NAME: &'static str = "vertex_layout";
    const EXTENSIONS: &'static [&'static str] = &["vlayout"];
    type Source = VertexLayoutSource;
    type Asset = VertexLayout;
    type Deps = ();

    fn pipeline(
        _source: &VertexLayoutSource,
        _deps: &(),
        env: &LoadEnv,
    ) -> Vec<Box<dyn AssetStage>> {
        let mut stages: Vec<Box<dyn AssetStage>> = Vec::new();
        // The read is omitted only if the path didn't resolve — then the decode
        // stage fails cleanly (it receives the unit input instead of bytes).
        if let Some(path) = &env.path {
            stages.push(Box::new(ReadFileStage {
                path: path.clone(),
                vfs: env.vfs.clone(),
            }));
        }
        stages.push(Box::new(DeserializeLayoutStage));
        stages
    }
}

/// IO stage: read the layout file's bytes.
struct ReadFileStage {
    path: AssetPath,
    vfs: Vfs,
}

impl AssetStage for ReadFileStage {
    fn executor(&self) -> Executor {
        Executor::Io
    }
    fn run_async(&self, _input: AnyAsset) -> StageFuture {
        let path = self.path.clone();
        let vfs = self.vfs.clone();
        Box::pin(async move {
            let raw = format!("{}/{}", path.mount, path.path);
            let bytes = vfs
                .read(&raw)
                .await
                .map_err(|e| AssetError::Io(e.to_string()))?;
            Ok(Box::new(bytes) as AnyAsset)
        })
    }
}

/// CPU stage: deserialize the RON bytes into a [`VertexLayout`].
struct DeserializeLayoutStage;

impl AssetStage for DeserializeLayoutStage {
    fn executor(&self) -> Executor {
        Executor::Cpu
    }
    fn run_async(&self, input: AnyAsset) -> StageFuture {
        Box::pin(async move {
            let bytes = input
                .downcast::<Vec<u8>>()
                .map_err(|_| AssetError::Decode("vertex_layout: expected file bytes".into()))?;
            let text = std::str::from_utf8(&bytes)
                .map_err(|e| AssetError::Decode(format!("vertex_layout: invalid utf-8: {e}")))?;
            let layout: VertexLayout = ron::from_str(text)
                .map_err(|e| AssetError::Decode(format!("vertex_layout: ron: {e}")))?;
            Ok(Box::new(layout) as AnyAsset)
        })
    }
}

#[cfg(test)]
mod tests {
    use redlilium_core::mesh::VertexLayout;

    /// The on-disk RON form round-trips back to an equal `VertexLayout` — this is
    /// the contract the deserialize stage relies on.
    #[test]
    fn ron_roundtrip() {
        for layout in [
            (*VertexLayout::position_only()).clone(),
            (*VertexLayout::position_normal_uv()).clone(),
            (*VertexLayout::pbr()).clone(),
        ] {
            let text = ron::to_string(&layout).expect("serialize layout to RON");
            let back: VertexLayout = ron::from_str(&text).expect("deserialize layout from RON");
            assert_eq!(layout, back);
        }
    }
}
