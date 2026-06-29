//! The vertex-layout loader. `Asset = VertexLayout`, so an
//! `AssetHandle<VertexLayout>` resolves to a shared `Arc<VertexLayout>`.
//!
//! A vertex layout has **no file content** — its `.vlayout` file is empty and its
//! parameters live in the DB record's `settings` (RON-encoded `VertexLayout`).
//! The pipeline is therefore a single CPU stage that deserializes
//! [`LoadEnv::settings`]; no IO. Sharing across consumers (so a mesh and a
//! material bind the *same* `Arc<VertexLayout>` for pointer-equality batching) is
//! the job of the
//! [`VertexLayoutManager`](super::super::VertexLayoutManager), the sole requester
//! per source.

use redlilium_assets::{
    AnyAsset, AssetError, AssetLoader, AssetSource, AssetStage, Executor, Guid, LoadEnv,
    StageFuture,
};
use redlilium_core::mesh::VertexLayout;

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

/// Loads a [`VertexLayout`] from its DB record's `settings` (the empty file is
/// only the asset's VFS presence).
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
        vec![Box::new(LayoutFromSettingsStage {
            settings: env.settings.clone(),
        })]
    }
}

/// CPU stage: deserialize the layout from the record's settings (RON).
struct LayoutFromSettingsStage {
    settings: Option<String>,
}

impl AssetStage for LayoutFromSettingsStage {
    fn executor(&self) -> Executor {
        Executor::Cpu
    }
    fn run_async(&self, _input: AnyAsset) -> StageFuture {
        let settings = self.settings.clone();
        Box::pin(async move {
            let text = settings.ok_or_else(|| {
                AssetError::Decode("vertex_layout: no parameters in the DB record".into())
            })?;
            let layout: VertexLayout = ron::from_str(&text)
                .map_err(|e| AssetError::Decode(format!("vertex_layout: ron: {e}")))?;
            Ok(Box::new(layout) as AnyAsset)
        })
    }
}

#[cfg(test)]
mod tests {
    use redlilium_core::mesh::VertexLayout;

    /// The settings RON form round-trips back to an equal `VertexLayout` — this is
    /// the contract the loader relies on.
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
