//! The shader loader: reads a self-contained `.slang` source file. `Asset =
//! Shader` (the raw source). A shader's data IS its file content (its DB record
//! has no settings); its guid binds a material to it. Compilation happens later,
//! when a material builds its pipeline from the source.

use redlilium_assets::{
    AnyAsset, AssetError, AssetLoader, AssetPath, AssetSource, AssetStage, Executor, Guid, LoadEnv,
    StageFuture,
};
use redlilium_vfs::Vfs;

/// A loaded shader: its raw source bytes (Slang), shared via the asset system.
/// A material turns this into per-stage `graphics::ShaderSource`s at pipeline
/// creation (one `.slang` file carries multiple entry points, e.g. vs/fs).
#[derive(Debug, Clone)]
pub struct Shader {
    pub source: Vec<u8>,
}

/// Identity of a shader asset: a `.slang` file resolved from `guid` via the DB.
/// Serialized where a material binds its shader.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct ShaderSource {
    pub guid: Guid,
}

impl AssetSource for ShaderSource {
    fn file_guid(&self) -> Option<Guid> {
        Some(self.guid)
    }
}

/// Loads a `.slang` shader's source from a file.
pub struct ShaderLoader;

impl AssetLoader for ShaderLoader {
    const NAME: &'static str = "shader";
    const EXTENSIONS: &'static [&'static str] = &["slang"];
    type Source = ShaderSource;
    type Asset = Shader;
    type Deps = ();

    fn pipeline(_source: &ShaderSource, _deps: &(), env: &LoadEnv) -> Vec<Box<dyn AssetStage>> {
        let mut stages: Vec<Box<dyn AssetStage>> = Vec::new();
        if let Some(path) = &env.path {
            stages.push(Box::new(ReadShaderStage {
                path: path.clone(),
                vfs: env.vfs.clone(),
            }));
        }
        stages
    }
}

/// IO stage: read the shader source bytes (which are the final `Shader`).
struct ReadShaderStage {
    path: AssetPath,
    vfs: Vfs,
}

impl AssetStage for ReadShaderStage {
    fn executor(&self) -> Executor {
        Executor::Io
    }
    fn run_async(&self, _input: AnyAsset) -> StageFuture {
        let path = self.path.clone();
        let vfs = self.vfs.clone();
        Box::pin(async move {
            let raw = format!("{}/{}", path.mount, path.path);
            let source = vfs
                .read(&raw)
                .await
                .map_err(|e| AssetError::Io(e.to_string()))?;
            Ok(Box::new(Shader { source }) as AnyAsset)
        })
    }
}
