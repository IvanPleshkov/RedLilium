//! [`AssetRef`] as a [`ComponentField`]: components can hold asset references
//! as plain derived fields — serialization writes only the source, the
//! inspector shows the source + load state, and the visit hooks yield the ref
//! to the generic asset-sync system (which resolves it demand-driven and on hot
//! reload). This is what makes a user component with an `AssetRef` field work
//! with zero extra code.

use std::any::Any;

use redlilium_assets::{AssetRef, AssetRefSource};

use crate::ComponentField;
use crate::serialize::{DeserializeContext, DeserializeError, SerializeContext, SerializeError};

impl<S> ComponentField for AssetRef<S>
where
    S: AssetRefSource
        + Clone
        + std::fmt::Debug
        + serde::Serialize
        + serde::de::DeserializeOwned
        + Send
        + Sync
        + 'static,
    S::Asset: Send + Sync,
{
    fn inspect_field(&self, name: &str, ui: &mut egui::Ui) -> Option<Self> {
        // Identity + load state, read-only. Editing the identity (e.g. dropping
        // an asset from the browser) is a future step.
        ui.horizontal(|ui| {
            ui.label(name);
            let source = format!("{:?}", self.source());
            if self.get().is_some() {
                ui.monospace(source);
            } else {
                ui.weak(format!("{source} (loading…)"));
            }
        });
        None
    }

    fn serialize_field(
        &self,
        name: &str,
        ctx: &mut SerializeContext<'_>,
    ) -> Result<(), SerializeError> {
        // Only the identity is serialized; the resolution is runtime state.
        ctx.write_serde(name, self.source())
    }

    fn deserialize_field(
        name: &str,
        ctx: &mut DeserializeContext<'_>,
    ) -> Result<Self, DeserializeError> {
        // Starts unresolved; the sync system requests + resolves it on sight.
        Ok(AssetRef::new(ctx.read_serde(name)?))
    }

    fn visit_asset_refs(&self, f: &mut dyn FnMut(&dyn Any)) {
        f(self);
    }

    fn visit_asset_refs_mut(&mut self, f: &mut dyn FnMut(&mut dyn Any)) {
        f(self);
    }
}
