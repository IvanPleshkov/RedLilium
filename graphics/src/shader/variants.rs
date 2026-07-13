//! Shader variant spaces and keys (#6, `docs/MATERIAL_ASSETS.md` Decision 5).
//!
//! The universal pattern every engine converged on: **the shader declares its
//! variant space, selection is split between the material and the render
//! pipeline, and caches key by the full variant.** This module is the
//! graphics-level API for all three:
//!
//! - [`ShaderVariantSpace::parse`] reads the space from `//#pragma variant`
//!   lines in the `.slang` source — the space physically travels with the
//!   source, so the offline bake (`xtask bake-shaders`) and the runtime can
//!   never drift apart.
//! - [`ShaderVariantSpace::select`] builds a validated [`VariantKey`]: a typo
//!   in an axis name is an error at key-build time, not a shader silently
//!   compiled without the feature.
//! - [`VariantKey`] is canonical (sorted, deduped) and feeds
//!   [`MaterialDescriptor::with_variant`](crate::materials::MaterialDescriptor::with_variant)
//!   and pipeline-cache keys.
//!
//! # Pragma grammar
//!
//! ```text
//! //#pragma variant HAS_NORMAL_MAP              bool feature, default off
//! //#pragma variant HAS_NORMAL_MAP default 1    bool feature, default on
//! //#pragma variant QUALITY 0 1 2               enum feature, default = first value
//! //#pragma variant QUALITY 0 1 2 default 1     enum feature, explicit default
//! //#pragma variant_system HDR_OUTPUT           bool, selected by the render pipeline
//! //#pragma variant_system MODE 0 1             enum, selected by the render pipeline
//! ```
//!
//! `variant` axes are **features** — selected by the material, always carrying
//! a default so call sites only mention what they turn on. `variant_system`
//! axes are selected by the **render pipeline** (pass, quality, color space)
//! and are deliberately default-free: every call site must set them
//! explicitly, so a pipeline cannot "forget" an axis.
//!
//! # Define emission
//!
//! A boolean axis set to `true` emits a value-less define `(NAME, "")` and to
//! `false` emits **nothing** — matching plain `#ifdef NAME` in the shader and
//! the pre-existing ad-hoc keys (an empty selection hashes identically to the
//! historical "no defines" bake). An enum axis always emits `(NAME, value)`
//! for `#if NAME == …`.

use std::collections::HashSet;
use std::fmt;

/// Hard cap on the variant count of one shader. Exceeding it fails the bake
/// loudly instead of silently exploding the baked table — combinatorial growth
/// must be a reviewed decision, not an accident.
pub const MAX_VARIANTS_PER_SHADER: usize = 64;

/// Who selects an axis (Decision 5's split ownership).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VariantSelector {
    /// Selected by the material (`//#pragma variant`); has a default.
    Feature,
    /// Selected by the render pipeline (`//#pragma variant_system`); no
    /// default — must be set explicitly on every key build.
    System,
}

/// Value shape of an axis.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VariantAxisKind {
    /// `#ifdef`-style flag. `true` emits `(name, "")`, `false` emits nothing.
    Bool {
        /// Default for feature axes (system axes never consult it).
        default: bool,
    },
    /// `#if NAME == …`-style value; always emitted as `(name, value)`.
    Enum {
        /// Allowed values, in declaration order.
        values: Vec<String>,
        /// Default for feature axes (system axes never consult it).
        default: String,
    },
}

/// One declared variant axis.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VariantAxis {
    /// Define name (`[A-Za-z_][A-Za-z0-9_]*`).
    pub name: String,
    /// Value shape and default.
    pub kind: VariantAxisKind,
    /// Who selects it.
    pub selector: VariantSelector,
}

impl VariantAxis {
    fn value_count(&self) -> usize {
        match &self.kind {
            VariantAxisKind::Bool { .. } => 2,
            VariantAxisKind::Enum { values, .. } => values.len(),
        }
    }
}

/// The variant space one shader declares (possibly empty).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ShaderVariantSpace {
    /// Axes sorted by name (deterministic enumeration and diagnostics).
    axes: Vec<VariantAxis>,
}

/// Errors from parsing a space or building a key. Every message names the
/// axis involved — a typo must be diagnosable from the error alone.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VariantError {
    /// A `//#pragma variant…` line failed to parse.
    BadPragma { line: usize, reason: String },
    /// The same axis was declared twice.
    DuplicateAxis(String),
    /// A selection named an axis the space does not declare.
    UnknownAxis(String),
    /// A feature axis was set via `.system()` or vice versa.
    SelectorMismatch {
        axis: String,
        expected: VariantSelector,
    },
    /// The value does not fit the axis (bool vs enum, or not an allowed
    /// enum value).
    InvalidValue { axis: String, value: String },
    /// The same axis was set twice in one selection.
    DuplicateSelection(String),
    /// A system axis was never set (system axes have no defaults).
    MissingSystemAxis(String),
    /// Cartesian enumeration exceeds [`MAX_VARIANTS_PER_SHADER`].
    TooManyVariants { count: usize },
}

impl fmt::Display for VariantError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BadPragma { line, reason } => {
                write!(f, "bad variant pragma on line {line}: {reason}")
            }
            Self::DuplicateAxis(name) => write!(f, "variant axis {name:?} declared twice"),
            Self::UnknownAxis(name) => {
                write!(
                    f,
                    "unknown variant axis {name:?} (not declared by the shader)"
                )
            }
            Self::SelectorMismatch { axis, expected } => write!(
                f,
                "variant axis {axis:?} is selected by the {}",
                match expected {
                    VariantSelector::Feature => "material (use .feature())",
                    VariantSelector::System => "render pipeline (use .system())",
                }
            ),
            Self::InvalidValue { axis, value } => {
                write!(f, "value {value:?} is not valid for variant axis {axis:?}")
            }
            Self::DuplicateSelection(name) => {
                write!(f, "variant axis {name:?} set twice in one selection")
            }
            Self::MissingSystemAxis(name) => write!(
                f,
                "system variant axis {name:?} not set (system axes have no defaults; \
                 every render-pipeline call site must set them explicitly)"
            ),
            Self::TooManyVariants { count } => write!(
                f,
                "shader declares {count} variant combinations, above the cap of \
                 {MAX_VARIANTS_PER_SHADER}; growing the space must be a reviewed decision"
            ),
        }
    }
}

impl std::error::Error for VariantError {}

fn is_ident(s: &str) -> bool {
    let mut chars = s.chars();
    matches!(chars.next(), Some(c) if c.is_ascii_alphabetic() || c == '_')
        && chars.all(|c| c.is_ascii_alphanumeric() || c == '_')
}

impl ShaderVariantSpace {
    /// Parse the variant space from `//#pragma variant` / `//#pragma
    /// variant_system` comment lines in a shader source. A shader with no
    /// such pragmas has an empty space (exactly one variant: the empty key).
    pub fn parse(source: &str) -> Result<Self, VariantError> {
        let mut axes: Vec<VariantAxis> = Vec::new();

        for (i, raw_line) in source.lines().enumerate() {
            let line = i + 1;
            let Some(rest) = raw_line.trim_start().strip_prefix("//") else {
                continue;
            };
            let Some(rest) = rest.trim_start().strip_prefix("#pragma") else {
                continue;
            };
            let mut tokens = rest.split_whitespace();
            let selector = match tokens.next() {
                Some("variant") => VariantSelector::Feature,
                Some("variant_system") => VariantSelector::System,
                _ => continue, // not our pragma
            };

            let name = tokens
                .next()
                .ok_or_else(|| VariantError::BadPragma {
                    line,
                    reason: "missing axis name".to_string(),
                })?
                .to_string();
            if !is_ident(&name) {
                return Err(VariantError::BadPragma {
                    line,
                    reason: format!("axis name {name:?} is not an identifier"),
                });
            }
            if axes.iter().any(|a| a.name == name) {
                return Err(VariantError::DuplicateAxis(name));
            }

            // Remaining tokens: enum values, optionally terminated by
            // `default <value>`.
            let mut values: Vec<String> = Vec::new();
            let mut default: Option<String> = None;
            while let Some(tok) = tokens.next() {
                if tok == "default" {
                    let val = tokens.next().ok_or_else(|| VariantError::BadPragma {
                        line,
                        reason: "`default` without a value".to_string(),
                    })?;
                    if tokens.next().is_some() {
                        return Err(VariantError::BadPragma {
                            line,
                            reason: "`default <value>` must be the last tokens".to_string(),
                        });
                    }
                    default = Some(val.to_string());
                    break;
                }
                if values.contains(&tok.to_string()) {
                    return Err(VariantError::BadPragma {
                        line,
                        reason: format!("duplicate enum value {tok:?}"),
                    });
                }
                values.push(tok.to_string());
            }

            if selector == VariantSelector::System && default.is_some() {
                return Err(VariantError::BadPragma {
                    line,
                    reason: format!(
                        "system axis {name:?} must not declare a default \
                         (system axes are set explicitly at every call site)"
                    ),
                });
            }

            let kind = if values.is_empty() {
                let default = match default.as_deref() {
                    None | Some("0") => false,
                    Some("1") => true,
                    Some(other) => {
                        return Err(VariantError::BadPragma {
                            line,
                            reason: format!("bool default must be 0 or 1, got {other:?}"),
                        });
                    }
                };
                VariantAxisKind::Bool { default }
            } else {
                let default = match default {
                    Some(d) if values.contains(&d) => d,
                    Some(d) => {
                        return Err(VariantError::BadPragma {
                            line,
                            reason: format!("default {d:?} is not among the declared values"),
                        });
                    }
                    None => values[0].clone(),
                };
                VariantAxisKind::Enum { values, default }
            };

            axes.push(VariantAxis {
                name,
                kind,
                selector,
            });
        }

        axes.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(Self { axes })
    }

    /// Whether the shader declares no axes.
    pub fn is_empty(&self) -> bool {
        self.axes.is_empty()
    }

    /// The declared axes, sorted by name.
    pub fn axes(&self) -> &[VariantAxis] {
        &self.axes
    }

    /// Start building a validated [`VariantKey`] against this space.
    pub fn select(&self) -> VariantSelection<'_> {
        VariantSelection {
            space: self,
            set: Vec::new(),
        }
    }

    /// Build the **material's feature half** of a variant key (Decision 5's
    /// split): validates and canonicalizes the given feature-axis selections
    /// (feature defaults applied), while system axes are left unset — the
    /// render pipeline supplies them at draw time via
    /// [`VariantSelection::with_features`] + `.system(…)` + `build()`.
    ///
    /// Naming a system axis here is a [`VariantError::SelectorMismatch`];
    /// unknown axes and invalid values error as usual — a typo in a material
    /// asset is diagnosed at resolve time, not silently ignored.
    pub fn build_features(
        &self,
        features: &[(String, VariantValue)],
    ) -> Result<VariantKey, VariantError> {
        let mut selection = self.select();
        for (name, value) in features {
            selection = selection.feature(name, value.clone());
        }
        selection.build_feature_half()
    }

    /// Every variant of this space (cartesian product over all axes), for the
    /// offline bake. Errors above [`MAX_VARIANTS_PER_SHADER`].
    ///
    /// An empty space yields exactly one empty key — the historical
    /// "no defines" permutation.
    pub fn enumerate_all(&self) -> Result<Vec<VariantKey>, VariantError> {
        let count = self
            .axes
            .iter()
            .map(VariantAxis::value_count)
            .product::<usize>();
        if count > MAX_VARIANTS_PER_SHADER {
            return Err(VariantError::TooManyVariants { count });
        }

        let mut keys = vec![Vec::new()];
        for axis in &self.axes {
            let mut next = Vec::with_capacity(keys.len() * axis.value_count());
            for base in &keys {
                match &axis.kind {
                    VariantAxisKind::Bool { .. } => {
                        next.push(base.clone()); // false: no define
                        let mut on = base.clone();
                        on.push((axis.name.clone(), String::new()));
                        next.push(on);
                    }
                    VariantAxisKind::Enum { values, .. } => {
                        for v in values {
                            let mut with = base.clone();
                            with.push((axis.name.clone(), v.clone()));
                            next.push(with);
                        }
                    }
                }
            }
            keys = next;
        }

        Ok(keys
            .into_iter()
            .map(|mut defines| {
                defines.sort();
                VariantKey { defines }
            })
            .collect())
    }
}

/// A value being assigned to an axis. Constructed via `Into` from `bool`
/// (bool axes) or integers / strings (enum axes).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum VariantValue {
    /// For bool axes.
    Bool(bool),
    /// For enum axes (the token as declared in the pragma).
    Value(String),
}

impl From<bool> for VariantValue {
    fn from(v: bool) -> Self {
        Self::Bool(v)
    }
}
impl From<u32> for VariantValue {
    fn from(v: u32) -> Self {
        Self::Value(v.to_string())
    }
}
impl From<i32> for VariantValue {
    fn from(v: i32) -> Self {
        Self::Value(v.to_string())
    }
}
impl From<&str> for VariantValue {
    fn from(v: &str) -> Self {
        Self::Value(v.to_string())
    }
}

/// Builder for a [`VariantKey`]; validation happens in [`build`](Self::build)
/// so calls chain without `Result` noise.
pub struct VariantSelection<'a> {
    space: &'a ShaderVariantSpace,
    set: Vec<(String, VariantValue, VariantSelector)>,
}

impl VariantSelection<'_> {
    /// Set a material-selected (feature) axis.
    pub fn feature(mut self, name: &str, value: impl Into<VariantValue>) -> Self {
        self.set
            .push((name.to_string(), value.into(), VariantSelector::Feature));
        self
    }

    /// Set a render-pipeline-selected (system) axis.
    pub fn system(mut self, name: &str, value: impl Into<VariantValue>) -> Self {
        self.set
            .push((name.to_string(), value.into(), VariantSelector::System));
        self
    }

    /// Seed this selection with a prebuilt **feature half**
    /// (see [`ShaderVariantSpace::build_features`]): the render pipeline then
    /// adds its `.system()` axes and [`build`](Self::build)s the full key.
    /// Feature axes present in `features` count as explicitly set (a duplicate
    /// `.feature()` call for the same axis is an error, as always).
    pub fn with_features(mut self, features: &VariantKey) -> Self {
        for (name, value) in features.defines() {
            let value = match self.space.axes.iter().find(|a| a.name == *name) {
                Some(VariantAxis {
                    kind: VariantAxisKind::Bool { .. },
                    ..
                }) => VariantValue::Bool(true),
                _ => VariantValue::Value(value.clone()),
            };
            self.set
                .push((name.clone(), value, VariantSelector::Feature));
        }
        // Bool feature axes the half resolved to OFF are absent from its
        // defines; mark them explicitly set so build() does not re-apply a
        // `default 1` over the material's deliberate `false`.
        for axis in &self.space.axes {
            if axis.selector == VariantSelector::Feature
                && !self.set.iter().any(|(n, _, _)| *n == axis.name)
                && matches!(axis.kind, VariantAxisKind::Bool { .. })
            {
                self.set
                    .push((axis.name.clone(), VariantValue::Bool(false), axis.selector));
            }
        }
        self
    }

    /// Validate the selection against the space and produce the canonical key.
    ///
    /// Feature axes fall back to their declared defaults; every system axis
    /// must have been set. Unknown names, kind/selector mismatches and
    /// duplicates are errors — never silently ignored.
    pub fn build(self) -> Result<VariantKey, VariantError> {
        self.build_impl(true)
    }

    /// As [`build`](Self::build), but for the **feature half** only: system
    /// axes are not required (the render pipeline sets them at draw time) and
    /// contribute nothing to the key. Setting one explicitly still errors.
    fn build_feature_half(self) -> Result<VariantKey, VariantError> {
        self.build_impl(false)
    }

    fn build_impl(self, require_system: bool) -> Result<VariantKey, VariantError> {
        let mut seen: HashSet<&str> = HashSet::new();
        for (name, _, _) in &self.set {
            if !seen.insert(name.as_str()) {
                return Err(VariantError::DuplicateSelection(name.clone()));
            }
        }

        let mut defines: Vec<(String, String)> = Vec::new();
        for axis in &self.space.axes {
            let explicit = self.set.iter().find(|(n, _, _)| *n == axis.name);

            if let Some((_, _, used_selector)) = explicit
                && *used_selector != axis.selector
            {
                return Err(VariantError::SelectorMismatch {
                    axis: axis.name.clone(),
                    expected: axis.selector,
                });
            }

            match (&axis.kind, explicit) {
                (VariantAxisKind::Bool { .. }, Some((_, VariantValue::Bool(true), _))) => {
                    defines.push((axis.name.clone(), String::new()));
                }
                (VariantAxisKind::Bool { .. }, Some((_, VariantValue::Bool(false), _))) => {}
                (VariantAxisKind::Bool { .. }, Some((_, VariantValue::Value(v), _))) => {
                    return Err(VariantError::InvalidValue {
                        axis: axis.name.clone(),
                        value: v.clone(),
                    });
                }
                (VariantAxisKind::Enum { values, .. }, Some((_, VariantValue::Value(v), _))) => {
                    if !values.contains(v) {
                        return Err(VariantError::InvalidValue {
                            axis: axis.name.clone(),
                            value: v.clone(),
                        });
                    }
                    defines.push((axis.name.clone(), v.clone()));
                }
                (VariantAxisKind::Enum { .. }, Some((_, VariantValue::Bool(b), _))) => {
                    return Err(VariantError::InvalidValue {
                        axis: axis.name.clone(),
                        value: b.to_string(),
                    });
                }
                (kind, None) => match axis.selector {
                    VariantSelector::System => {
                        if require_system {
                            return Err(VariantError::MissingSystemAxis(axis.name.clone()));
                        }
                    }
                    VariantSelector::Feature => match kind {
                        VariantAxisKind::Bool { default: true } => {
                            defines.push((axis.name.clone(), String::new()));
                        }
                        VariantAxisKind::Bool { default: false } => {}
                        VariantAxisKind::Enum { default, .. } => {
                            defines.push((axis.name.clone(), default.clone()));
                        }
                    },
                },
            }
        }

        // Axes set but not declared: everything declared was consumed above,
        // so any leftover name is unknown.
        for (name, _, _) in &self.set {
            if !self.space.axes.iter().any(|a| a.name == *name) {
                return Err(VariantError::UnknownAxis(name.clone()));
            }
        }

        defines.sort();
        Ok(VariantKey { defines })
    }
}

/// A canonical variant selection: the sorted `(define, value)` pairs one
/// shader set is compiled with. Hashable and ordered — the variant component
/// of pipeline-cache keys (Decision 4's full formula:
/// `shader + defines + layout + state + formats`).
#[derive(Debug, Clone, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct VariantKey {
    defines: Vec<(String, String)>,
}

impl VariantKey {
    /// The empty key (a shader with no axes, or all-default/off).
    pub fn empty() -> Self {
        Self::default()
    }

    /// Whether no defines are set.
    pub fn is_empty(&self) -> bool {
        self.defines.is_empty()
    }

    /// The sorted `(name, value)` defines this key compiles with.
    pub fn defines(&self) -> &[(String, String)] {
        &self.defines
    }
}

impl fmt::Display for VariantKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "[")?;
        for (i, (k, v)) in self.defines.iter().enumerate() {
            if i > 0 {
                write!(f, ",")?;
            }
            if v.is_empty() {
                write!(f, "{k}")?;
            } else {
                write!(f, "{k}={v}")?;
            }
        }
        write!(f, "]")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SHADER: &str = r#"
// A comment that is not a pragma.
//#pragma variant HAS_NORMAL_MAP
//#pragma variant GLOW default 1
//#pragma variant QUALITY 0 1 2 default 1
//#pragma variant_system HDR_OUTPUT
void main() {}
"#;

    #[test]
    fn parse_reads_axes_sorted_with_defaults() {
        let space = ShaderVariantSpace::parse(SHADER).unwrap();
        let names: Vec<&str> = space.axes().iter().map(|a| a.name.as_str()).collect();
        assert_eq!(names, ["GLOW", "HAS_NORMAL_MAP", "HDR_OUTPUT", "QUALITY"]);
        assert_eq!(
            space.axes()[0].kind,
            VariantAxisKind::Bool { default: true }
        );
        assert_eq!(space.axes()[2].selector, VariantSelector::System);
        assert_eq!(
            space.axes()[3].kind,
            VariantAxisKind::Enum {
                values: vec!["0".into(), "1".into(), "2".into()],
                default: "1".into(),
            }
        );
    }

    #[test]
    fn no_pragmas_is_empty_space_with_one_variant() {
        let space = ShaderVariantSpace::parse("void main() {}").unwrap();
        assert!(space.is_empty());
        let all = space.enumerate_all().unwrap();
        assert_eq!(all, vec![VariantKey::empty()]);
    }

    #[test]
    fn select_applies_feature_defaults_and_requires_system() {
        let space = ShaderVariantSpace::parse(SHADER).unwrap();

        // System axis unset → error naming it.
        assert_eq!(
            space.select().build(),
            Err(VariantError::MissingSystemAxis("HDR_OUTPUT".into()))
        );

        // Defaults: GLOW on (default 1), QUALITY=1, HAS_NORMAL_MAP off.
        let key = space.select().system("HDR_OUTPUT", false).build().unwrap();
        assert_eq!(
            key.defines(),
            [
                ("GLOW".to_string(), String::new()),
                ("QUALITY".to_string(), "1".to_string()),
            ]
        );

        let key = space
            .select()
            .feature("HAS_NORMAL_MAP", true)
            .feature("QUALITY", 2)
            .system("HDR_OUTPUT", true)
            .build()
            .unwrap();
        assert_eq!(
            key.to_string(),
            "[GLOW,HAS_NORMAL_MAP,HDR_OUTPUT,QUALITY=2]"
        );
    }

    #[test]
    fn select_rejects_typos_kinds_and_wrong_selector() {
        let space = ShaderVariantSpace::parse(SHADER).unwrap();
        assert_eq!(
            space
                .select()
                .feature("HAS_NORMLA_MAP", true)
                .system("HDR_OUTPUT", false)
                .build(),
            Err(VariantError::UnknownAxis("HAS_NORMLA_MAP".into()))
        );
        assert_eq!(
            space
                .select()
                .feature("QUALITY", 7)
                .system("HDR_OUTPUT", false)
                .build(),
            Err(VariantError::InvalidValue {
                axis: "QUALITY".into(),
                value: "7".into()
            })
        );
        // Feature axis via .system() — split ownership is enforced.
        assert_eq!(
            space
                .select()
                .system("GLOW", true)
                .system("HDR_OUTPUT", false)
                .build(),
            Err(VariantError::SelectorMismatch {
                axis: "GLOW".into(),
                expected: VariantSelector::Feature
            })
        );
        assert_eq!(
            space
                .select()
                .feature("GLOW", true)
                .feature("GLOW", false)
                .system("HDR_OUTPUT", false)
                .build(),
            Err(VariantError::DuplicateSelection("GLOW".into()))
        );
    }

    #[test]
    fn enumerate_covers_cartesian_and_caps() {
        let space = ShaderVariantSpace::parse(SHADER).unwrap();
        // 2 (HAS_NORMAL_MAP) × 2 (GLOW) × 3 (QUALITY) × 2 (HDR_OUTPUT) = 24
        let all = space.enumerate_all().unwrap();
        assert_eq!(all.len(), 24);
        // All keys distinct and sorted-canonical.
        let set: HashSet<&VariantKey> = all.iter().collect();
        assert_eq!(set.len(), 24);

        // 2^7 = 128 > cap.
        let big: String = (0..7)
            .map(|i| format!("//#pragma variant AXIS_{i}\n"))
            .collect();
        assert_eq!(
            ShaderVariantSpace::parse(&big).unwrap().enumerate_all(),
            Err(VariantError::TooManyVariants { count: 128 })
        );
    }

    #[test]
    fn parse_rejects_bad_pragmas() {
        assert!(matches!(
            ShaderVariantSpace::parse("//#pragma variant"),
            Err(VariantError::BadPragma { .. })
        ));
        assert!(matches!(
            ShaderVariantSpace::parse("//#pragma variant 9BAD"),
            Err(VariantError::BadPragma { .. })
        ));
        assert_eq!(
            ShaderVariantSpace::parse("//#pragma variant A\n//#pragma variant_system A"),
            Err(VariantError::DuplicateAxis("A".into()))
        );
        // System axes must not declare defaults.
        assert!(matches!(
            ShaderVariantSpace::parse("//#pragma variant_system HDR default 1"),
            Err(VariantError::BadPragma { .. })
        ));
        assert!(matches!(
            ShaderVariantSpace::parse("//#pragma variant Q 0 1 default 5"),
            Err(VariantError::BadPragma { .. })
        ));
    }

    /// Bool-off emits nothing: the all-off key of a bool-only space hashes
    /// like the historical "no defines" bake entry.
    #[test]
    fn all_off_bool_key_is_empty() {
        let space =
            ShaderVariantSpace::parse("//#pragma variant_system HDR_OUTPUT\n//#pragma variant X")
                .unwrap();
        let key = space.select().system("HDR_OUTPUT", false).build().unwrap();
        assert!(key.is_empty());
        assert_eq!(key, VariantKey::empty());
    }
}

#[cfg(test)]
mod split_selection_tests {
    use super::*;

    const SHADER: &str = r#"
//#pragma variant ALPHA_CUTOUT
//#pragma variant GLOW default 1
//#pragma variant QUALITY 0 1 2 default 1
//#pragma variant_system HDR_OUTPUT
"#;

    /// The material half validates feature axes and applies their defaults,
    /// without requiring (or emitting) system axes.
    #[test]
    fn feature_half_builds_without_system_axes() {
        let space = ShaderVariantSpace::parse(SHADER).unwrap();
        let half = space
            .build_features(&[("ALPHA_CUTOUT".into(), true.into())])
            .unwrap();
        assert_eq!(half.to_string(), "[ALPHA_CUTOUT,GLOW,QUALITY=1]");

        // Typos and system axes are errors at material-resolve time.
        assert_eq!(
            space.build_features(&[("ALPHA_CUTOUP".into(), true.into())]),
            Err(VariantError::UnknownAxis("ALPHA_CUTOUP".into()))
        );
        assert_eq!(
            space.build_features(&[("HDR_OUTPUT".into(), true.into())]),
            Err(VariantError::SelectorMismatch {
                axis: "HDR_OUTPUT".into(),
                expected: VariantSelector::System
            })
        );
    }

    /// The pipeline completes a feature half with its system axes; the result
    /// equals a one-shot full selection of the same values.
    #[test]
    fn with_features_completes_to_the_same_full_key() {
        let space = ShaderVariantSpace::parse(SHADER).unwrap();
        let half = space
            .build_features(&[
                ("ALPHA_CUTOUT".into(), true.into()),
                ("QUALITY".into(), 2.into()),
            ])
            .unwrap();
        let full = space
            .select()
            .with_features(&half)
            .system("HDR_OUTPUT", true)
            .build()
            .unwrap();
        let oneshot = space
            .select()
            .feature("ALPHA_CUTOUT", true)
            .feature("QUALITY", 2)
            .system("HDR_OUTPUT", true)
            .build()
            .unwrap();
        assert_eq!(full, oneshot);
    }

    /// A bool feature the material deliberately turned OFF (against a
    /// `default 1`) must stay off through the half → full round trip.
    #[test]
    fn deliberate_false_survives_the_round_trip() {
        let space = ShaderVariantSpace::parse(SHADER).unwrap();
        let half = space
            .build_features(&[("GLOW".into(), false.into())])
            .unwrap();
        assert!(!half.defines().iter().any(|(n, _)| n == "GLOW"));
        let full = space
            .select()
            .with_features(&half)
            .system("HDR_OUTPUT", false)
            .build()
            .unwrap();
        assert!(
            !full.defines().iter().any(|(n, _)| n == "GLOW"),
            "default 1 must not resurrect a deliberately-off feature"
        );
    }
}
