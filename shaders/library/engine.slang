// RedLilium Shader Library - Engine Module
// Engine-side declarations shared by all shaders.

// Update-frequency class of a binding set (a `ParameterBlock`), read by the
// engine via reflection (docs/MATERIAL_ASSETS.md Decision 7). The render
// system binds each set according to its class:
//   [UpdateRate("external")] - owned by the render system, per view/frame
//   [UpdateRate("dynamic")]  - ring-buffered, per draw (dynamic offset)
//   [UpdateRate("static")]   - per instance, built once
[__AttributeUsage(_AttributeTargets.Var)]
struct UpdateRateAttribute {
    string rate;
};
