use proc_macro::TokenStream;
use quote::quote;
use syn::{Data, DeriveInput, Fields, parse_macro_input};

/// Derive the `Component` trait with automatic inspector UI generation.
///
/// Generates `component_name()` returning the struct name, and `inspect_ui()`
/// using the [`Inspect`](redlilium_ecs::inspect::Inspect) wrapper for each field.
///
/// Fields starting with `_` are skipped in the inspector.
///
/// # Example
///
/// ```ignore
/// #[derive(Component)]
/// struct Health {
///     current: f32,
///     max: f32,
/// }
/// ```
#[proc_macro_derive(Component)]
pub fn derive_component(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;
    let name_str = name.to_string();
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    let inspect_body = match &input.data {
        Data::Struct(data) => match &data.fields {
            Fields::Named(fields) => {
                let stmts = fields
                    .named
                    .iter()
                    .filter(|f| {
                        !f.ident
                            .as_ref()
                            .is_some_and(|id| id.to_string().starts_with('_'))
                    })
                    .map(|f| {
                        let fname = f.ident.as_ref().unwrap();
                        let fname_str = fname.to_string();
                        quote! {
                            redlilium_ecs::inspect::Inspect(&mut self.#fname).show(#fname_str, ui);
                        }
                    });
                quote! { #(#stmts)* }
            }
            Fields::Unnamed(fields) => {
                let stmts = fields.unnamed.iter().enumerate().map(|(i, _)| {
                    let idx_str = i.to_string();
                    let idx = syn::Index::from(i);
                    quote! {
                        redlilium_ecs::inspect::Inspect(&mut self.#idx).show(#idx_str, ui);
                    }
                });
                quote! { #(#stmts)* }
            }
            Fields::Unit => {
                quote! { let _ = ui; }
            }
        },
        _ => {
            return syn::Error::new_spanned(
                &input.ident,
                "Component can only be derived for structs",
            )
            .to_compile_error()
            .into();
        }
    };

    let expanded = quote! {
        impl #impl_generics redlilium_ecs::Component for #name #ty_generics #where_clause {
            const NAME: &'static str = #name_str;

            fn inspect_ui(&mut self, ui: &mut redlilium_ecs::egui::Ui) {
                #[allow(unused_imports)]
                use redlilium_ecs::inspect::InspectFallback as _;
                #inspect_body
            }
        }
    };

    expanded.into()
}

/// Derive the `Bundle` trait for a struct, allowing it to be inserted as a
/// group of components on an entity.
///
/// Each field is inserted as an individual component via `world.insert()`.
/// Fields annotated with `#[bundle]` are treated as nested bundles and
/// inserted via `Bundle::insert_into()` instead.
///
/// Only named-field structs are supported.
///
/// # Example
///
/// ```ignore
/// #[derive(Bundle)]
/// struct PlayerBundle {
///     transform: Transform,
///     global_transform: GlobalTransform,
///     visibility: Visibility,
///     name: Name,
/// }
///
/// // Nested bundles
/// #[derive(Bundle)]
/// struct EnemyBundle {
///     health: Health,
///     #[bundle]
///     spatial: SpatialBundle,
/// }
/// ```
#[proc_macro_derive(Bundle, attributes(bundle))]
pub fn derive_bundle(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    let fields = match &input.data {
        Data::Struct(data) => match &data.fields {
            Fields::Named(fields) => &fields.named,
            _ => {
                return syn::Error::new_spanned(
                    &input.ident,
                    "Bundle can only be derived for structs with named fields",
                )
                .to_compile_error()
                .into();
            }
        },
        _ => {
            return syn::Error::new_spanned(&input.ident, "Bundle can only be derived for structs")
                .to_compile_error()
                .into();
        }
    };

    let field_names: Vec<_> = fields.iter().map(|f| f.ident.as_ref().unwrap()).collect();

    let insert_stmts: Vec<_> = fields
        .iter()
        .map(|f| {
            let fname = f.ident.as_ref().unwrap();
            let is_bundle = f.attrs.iter().any(|a| a.path().is_ident("bundle"));
            if is_bundle {
                quote! {
                    redlilium_ecs::Bundle::insert_into(#fname, world, entity)?;
                }
            } else {
                quote! {
                    world.insert(entity, #fname)?;
                }
            }
        })
        .collect();

    let expanded = quote! {
        impl #impl_generics redlilium_ecs::Bundle for #name #ty_generics #where_clause {
            fn insert_into(
                self,
                world: &mut redlilium_ecs::World,
                entity: redlilium_ecs::Entity,
            ) -> Result<(), redlilium_ecs::ComponentNotRegistered> {
                let Self { #(#field_names,)* } = self;
                #(#insert_stmts)*
                Ok(())
            }
        }
    };

    expanded.into()
}
