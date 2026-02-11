use proc_macro::TokenStream;
use quote::quote;
use syn::{Data, DeriveInput, Fields, Type, parse_macro_input};

/// Derive the `Component` trait for a Pod struct, providing runtime reflection.
///
/// The struct must also derive `bytemuck::Pod`, `bytemuck::Zeroable`, `Copy`,
/// `Clone`, and have `#[repr(C)]`.
///
/// # Named structs
///
/// ```ignore
/// #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Component)]
/// #[repr(C)]
/// struct Transform {
///     translation: Vec3,
///     rotation: Quat,
///     scale: Vec3,
/// }
/// ```
///
/// # Tuple structs
///
/// ```ignore
/// #[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable, Component)]
/// #[repr(C)]
/// struct GlobalTransform(pub Mat4);
/// ```
#[proc_macro_derive(Component)]
pub fn derive_component(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;
    let name_str = name.to_string();
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    let (field_infos, field_match, field_mut_match) = match &input.data {
        Data::Struct(data) => match &data.fields {
            Fields::Named(fields) => {
                // Skip fields starting with `_` (padding fields for Pod alignment)
                let visible_fields: Vec<_> = fields
                    .named
                    .iter()
                    .filter(|f| {
                        !f.ident
                            .as_ref()
                            .is_some_and(|id| id.to_string().starts_with('_'))
                    })
                    .collect();

                let infos = visible_fields.iter().map(|f| {
                    let fname = f.ident.as_ref().unwrap();
                    let fname_str = fname.to_string();
                    let ftype = &f.ty;
                    let kind = infer_field_kind(ftype);
                    quote! {
                        redlilium_ecs::FieldInfo {
                            name: #fname_str,
                            type_name: ::core::any::type_name::<#ftype>(),
                            kind: #kind,
                        }
                    }
                });

                let field_arms = visible_fields.iter().map(|f| {
                    let fname = f.ident.as_ref().unwrap();
                    let fname_str = fname.to_string();
                    quote! {
                        #fname_str => ::core::option::Option::Some(&self.#fname as &dyn ::core::any::Any)
                    }
                });

                let field_mut_arms = visible_fields.iter().map(|f| {
                    let fname = f.ident.as_ref().unwrap();
                    let fname_str = fname.to_string();
                    quote! {
                        #fname_str => ::core::option::Option::Some(&mut self.#fname as &mut dyn ::core::any::Any)
                    }
                });

                (
                    quote! { &[#(#infos),*] },
                    quote! { match name { #(#field_arms,)* _ => ::core::option::Option::None } },
                    quote! { match name { #(#field_mut_arms,)* _ => ::core::option::Option::None } },
                )
            }
            Fields::Unnamed(fields) => {
                let infos = fields.unnamed.iter().enumerate().map(|(i, f)| {
                    let idx_str = i.to_string();
                    let ftype = &f.ty;
                    let kind = infer_field_kind(ftype);
                    quote! {
                        redlilium_ecs::FieldInfo {
                            name: #idx_str,
                            type_name: ::core::any::type_name::<#ftype>(),
                            kind: #kind,
                        }
                    }
                });

                let field_arms = fields.unnamed.iter().enumerate().map(|(i, _f)| {
                    let idx_str = i.to_string();
                    let idx = syn::Index::from(i);
                    quote! {
                        #idx_str => ::core::option::Option::Some(&self.#idx as &dyn ::core::any::Any)
                    }
                });

                let field_mut_arms = fields.unnamed.iter().enumerate().map(|(i, _f)| {
                    let idx_str = i.to_string();
                    let idx = syn::Index::from(i);
                    quote! {
                        #idx_str => ::core::option::Option::Some(&mut self.#idx as &mut dyn ::core::any::Any)
                    }
                });

                (
                    quote! { &[#(#infos),*] },
                    quote! { match name { #(#field_arms,)* _ => ::core::option::Option::None } },
                    quote! { match name { #(#field_mut_arms,)* _ => ::core::option::Option::None } },
                )
            }
            Fields::Unit => (
                quote! { &[] },
                quote! { { let _ = name; ::core::option::Option::None } },
                quote! { { let _ = name; ::core::option::Option::None } },
            ),
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
            fn component_name(&self) -> &'static str {
                #name_str
            }

            fn field_infos(&self) -> &'static [redlilium_ecs::FieldInfo] {
                static INFOS: ::std::sync::LazyLock<::std::vec::Vec<redlilium_ecs::FieldInfo>> =
                    ::std::sync::LazyLock::new(|| ::std::vec::Vec::from(#field_infos));
                &INFOS
            }

            fn field(&self, name: &str) -> ::core::option::Option<&dyn ::core::any::Any> {
                #field_match
            }

            fn field_mut(&mut self, name: &str) -> ::core::option::Option<&mut dyn ::core::any::Any> {
                #field_mut_match
            }
        }
    };

    expanded.into()
}

/// Infer `FieldKind` from a type by matching the last path segment.
fn infer_field_kind(ty: &Type) -> proc_macro2::TokenStream {
    let type_name = extract_last_segment(ty);
    match type_name.as_str() {
        "f32" => quote! { redlilium_ecs::FieldKind::F32 },
        "u8" => quote! { redlilium_ecs::FieldKind::U8 },
        "u32" => quote! { redlilium_ecs::FieldKind::U32 },
        "i32" => quote! { redlilium_ecs::FieldKind::I32 },
        "Vec2" | "Vector2" => quote! { redlilium_ecs::FieldKind::Vec2 },
        "Vec3" | "Vec3A" | "Vector3" => quote! { redlilium_ecs::FieldKind::Vec3 },
        "Vec4" | "Vector4" => quote! { redlilium_ecs::FieldKind::Vec4 },
        "Quat" | "Quaternion" => quote! { redlilium_ecs::FieldKind::Quat },
        "Mat4" | "Matrix4" => quote! { redlilium_ecs::FieldKind::Mat4 },
        "StringId" => quote! { redlilium_ecs::FieldKind::StringId },
        _ => {
            let msg = format!(
                "Component derive: unknown field type `{}`. Expected one of: f32, u8, u32, i32, Vec2, Vec3, Vec4, Quat, Mat4, StringId.",
                type_name
            );
            quote! { compile_error!(#msg) }
        }
    }
}

/// Extract the last segment name from a type path (e.g. `nalgebra::Vector3` → `"Vector3"`).
fn extract_last_segment(ty: &Type) -> String {
    match ty {
        Type::Path(type_path) => {
            if let Some(segment) = type_path.path.segments.last() {
                segment.ident.to_string()
            } else {
                String::new()
            }
        }
        _ => String::new(),
    }
}
