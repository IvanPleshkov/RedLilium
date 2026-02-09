use proc_macro::TokenStream;
use quote::quote;
use syn::{Data, DeriveInput, Fields, parse_macro_input};

/// Derive the `Component` trait for a struct, providing runtime reflection.
///
/// # Named structs
///
/// ```ignore
/// #[derive(Component)]
/// struct Transform {
///     translation: Vec3,
///     rotation: Quat,
///     scale: Vec3,
/// }
/// ```
///
/// Field names become `"translation"`, `"rotation"`, `"scale"`.
///
/// # Tuple structs
///
/// ```ignore
/// #[derive(Component)]
/// struct GlobalTransform(pub Mat4);
/// ```
///
/// Field names become `"0"`.
///
/// # Unit structs
///
/// Supported but have no fields.
#[proc_macro_derive(Component)]
pub fn derive_component(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;
    let name_str = name.to_string();
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    let (field_infos, field_match, field_mut_match) = match &input.data {
        Data::Struct(data) => match &data.fields {
            Fields::Named(fields) => {
                let infos = fields.named.iter().map(|f| {
                    let fname = f.ident.as_ref().unwrap();
                    let fname_str = fname.to_string();
                    let ftype = &f.ty;
                    quote! {
                        redlilium_ecs::FieldInfo {
                            name: #fname_str,
                            type_name: ::core::any::type_name::<#ftype>(),
                            type_id: ::core::any::TypeId::of::<#ftype>(),
                        }
                    }
                });

                let field_arms = fields.named.iter().map(|f| {
                    let fname = f.ident.as_ref().unwrap();
                    let fname_str = fname.to_string();
                    quote! {
                        #fname_str => ::core::option::Option::Some(&self.#fname as &dyn ::core::any::Any)
                    }
                });

                let field_mut_arms = fields.named.iter().map(|f| {
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
                    quote! {
                        redlilium_ecs::FieldInfo {
                            name: #idx_str,
                            type_name: ::core::any::type_name::<#ftype>(),
                            type_id: ::core::any::TypeId::of::<#ftype>(),
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
