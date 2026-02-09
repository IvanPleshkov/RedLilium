use proc_macro::TokenStream;
use quote::quote;
use syn::{Data, DeriveInput, Fields, ImplItem, ItemImpl, ReturnType, Type, parse_macro_input};

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
        "Vec2" => quote! { redlilium_ecs::FieldKind::Vec2 },
        "Vec3" | "Vec3A" => quote! { redlilium_ecs::FieldKind::Vec3 },
        "Vec4" => quote! { redlilium_ecs::FieldKind::Vec4 },
        "Quat" => quote! { redlilium_ecs::FieldKind::Quat },
        "Mat4" => quote! { redlilium_ecs::FieldKind::Mat4 },
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

/// Extract the last segment name from a type path (e.g. `glam::Vec3` → `"Vec3"`).
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

/// Generates an `impl System for ...` block from a simplified async impl.
///
/// # Usage
///
/// ```ignore
/// #[system]
/// impl UpdateGlobalTransforms {
///     async fn run(&self, access: QueryAccess<'_>) {
///         access.scope(|world| update_global_transforms(world));
///     }
///
///     fn access(&self) -> Access {
///         let mut access = Access::new();
///         access.add_read::<Transform>();
///         access.add_write::<GlobalTransform>();
///         access
///     }
/// }
/// ```
///
/// # With return value
///
/// When `run` returns a non-`()` type, the result is automatically stored as
/// a [`SystemResult<Self, T>`](redlilium_ecs::SystemResult) resource in the World.
///
/// ```ignore
/// #[system]
/// impl PhysicsSystem {
///     async fn run(&self, access: QueryAccess<'_>) -> PhysicsResult {
///         // ... compute ...
///         PhysicsResult { collision_count: 42 }
///     }
///     fn access(&self) -> Access { ... }
/// }
/// ```
#[proc_macro_attribute]
pub fn system(_attr: TokenStream, item: TokenStream) -> TokenStream {
    let input = parse_macro_input!(item as ItemImpl);

    // Extract the struct type name
    let self_ty = &input.self_ty;

    // Find the `run` and `access` methods
    let mut run_method = None;
    let mut access_method = None;
    let mut other_items = Vec::new();

    for item in &input.items {
        match item {
            ImplItem::Fn(method) => {
                let name = method.sig.ident.to_string();
                if name == "run" {
                    run_method = Some(method);
                } else if name == "access" {
                    access_method = Some(method);
                } else {
                    other_items.push(item);
                }
            }
            other => other_items.push(other),
        }
    }

    let run = match run_method {
        Some(m) => m,
        None => {
            return syn::Error::new_spanned(
                self_ty,
                "#[system] impl must contain an `async fn run`",
            )
            .to_compile_error()
            .into();
        }
    };

    let access = match access_method {
        Some(m) => m,
        None => {
            return syn::Error::new_spanned(self_ty, "#[system] impl must contain an `fn access`")
                .to_compile_error()
                .into();
        }
    };

    // Extract the run body
    let run_body = &run.block;

    // Check return type: () or some T
    let has_return_type = matches!(&run.sig.output, ReturnType::Type(_, ty) if !is_unit_type(ty));

    let (run_impl, access_extra) = if has_return_type {
        let ret_ty = match &run.sig.output {
            ReturnType::Type(_, ty) => ty,
            _ => unreachable!(),
        };
        // Wrap body to store result as SystemResult resource
        (
            quote! {
                fn run<'a>(&'a self, access: redlilium_ecs::QueryAccess<'a>) -> redlilium_ecs::SystemFuture<'a> {
                    redlilium_ecs::SystemFuture::new(async move {
                        let __system_result: #ret_ty = async #run_body.await;
                        access.scope(|world| {
                            world.insert_resource(
                                redlilium_ecs::SystemResult::<#self_ty, #ret_ty>::new(__system_result)
                            );
                        });
                    })
                }
            },
            quote! {
                __access.add_resource_write::<redlilium_ecs::SystemResult<#self_ty, #ret_ty>>();
            },
        )
    } else {
        (
            quote! {
                fn run<'a>(&'a self, access: redlilium_ecs::QueryAccess<'a>) -> redlilium_ecs::SystemFuture<'a> {
                    redlilium_ecs::SystemFuture::new(async move #run_body)
                }
            },
            quote! {},
        )
    };

    // Extract the access body, injecting extra resource access if needed
    let access_body = &access.block;
    let access_impl = if has_return_type {
        quote! {
            fn access(&self) -> redlilium_ecs::Access {
                let mut __access: redlilium_ecs::Access = (|| #access_body)();
                #access_extra
                __access
            }
        }
    } else {
        quote! {
            fn access(&self) -> redlilium_ecs::Access #access_body
        }
    };

    let expanded = quote! {
        impl redlilium_ecs::System for #self_ty {
            #run_impl
            #access_impl
        }
    };

    expanded.into()
}

/// Returns true if the type is `()`.
fn is_unit_type(ty: &Type) -> bool {
    match ty {
        Type::Tuple(tuple) => tuple.elems.is_empty(),
        _ => false,
    }
}
