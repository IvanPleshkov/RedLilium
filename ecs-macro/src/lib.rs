use proc_macro::TokenStream;
use quote::quote;
use syn::{Data, DeriveInput, Fields, Meta, parse_macro_input};

/// Derive the `Component` trait with automatic inspector UI, entity collection,
/// entity remapping, and serialization.
///
/// Generates:
/// - `inspect_ui()` using [`Inspect`](redlilium_ecs::inspect::Inspect) wrappers
/// - `collect_entities()` using [`EntityRef`](redlilium_ecs::map_entities::EntityRef) wrappers
/// - `remap_entities()` using [`EntityMut`](redlilium_ecs::map_entities::EntityMut) wrappers
/// - `serialize_component()` using [`SerializeField`](redlilium_ecs::serialize::SerializeField) wrappers
/// - `deserialize_component()` using [`DeserializeField`](redlilium_ecs::serialize::DeserializeField) wrappers
///
/// Fields starting with `_` are skipped in all generated methods.
/// Skipped fields use `Default::default()` during deserialization.
///
/// Use `#[skip_serialization]` on the struct to opt out of generated
/// serialize/deserialize methods (they will use the default "not serializable"
/// implementations from the trait).
///
/// # Example
///
/// ```ignore
/// #[derive(Component)]
/// struct Health {
///     current: f32,
///     max: f32,
/// }
///
/// // Opt out of serialization for GPU resource wrappers:
/// #[derive(Component)]
/// #[skip_serialization]
/// struct RenderMesh(pub Arc<Mesh>);
/// ```
#[proc_macro_derive(Component, attributes(require, skip_serialization))]
pub fn derive_component(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    let name = &input.ident;
    let name_str = name.to_string();
    let (impl_generics, ty_generics, where_clause) = input.generics.split_for_impl();

    // Check for #[skip_serialization] attribute
    let skip_serialization = input
        .attrs
        .iter()
        .any(|a| a.path().is_ident("skip_serialization"));

    // Collect required component types from #[require(Type1, Type2, ...)]
    let mut required_types = Vec::new();
    for attr in &input.attrs {
        if attr.path().is_ident("require")
            && let Meta::List(meta_list) = &attr.meta
        {
            let result = meta_list.parse_args_with(
                syn::punctuated::Punctuated::<syn::Path, syn::Token![,]>::parse_terminated,
            );
            match result {
                Ok(paths) => required_types.extend(paths),
                Err(e) => return e.to_compile_error().into(),
            }
        }
    }

    let (inspect_body, collect_body, remap_body, serialize_body, deserialize_body) = match &input
        .data
    {
        Data::Struct(data) => match &data.fields {
            Fields::Named(fields) => {
                let all_fields: Vec<_> = fields.named.iter().collect();
                let visible_fields: Vec<_> = all_fields
                    .iter()
                    .filter(|f| {
                        !f.ident
                            .as_ref()
                            .is_some_and(|id| id.to_string().starts_with('_'))
                    })
                    .copied()
                    .collect();
                let skipped_fields: Vec<_> = all_fields
                    .iter()
                    .filter(|f| {
                        f.ident
                            .as_ref()
                            .is_some_and(|id| id.to_string().starts_with('_'))
                    })
                    .copied()
                    .collect();

                let inspect_stmts = visible_fields.iter().map(|f| {
                    let fname = f.ident.as_ref().unwrap();
                    let fname_str = fname.to_string();
                    quote! {
                        let #fname = match redlilium_ecs::inspect::Inspect(&self.#fname).show(#fname_str, ui) {
                            Some(v) => { _changed = true; v }
                            None => self.#fname.clone(),
                        };
                    }
                });
                let visible_names: Vec<_> = visible_fields
                    .iter()
                    .map(|f| f.ident.as_ref().unwrap())
                    .collect();
                let skipped_clone = skipped_fields.iter().map(|f| {
                    let fname = f.ident.as_ref().unwrap();
                    quote! { #fname: self.#fname.clone() }
                });
                let collect_stmts = visible_fields.iter().map(|f| {
                        let fname = f.ident.as_ref().unwrap();
                        quote! {
                            redlilium_ecs::map_entities::EntityRef(&self.#fname).collect_entities(collector);
                        }
                    });
                let remap_stmts = visible_fields.iter().map(|f| {
                        let fname = f.ident.as_ref().unwrap();
                        quote! {
                            redlilium_ecs::map_entities::EntityMut(&mut self.#fname).remap_entities(map);
                        }
                    });

                let serialize_body = if skip_serialization {
                    quote! {}
                } else {
                    let ser_stmts = visible_fields.iter().map(|f| {
                            let fname = f.ident.as_ref().unwrap();
                            let fname_str = fname.to_string();
                            quote! {
                                redlilium_ecs::serialize::SerializeField(&self.#fname).serialize_field(#fname_str, ctx)?;
                            }
                        });
                    quote! {
                        fn serialize_component(
                            &self,
                            ctx: &mut redlilium_ecs::serialize::SerializeContext<'_>,
                        ) -> Result<redlilium_ecs::serialize::Value, redlilium_ecs::serialize::SerializeError> {
                            #[allow(unused_imports)]
                            use redlilium_ecs::serialize::SerializeFieldFallback as _;
                            ctx.begin_struct(Self::NAME)?;
                            #(#ser_stmts)*
                            ctx.end_struct()
                        }
                    }
                };

                let deserialize_body = if skip_serialization {
                    quote! {}
                } else {
                    let deser_stmts = visible_fields.iter().map(|f| {
                            let fname = f.ident.as_ref().unwrap();
                            let fname_str = fname.to_string();
                            let fty = &f.ty;
                            quote! {
                                let #fname = redlilium_ecs::serialize::DeserializeField::<#fty>::deserialize_field(#fname_str, ctx)?;
                            }
                        });
                    let visible_names: Vec<_> = visible_fields
                        .iter()
                        .map(|f| f.ident.as_ref().unwrap())
                        .collect();
                    let skipped_defaults = skipped_fields.iter().map(|f| {
                        let fname = f.ident.as_ref().unwrap();
                        quote! { #fname: Default::default() }
                    });
                    quote! {
                        fn deserialize_component(
                            ctx: &mut redlilium_ecs::serialize::DeserializeContext<'_>,
                        ) -> Result<Self, redlilium_ecs::serialize::DeserializeError>
                        where
                            Self: Sized,
                        {
                            #[allow(unused_imports)]
                            use redlilium_ecs::serialize::DeserializeFieldFallback as _;
                            ctx.begin_struct(Self::NAME)?;
                            #(#deser_stmts)*
                            ctx.end_struct()?;
                            Ok(Self { #(#visible_names,)* #(#skipped_defaults,)* })
                        }
                    }
                };

                (
                    quote! {
                        let mut _changed = false;
                        #(#inspect_stmts)*
                        if _changed { Some(Self { #(#visible_names,)* #(#skipped_clone,)* }) } else { None }
                    },
                    quote! { #(#collect_stmts)* },
                    quote! { #(#remap_stmts)* },
                    serialize_body,
                    deserialize_body,
                )
            }
            Fields::Unnamed(fields) => {
                let indices: Vec<_> = (0..fields.unnamed.len()).map(syn::Index::from).collect();

                let inspect_field_vars: Vec<_> = indices
                    .iter()
                    .map(|idx| {
                        syn::Ident::new(
                            &format!("_field_{}", idx.index),
                            proc_macro2::Span::call_site(),
                        )
                    })
                    .collect();
                let inspect_stmts: Vec<_> = indices.iter().zip(inspect_field_vars.iter()).map(|(idx, var)| {
                    let idx_str = idx.index.to_string();
                    quote! {
                        let #var = match redlilium_ecs::inspect::Inspect(&self.#idx).show(#idx_str, ui) {
                            Some(v) => { _changed = true; v }
                            None => self.#idx.clone(),
                        };
                    }
                }).collect();
                let collect_stmts = indices.iter().map(|idx| {
                        quote! {
                            redlilium_ecs::map_entities::EntityRef(&self.#idx).collect_entities(collector);
                        }
                    });
                let remap_stmts = indices.iter().map(|idx| {
                    quote! {
                        redlilium_ecs::map_entities::EntityMut(&mut self.#idx).remap_entities(map);
                    }
                });

                let serialize_body = if skip_serialization {
                    quote! {}
                } else {
                    let ser_stmts = indices.iter().zip(fields.unnamed.iter()).map(
                            |(idx, f)| {
                                let idx_str = idx.index.to_string();
                                let _ = &f.ty;
                                quote! {
                                    redlilium_ecs::serialize::SerializeField(&self.#idx).serialize_field(#idx_str, ctx)?;
                                }
                            },
                        );
                    quote! {
                        fn serialize_component(
                            &self,
                            ctx: &mut redlilium_ecs::serialize::SerializeContext<'_>,
                        ) -> Result<redlilium_ecs::serialize::Value, redlilium_ecs::serialize::SerializeError> {
                            #[allow(unused_imports)]
                            use redlilium_ecs::serialize::SerializeFieldFallback as _;
                            ctx.begin_struct(Self::NAME)?;
                            #(#ser_stmts)*
                            ctx.end_struct()
                        }
                    }
                };

                let deserialize_body = if skip_serialization {
                    quote! {}
                } else {
                    let deser_stmts =
                            indices.iter().zip(fields.unnamed.iter()).map(|(idx, f)| {
                                let idx_str = idx.index.to_string();
                                let fty = &f.ty;
                                let var_name = syn::Ident::new(
                                    &format!("field_{}", idx.index),
                                    proc_macro2::Span::call_site(),
                                );
                                quote! {
                                    let #var_name = redlilium_ecs::serialize::DeserializeField::<#fty>::deserialize_field(#idx_str, ctx)?;
                                }
                            });
                    let field_vars: Vec<_> = indices
                        .iter()
                        .map(|idx| {
                            syn::Ident::new(
                                &format!("field_{}", idx.index),
                                proc_macro2::Span::call_site(),
                            )
                        })
                        .collect();
                    quote! {
                        fn deserialize_component(
                            ctx: &mut redlilium_ecs::serialize::DeserializeContext<'_>,
                        ) -> Result<Self, redlilium_ecs::serialize::DeserializeError>
                        where
                            Self: Sized,
                        {
                            #[allow(unused_imports)]
                            use redlilium_ecs::serialize::DeserializeFieldFallback as _;
                            ctx.begin_struct(Self::NAME)?;
                            #(#deser_stmts)*
                            ctx.end_struct()?;
                            Ok(Self(#(#field_vars,)*))
                        }
                    }
                };

                (
                    quote! {
                        let mut _changed = false;
                        #(#inspect_stmts)*
                        if _changed { Some(Self(#(#inspect_field_vars,)*)) } else { None }
                    },
                    quote! { #(#collect_stmts)* },
                    quote! { #(#remap_stmts)* },
                    serialize_body,
                    deserialize_body,
                )
            }
            Fields::Unit => (
                quote! { let _ = ui; None },
                quote! { let _ = collector; },
                quote! { let _ = map; },
                if skip_serialization {
                    quote! {}
                } else {
                    quote! {
                        fn serialize_component(
                            &self,
                            ctx: &mut redlilium_ecs::serialize::SerializeContext<'_>,
                        ) -> Result<redlilium_ecs::serialize::Value, redlilium_ecs::serialize::SerializeError> {
                            ctx.begin_struct(Self::NAME)?;
                            ctx.end_struct()
                        }
                    }
                },
                if skip_serialization {
                    quote! {}
                } else {
                    quote! {
                        fn deserialize_component(
                            ctx: &mut redlilium_ecs::serialize::DeserializeContext<'_>,
                        ) -> Result<Self, redlilium_ecs::serialize::DeserializeError>
                        where
                            Self: Sized,
                        {
                            ctx.begin_struct(Self::NAME)?;
                            ctx.end_struct()?;
                            Ok(Self)
                        }
                    }
                },
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

    let register_required_body = if required_types.is_empty() {
        quote! {}
    } else {
        let stmts = required_types.iter().map(|ty| {
            quote! {
                world.register_required::<Self, #ty>();
            }
        });
        quote! {
            fn register_required(world: &mut redlilium_ecs::World) {
                #(#stmts)*
            }
        }
    };

    let expanded = quote! {
        impl #impl_generics redlilium_ecs::Component for #name #ty_generics #where_clause {
            const NAME: &'static str = #name_str;

            fn inspect_ui(&self, ui: &mut redlilium_ecs::egui::Ui) -> Option<Self> {
                #[allow(unused_imports)]
                use redlilium_ecs::inspect::InspectFallback as _;
                #inspect_body
            }

            fn collect_entities(&self, collector: &mut Vec<redlilium_ecs::Entity>) {
                #[allow(unused_imports)]
                use redlilium_ecs::map_entities::EntityRefFallback as _;
                #collect_body
            }

            fn remap_entities(&mut self, map: &mut dyn FnMut(redlilium_ecs::Entity) -> redlilium_ecs::Entity) {
                #[allow(unused_imports)]
                use redlilium_ecs::map_entities::EntityMutFallback as _;
                #remap_body
            }

            #register_required_body
            #serialize_body
            #deserialize_body
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
