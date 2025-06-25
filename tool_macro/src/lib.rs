extern crate proc_macro;
use proc_macro::TokenStream;
use proc_macro2::{Ident as Ident2, Span};
use quote::{format_ident, quote};
use serde_json::json;
use syn::{
    parse_macro_input, Attribute, Expr, FnArg, GenericArgument, ItemFn, Lit, Meta, Pat,
    PathArguments, Type,
};

/// Helper function to deny reference types in parameters
fn deny_references(ty: &Type) -> Result<(), syn::Error> {
    if matches!(ty, Type::Reference(_)) {
        Err(syn::Error::new_spanned(
            ty,
            "reference types (`&T`) are not supported; use owned types like `String` or `Vec<T>`",
        ))
    } else {
        Ok(())
    }
}

/// Extract the doc comments on the original function, concatenated and trimmed.
fn extract_doc(attrs: &[Attribute]) -> String {
    let mut out = String::new();
    for attr in attrs {
        if let Meta::NameValue(nv) = &attr.meta {
            if nv.path.is_ident("doc") {
                if let Expr::Lit(expr_lit) = &nv.value {
                    if let Lit::Str(lit_str) = &expr_lit.lit {
                        if !out.is_empty() {
                            out.push('\n');
                        }
                        out.push_str(lit_str.value().trim());
                    }
                }
            }
        }
    }
    out
}

/// Very small type-name mapper.
fn type_to_json_type(ty: &Type) -> String {
    if let Type::Path(tp) = ty {
        if let Some(ident) = tp.path.get_ident() {
            let name = ident.to_string();
            return match name.as_str() {
                "String" | "str" => "string".to_string(),
                "i32" | "i64" | "u32" | "u64" | "f32" | "f64" | "usize" | "isize" => {
                    "number".to_string()
                }
                "bool" => "boolean".to_string(),
                _ => "object".to_string(),
            };
        }
    }
    "object".to_string()
}

/// Attribute macro `#[tool]` that turns a nice Rust function into a `Tool`.
///
/// Example:
/// ```
/// use transformers::tool;
///
/// #[tool]
/// fn add(a: i32, b: i32) -> String {
///     (a + b).to_string()
/// }
///
/// // In user code:
/// let mut pipeline = ...;
/// pipeline.register_tools(tools![add]);
/// ```
#[proc_macro_attribute]
pub fn tool(_args: TokenStream, item: TokenStream) -> TokenStream {
    // Parse the source function.
    let input_fn = parse_macro_input!(item as ItemFn);

    let fn_name_ident = &input_fn.sig.ident;
    let fn_name_str = fn_name_ident.to_string();
    let wrapper_name = format_ident!("__{}_tool_wrapper", fn_name_ident);
    let tool_builder_name = format_ident!("__{}_tool_builder", fn_name_ident);

    // Doc comments become description.
    let description = extract_doc(&input_fn.attrs);

    // Gather parameter information.
    let mut schema_kv_pairs = Vec::new();
    let mut extraction_stmts = Vec::new();
    let mut call_args = Vec::new();

    // Traverse function arguments.
    for arg in input_fn.sig.inputs.iter() {
        if let FnArg::Typed(pat_type) = arg {
            if let Pat::Ident(pat_ident) = &*pat_type.pat {
                let param_name = &pat_ident.ident;
                let param_name_str = param_name.to_string();

                let ty = &*pat_type.ty;
                let json_type_str = type_to_json_type(ty);

                // Build the schema key-value pair
                schema_kv_pairs.push(quote! {
                    parameters.insert(#param_name_str.to_string(), #json_type_str.to_string());
                });

                // Extract the parameter from the HashMap at runtime
                extraction_stmts.push(quote! {
                    let #param_name: #ty = parameters
                        .remove(#param_name_str)
                        .expect("Missing parameter for tool call")
                        .parse()
                        .expect("Failed to parse parameter for tool call");
                });

                // Pass it to the original function
                call_args.push(quote! { #param_name });
            }
        }
    }

    // Generate the output tokens: keep original fn, plus wrapper + data.
    let expanded = quote! {
        // Keep the original function as-is
        #input_fn

        // Automatically generated wrapper that matches the `Tool` function signature.
        #[doc(hidden)]
        fn #wrapper_name(mut parameters: std::collections::HashMap<String, String>) -> String {
            #( #extraction_stmts )*
            #fn_name_ident( #(#call_args),* )
        }

        // Hidden function used by the tools! macro
        #[doc(hidden)]
        pub fn #tool_builder_name() -> transformers::pipelines::text_generation_pipeline::text_generation_model::Tool {
            let mut parameters = std::collections::HashMap::new();
            #( #schema_kv_pairs )*

            transformers::pipelines::text_generation_pipeline::text_generation_model::Tool::new(
                #fn_name_str.to_string(),
                #description.to_string(),
                parameters,
                #wrapper_name,
            )
        }

        // Generate a module with the same name as the function
        // This allows the tools! macro to find the builder
        #[doc(hidden)]
        pub mod #fn_name_ident {
            use super::*;

            pub fn __tool() -> transformers::pipelines::text_generation_pipeline::text_generation_model::Tool {
                #tool_builder_name()
            }
        }
    };

    TokenStream::from(expanded)
}
