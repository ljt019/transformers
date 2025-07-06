pub mod core;
mod loaders;
mod models;
pub mod pipelines;

// Re-export the `#[tool]` procedural macro so users can simply write
// `use transformers::tool;` and annotate their functions without adding an
// explicit dependency on the `tool_macro` crate.
// The macro lives in the separate `tool_macro` crate to avoid a proc-macro/
// normal crate cyclic dependency, but re-exporting it here keeps the public
// API surface of `transformers` ergonomic.

pub use tool_macro::tool;

// Re-export core types
pub use core::{Message, MessageVecExt};
