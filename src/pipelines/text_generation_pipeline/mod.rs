pub mod text_generation_model;
pub mod text_generation_pipeline;
pub mod text_generation_pipeline_builder;
pub mod tool_error;
pub mod xml_parser;
pub mod base_pipeline;
pub mod xml_generation_pipeline;

pub use text_generation_pipeline_builder::TextGenerationPipelineBuilder;
pub use text_generation_pipeline::{TextGenerationPipeline, Input};
pub use xml_generation_pipeline::XmlGenerationPipeline;

// Convenience re-exports so users can simply
// `use transformers::pipelines::text_generation_pipeline::*;` and access
// the common model size enums and the `#[tool]` macro without additional
// import clutter.

pub use crate::models::quantized_gemma3::Gemma3Size;
pub use crate::models::quantized_qwen3::Qwen3Size;

// Re-export the procedural macro (functions as an item in Rust 2018+).
pub use crate::tool;

// Re-export `futures::StreamExt` so users iterating over streaming outputs
// get the `next`/`try_next` extension methods automatically when they
// glob-import this module.
pub use futures::StreamExt;
pub use futures::TryStreamExt;

// Re-export commonly used types and traits
pub use crate::{Message, MessageVecExt};

// Re-export Result type for convenience
pub use anyhow::Result;

// Re-export std::io::Write for flushing stdout in examples
pub use std::io::Write;

pub use tool_error::ToolError;
pub use xml_parser::{Event, XmlParser, XmlParserBuilder};

#[macro_export]
macro_rules! tools {
    ($($tool:ident),+ $(,)?) => {
        vec![
            $(
                $tool::__tool()
            ),+
        ]
    };
}

pub use tools;
