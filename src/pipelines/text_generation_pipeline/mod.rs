pub mod text_generation_model;
pub mod text_generation_pipeline;
pub mod text_generation_pipeline_builder;
pub mod tool_error;
pub mod xml_parser;

pub use text_generation_pipeline_builder::TextGenerationPipelineBuilder;

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

/// Output from text generation that can be either raw strings or parsed XML events
#[derive(Debug, Clone, PartialEq)]
pub enum Output {
    /// Raw text output (when no XML parser is registered)
    Text(String),
    /// Parsed XML events (when an XML parser is registered)
    Events(Vec<Event>),
}

impl Output {
    /// Get the raw text content, regardless of output type
    pub fn as_text(&self) -> String {
        match self {
            Output::Text(text) => text.clone(),
            Output::Events(events) => {
                events.iter()
                    .map(|e| e.get_content())
                    .collect::<Vec<_>>()
                    .join("")
            }
        }
    }
    
    /// Get events if this is an Events output, None otherwise
    pub fn as_events(&self) -> Option<&Vec<Event>> {
        match self {
            Output::Events(events) => Some(events),
            Output::Text(_) => None,
        }
    }
    
    /// Check if this output contains events
    pub fn has_events(&self) -> bool {
        matches!(self, Output::Events(_))
    }
}

/// Streaming output that can be either strings or events
#[derive(Debug, Clone, PartialEq)]
pub enum StreamOutput {
    /// Raw text chunk (when no XML parser is registered)
    Text(String),
    /// A single XML event (when an XML parser is registered)
    Event(Event),
}

impl StreamOutput {
    /// Get the text content of this output
    pub fn as_text(&self) -> &str {
        match self {
            StreamOutput::Text(text) => text,
            StreamOutput::Event(event) => event.get_content(),
        }
    }
    
    /// Get the event if this is an Event output
    pub fn as_event(&self) -> Option<&Event> {
        match self {
            StreamOutput::Event(event) => Some(event),
            StreamOutput::Text(_) => None,
        }
    }
}

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
