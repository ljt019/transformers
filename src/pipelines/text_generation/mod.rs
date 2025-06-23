// Text generation pipeline module
pub mod basic_pipeline;
pub mod builder;
pub mod capabilities;
pub mod combined_pipelines;
pub mod model_options;
pub mod tool_calling_pipeline;
pub mod tools;
pub mod traits;

// Re-export key types for convenience
pub use basic_pipeline::BasicPipeline;
pub use builder::TextGenerationPipelineBuilder;
pub use capabilities::{ModelCapabilities, ReasoningSupport};
pub use combined_pipelines::ToggleableReasoningToolsPipeline;
pub use model_options::{
    Gemma3ModelOptions, Gemma3Size, ModelOptionsType, Phi4ModelOptions, Phi4Size,
    Qwen3ModelOptions, Qwen3Size,
};
pub use tool_calling_pipeline::ToolCallingPipeline;
pub use tools::{Tool, ToolCall, ToolCallResult};
pub use traits::*;
