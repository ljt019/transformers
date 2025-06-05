use super::tools::{Tool, ToolCallResult};
use crate::Message;

/// Base trait that all text generation pipelines must implement.
/// Provides the core text generation functionality.
pub trait TextGenerationPipeline {
    /// Generate text completion from a prompt string.
    fn prompt_completion(&self, prompt: &str, max_length: usize) -> anyhow::Result<String>;

    /// Generate text completion from a conversation using the Message format.
    fn message_completion(
        &self,
        messages: Vec<Message>,
        max_length: usize,
    ) -> anyhow::Result<String>;
}

/// Trait for pipelines that support always-on reasoning.
/// These models always use internal reasoning and may expose the reasoning trace.
pub trait TextGenerationPipelineWithReasoning: TextGenerationPipeline {
    /// Get the reasoning trace from the last generation, if available.
    fn get_reasoning_trace(&self) -> Option<&str>;
}

/// Trait for pipelines that support toggleable reasoning.
/// These models can enable/disable their internal reasoning process.
pub trait TextGenerationPipelineWithToggleableReasoning: TextGenerationPipeline {
    /// Enable reasoning for future generations.
    fn enable_reasoning(&mut self);

    /// Disable reasoning for future generations.
    fn disable_reasoning(&mut self);

    /// Check if reasoning is currently enabled.
    fn is_reasoning_enabled(&self) -> bool;

    /// Get the reasoning trace from the last generation, if reasoning was enabled.
    fn get_reasoning_trace(&self) -> Option<&str> {
        None // Default implementation
    }
}

/// Trait for pipelines that support tool calling.
/// These models can register tools and invoke them during generation.
pub trait TextGenerationPipelineWithTools: TextGenerationPipeline {
    /// Register a tool that the model can call during generation.
    fn register_tool(&mut self, tool: Tool) -> anyhow::Result<()>;

    /// Unregister a tool by name.
    fn unregister_tool(&mut self, tool_name: &str) -> anyhow::Result<()>;

    /// List all currently registered tools.
    fn list_tools(&self) -> Vec<&Tool>;

    /// Generate text with access to registered tools.
    /// The model may call tools as part of its response.
    fn call_with_tools(&self, prompt: &str, max_length: usize) -> anyhow::Result<ToolCallResult>;

    /// Generate text from messages with access to registered tools.
    fn call_with_tools_messages(
        &self,
        messages: Vec<Message>,
        max_length: usize,
    ) -> anyhow::Result<ToolCallResult>;
}
