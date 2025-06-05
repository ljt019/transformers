pub mod fill_mask_pipeline;
pub mod sentiment_analysis_pipeline;
pub mod text_generation_pipeline;
pub mod zero_shot_classification_pipeline;

// New modules for trait-based architecture
pub mod basic_pipeline;
pub mod capabilities;
pub mod combined_pipelines;
pub mod model_options;
pub mod tool_calling_pipeline;
pub mod tools;
pub mod traits;

use crate::Message;

// Re-export key types for convenience
pub use basic_pipeline::BasicPipeline;
pub use capabilities::{ModelCapabilities, ReasoningSupport};
pub use combined_pipelines::ToggleableReasoningToolsPipeline;
pub use model_options::{
    Gemma3ModelOptions, ModelOptionsType, Phi4ModelOptions, Qwen3ModelOptions,
};
pub use text_generation_pipeline::{
    Gemma3Size, Phi4Size, Qwen3Size, TextGenerationPipelineBuilder,
};
pub use tool_calling_pipeline::ToolCallingPipeline;
pub use tools::{Tool, ToolCall, ToolCallResult};
pub use traits::*;

pub trait TextGenerationModel {
    fn load_tokenizer(&self) -> anyhow::Result<tokenizers::Tokenizer>;

    fn get_eos_token_str(&self) -> &str;

    fn format_prompt(&self, prompt: &str) -> String;

    fn format_messages(&self, messages: Vec<Message>) -> anyhow::Result<String>;

    fn prompt_with_tokens(
        &self,
        tokens: &[u32],
        max_length: usize,
        eos_token: u32,
    ) -> anyhow::Result<Vec<u32>>;

    // New methods for advanced capabilities
    fn format_prompt_with_reasoning(&self, prompt: &str, reasoning_enabled: bool) -> String {
        // Default implementation - just use normal prompt formatting
        self.format_prompt(prompt)
    }

    fn format_prompt_with_tools(&self, prompt: &str, tools: &[Tool]) -> String {
        // Default implementation - just use normal prompt formatting
        self.format_prompt(prompt)
    }

    fn parse_tool_calls(&self, response: &str) -> anyhow::Result<Vec<ToolCall>> {
        // Default implementation returns empty vec
        Ok(vec![])
    }
}

pub trait FillMaskModel {
    fn fill_mask(prompt: &str) -> anyhow::Result<String>;
}

pub trait SentimentAnalysisModel {
    fn predict(text: &str) -> anyhow::Result<String>;
}

pub trait ZeroShotClassificationModel {
    fn predict(
        &self,
        premise: &str,
        candidate_labels: &[&str],
    ) -> anyhow::Result<Vec<(String, f32)>>;
}
