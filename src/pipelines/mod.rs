// Pipeline modules organized by functionality
pub mod fill_mask;
pub mod sentiment_analysis;
pub mod text_generation;
pub mod zero_shot_classification;

use crate::Message;

// Re-export text generation types for convenience
pub use text_generation::*;

// Re-export other pipeline types
pub use fill_mask::*;
pub use sentiment_analysis::*;
pub use zero_shot_classification::*;

/// Core trait that all text generation models must implement.
/// This provides the basic interface for loading tokenizers, formatting prompts,
/// and generating text with tokens.
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

    // Advanced capability methods with default implementations
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

/// Trait for models that support fill mask functionality.
pub trait FillMaskModel {
    fn fill_mask(prompt: &str) -> anyhow::Result<String>;
}

/// Trait for models that support sentiment analysis.
pub trait SentimentAnalysisModel {
    fn predict(text: &str) -> anyhow::Result<String>;
}

/// Trait for models that support zero-shot classification.
pub trait ZeroShotClassificationModel {
    fn predict(
        &self,
        premise: &str,
        candidate_labels: &[&str],
    ) -> anyhow::Result<Vec<(String, f32)>>;
}
