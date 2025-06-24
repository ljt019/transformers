use crate::Message;
use candle_core::Tensor;

/// Minimal interface required by the text-generation pipeline for a model context.
///
/// Both `Qwen3Model::Context` and `Gemma3Model::Context` already expose compatible
/// `generate` and `reset` methods, so we only need a thin trait wrapper that the
/// pipeline can work with generically.
pub trait LanguageModelContext {
    /// Forward the input tokens through the model, returning the logits for the
    /// next token.
    fn generate(&mut self, input: &Tensor) -> candle_core::Result<Tensor>;

    /// Clear the internal state (kv-cache, position, etc.).
    fn reset(&mut self);
}

pub trait TextGenerationModel {
    /// Type used to configure model loading (e.g. which checkpoint size).
    type Options;
    /// The context type that will be returned by `new_context` and consumed by
    /// the pipeline. It must implement [`LanguageModelContext`].
    type Context: LanguageModelContext;

    fn new(options: Self::Options) -> Self;

    fn get_tokenizer(&self) -> anyhow::Result<tokenizers::Tokenizer>;

    fn apply_chat_template(&self, messages: &[Message]) -> anyhow::Result<String>;

    fn get_eos_token(&self) -> u32;

    fn new_context(&self) -> Self::Context;

    fn clear_context(&self, context: &mut Self::Context) -> anyhow::Result<()>;
}

pub trait Reasoning {}

pub trait ToggleableReasoning {
    fn set_reasoning(&mut self, enable: bool) -> anyhow::Result<()>;
}

pub trait ToolCalling {
    fn register_tool(&mut self, tool: String);
    fn toggle_tools(&mut self, enable: bool);
}
