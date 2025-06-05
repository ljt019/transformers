use super::{
    basic_pipeline::BasicPipeline,
    capabilities::{ModelCapabilities, ReasoningSupport},
    combined_pipelines::ToggleableReasoningToolsPipeline,
    text_generation_pipeline::{Gemma3Size, Phi4Size, Qwen3Size},
    tool_calling_pipeline::ToolCallingPipeline,
    TextGenerationModel,
};
use crate::models::gemma_3::QuantizedGemma3Model;
use crate::models::phi_4::QuantizedPhi4Model;
use crate::models::qwen_3::QuantizedQwen3Model;
use crate::models::raw::generation::GenerationParams;
use crate::utils::configs::ModelConfig;

/// Trait for model option types that can build their associated pipeline.
/// Each model type implements this to specify what pipeline it creates.
pub trait ModelOptionsType: Clone {
    type Pipeline;

    fn capabilities(&self) -> ModelCapabilities;
    fn build_pipeline(self, params: GenerationParams) -> anyhow::Result<Self::Pipeline>;
}

/// Model options for Qwen3 models with toggleable reasoning and tool calling.
#[derive(Clone)]
pub struct Qwen3ModelOptions {
    pub size: Qwen3Size,
}

impl Qwen3ModelOptions {
    pub fn new(size: Qwen3Size) -> Self {
        Self { size }
    }
}

impl ModelOptionsType for Qwen3ModelOptions {
    type Pipeline = ToggleableReasoningToolsPipeline;

    fn capabilities(&self) -> ModelCapabilities {
        ModelCapabilities::with_reasoning_and_tools(ReasoningSupport::Toggleable)
    }

    fn build_pipeline(self, params: GenerationParams) -> anyhow::Result<Self::Pipeline> {
        // Build the Qwen3 model
        let cfg = ModelConfig::new(params)?;
        let model: Box<dyn TextGenerationModel> =
            Box::new(QuantizedQwen3Model::new(cfg, self.size)?);

        // Load tokenizer and get EOS token
        let tokenizer = model.load_tokenizer()?;
        let eos_token_str = model.get_eos_token_str();
        let eos_token_encoding = tokenizer.encode(eos_token_str, false).map_err(|e| {
            anyhow::anyhow!("Failed to encode EOS token '{}': {}", eos_token_str, e)
        })?;
        let eos_ids = eos_token_encoding.get_ids();
        if eos_ids.len() != 1 {
            anyhow::bail!(
                "EOS token string '{}' did not tokenize to a single ID. Got: {:?}",
                eos_token_str,
                eos_ids
            );
        }
        let eos_token_id = eos_ids[0];

        Ok(ToggleableReasoningToolsPipeline::new(
            model,
            tokenizer,
            eos_token_id,
        ))
    }
}

/// Model options for Gemma3 models with tool calling only.
#[derive(Clone)]
pub struct Gemma3ModelOptions {
    pub size: Gemma3Size,
}

impl Gemma3ModelOptions {
    pub fn new(size: Gemma3Size) -> Self {
        Self { size }
    }
}

impl ModelOptionsType for Gemma3ModelOptions {
    type Pipeline = ToolCallingPipeline;

    fn capabilities(&self) -> ModelCapabilities {
        ModelCapabilities::with_tools()
    }

    fn build_pipeline(self, params: GenerationParams) -> anyhow::Result<Self::Pipeline> {
        // Build the Gemma3 model
        let cfg = ModelConfig::new(params)?;
        let model: Box<dyn TextGenerationModel> =
            Box::new(QuantizedGemma3Model::new(cfg, self.size)?);

        // Load tokenizer and get EOS token
        let tokenizer = model.load_tokenizer()?;
        let eos_token_str = model.get_eos_token_str();
        let eos_token_encoding = tokenizer.encode(eos_token_str, false).map_err(|e| {
            anyhow::anyhow!("Failed to encode EOS token '{}': {}", eos_token_str, e)
        })?;
        let eos_ids = eos_token_encoding.get_ids();
        if eos_ids.len() != 1 {
            anyhow::bail!(
                "EOS token string '{}' did not tokenize to a single ID. Got: {:?}",
                eos_token_str,
                eos_ids
            );
        }
        let eos_token_id = eos_ids[0];

        Ok(ToolCallingPipeline::new(model, tokenizer, eos_token_id))
    }
}

/// Model options for Phi4 models with basic generation only.
#[derive(Clone)]
pub struct Phi4ModelOptions {
    pub size: Phi4Size,
}

impl Phi4ModelOptions {
    pub fn new(size: Phi4Size) -> Self {
        Self { size }
    }
}

impl ModelOptionsType for Phi4ModelOptions {
    type Pipeline = BasicPipeline;

    fn capabilities(&self) -> ModelCapabilities {
        ModelCapabilities::basic()
    }

    fn build_pipeline(self, params: GenerationParams) -> anyhow::Result<Self::Pipeline> {
        // Build the Phi4 model
        let cfg = ModelConfig::new(params)?;
        let model: Box<dyn TextGenerationModel> =
            Box::new(QuantizedPhi4Model::new(cfg, self.size)?);

        // Load tokenizer and get EOS token
        let tokenizer = model.load_tokenizer()?;
        let eos_token_str = model.get_eos_token_str();
        let eos_token_encoding = tokenizer.encode(eos_token_str, false).map_err(|e| {
            anyhow::anyhow!("Failed to encode EOS token '{}': {}", eos_token_str, e)
        })?;
        let eos_ids = eos_token_encoding.get_ids();
        if eos_ids.len() != 1 {
            anyhow::bail!(
                "EOS token string '{}' did not tokenize to a single ID. Got: {:?}",
                eos_token_str,
                eos_ids
            );
        }
        let eos_token_id = eos_ids[0];

        Ok(BasicPipeline::new(model, tokenizer, eos_token_id))
    }
}
