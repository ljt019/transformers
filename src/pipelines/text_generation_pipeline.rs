use crate::models::gemma_3::QuantizedGemma3Model;
use crate::models::phi_4::QuantizedPhi4Model;
use crate::models::qwen_3::QuantizedQwen3Model;
use crate::utils::configs::ModelConfig;

pub use crate::models::gemma_3::Gemma3Size;
pub use crate::models::phi_4::Phi4Size;
pub use crate::models::qwen_3::Qwen3Size;

use crate::models::raw::generation::GenerationParams;
use tokenizers::Tokenizer;

use super::TextGenerationModel;

use crate::Messages;

/// High-level selection of model family/architecture.
///
/// You must also specify the size of the model you want to use.
///
/// Example:
/// ```rust
/// use transformers::pipelines::text_generation_pipeline::{ModelOptions, Qwen3Size};
///
/// let model_choice = ModelOptions::Qwen3(Qwen3Size::Size0_6B);
/// ```
pub enum ModelOptions {
    Gemma3(Gemma3Size),
    Qwen3(Qwen3Size),
    Phi4(Phi4Size),
}

impl ModelOptions {
    /// Construct the quantized model instance and return with its HfConfig
    pub(crate) fn build_model(
        self,
        params: GenerationParams,
    ) -> anyhow::Result<Box<dyn TextGenerationModel>> {
        let cfg = ModelConfig::new(params)?;
        let model: Box<dyn TextGenerationModel> = match self {
            ModelOptions::Gemma3(size) => Box::new(QuantizedGemma3Model::new(cfg, size)?),
            ModelOptions::Qwen3(size) => Box::new(QuantizedQwen3Model::new(cfg, size)?),
            ModelOptions::Phi4(size) => Box::new(QuantizedPhi4Model::new(cfg, size)?),
        };
        Ok(model)
    }
}

/// Builder for configuring and constructing a text generation pipeline.
///
/// Start by creating a builder with `new(ModelOptions)`, then chain optional settings:
/// - `.temperature(f64)`: sampling temperature (default: DEFAULT_TEMPERATURE)
/// - `.repeat_penalty(f32)`: penalty for repeated tokens (default: DEFAULT_REPEAT_PENALTY)
/// - `.repeat_last_n(usize)`: context length for repeat penalty (default: DEFAULT_REPEAT_LAST_N)
/// - `.seed(u64)`: random seed (default: DEFAULT_SEED)
///
/// Finally, call `.build()` to obtain a `TextGenerationPipeline`.
pub struct TextGenerationPipelineBuilder {
    model_choice: ModelOptions,
    temperature: Option<f64>,
    repeat_penalty: Option<f32>,
    repeat_last_n: Option<usize>,
    seed: Option<u64>,
}

impl TextGenerationPipelineBuilder {
    pub fn new(model_choice: ModelOptions) -> Self {
        Self {
            model_choice,
            temperature: None,
            repeat_penalty: None,
            repeat_last_n: None,
            seed: None,
        }
    }

    pub fn temperature(mut self, temperature: f64) -> Self {
        self.temperature = Some(temperature);
        self
    }

    pub fn repeat_penalty(mut self, repeat_penalty: f32) -> Self {
        self.repeat_penalty = Some(repeat_penalty);
        self
    }

    pub fn repeat_last_n(mut self, repeat_last_n: usize) -> Self {
        self.repeat_last_n = Some(repeat_last_n);
        self
    }

    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    pub fn build(self) -> anyhow::Result<TextGenerationPipeline> {
        let temperature = self.temperature.unwrap_or(crate::DEFAULT_TEMPERATURE);
        let repeat_penalty = self.repeat_penalty.unwrap_or(crate::DEFAULT_REPEAT_PENALTY);
        let repeat_last_n = self.repeat_last_n.unwrap_or(crate::DEFAULT_REPEAT_LAST_N);
        let seed = self.seed.unwrap_or(crate::DEFAULT_SEED);

        let generation_params =
            GenerationParams::new(temperature, repeat_penalty, repeat_last_n, seed);

        let model = self.model_choice.build_model(generation_params)?;
        let tokenizer = model.load_tokenizer()?;

        // Get EOS token ID
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

        Ok(TextGenerationPipeline {
            model,
            tokenizer,
            eos_token_id,
        })
    }
}

/// A ready-to-use pipeline for generating text using a quantized model.
///
/// After building with `TextGenerationPipelineBuilder`, call
/// `generate_text(prompt, max_length)` to produce text completions.
///
/// Example:
/// ```rust
/// use transformers::pipelines::text_generation_pipeline::{TextGenerationPipelineBuilder, ModelOptions, Gemma3Size};
///
/// let pipeline = TextGenerationPipelineBuilder::new(
///     ModelOptions::Gemma3(Gemma3Size::Size1B),
/// )
/// .temperature(0.7)
/// .build().unwrap();
///
/// let output = pipeline.generate_text("What is the meaning of life?", 5).unwrap();
///
/// println!("{}", output);
/// ```
pub struct TextGenerationPipeline {
    /// Tokenizer corresponding to the model's vocabulary.
    model: Box<dyn TextGenerationModel>,
    tokenizer: Tokenizer,
    eos_token_id: u32,
}

impl TextGenerationPipeline {
    pub fn generate_text(&self, prompt: &str, max_length: usize) -> anyhow::Result<String> {
        // Format the prompt
        let formatted_prompt = self.model.format_prompt(prompt);

        // Turn the prompt into tokens
        let prompt_tokens = self.tokenizer.encode(formatted_prompt, true).unwrap();

        // Generate the response with the prompt tokens
        let response_as_tokens = self
            .model
            .prompt_with_tokens(&prompt_tokens.get_ids(), max_length, self.eos_token_id)
            .unwrap();

        // Turn the response tokens back into a string
        let response = self.tokenizer.decode(&response_as_tokens, true).unwrap();

        Ok(response)
    }

    pub fn generate_chat(&self, messages: Messages, max_length: usize) -> anyhow::Result<String> {
        // Format the messages
        let formatted_messages = self.model.format_messages(messages);

        // Turn the prompt into tokens
        let prompt_tokens = self.tokenizer.encode(formatted_messages, true).unwrap();

        // Generate the response with the prompt tokens
        let response_as_tokens = self
            .model
            .prompt_with_tokens(&prompt_tokens.get_ids(), max_length, self.eos_token_id)
            .unwrap();

        // Turn the response tokens back into a string
        let response = self.tokenizer.decode(&response_as_tokens, true).unwrap();

        Ok(response)
    }
}
