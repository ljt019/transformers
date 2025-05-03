use crate::models::quantized::gemma_3_quantized::QuantizedGemma3Model;
use crate::models::quantized::phi_4_quantized::QuantizedPhi4Model;
use crate::models::quantized::qwen3_quantized::QuantizedQwen3Model;
use crate::utils::ModelConfig;

use crate::utils::{GenerationParams, HfConfig};
use tokenizers::Tokenizer;

/// Available Gemma3 model sizes (e.g., 1B, 4B, 12B, 27B).
pub enum Gemma3Size {
    Size1B,
    Size4B,
    Size12B,
    Size27B,
}

/// Available Phi4 model sizes (e.g., 14B).
pub enum Phi4Size {
    Size14B,
}

// Available Qwen3 model sizes (e.g., 0.6B, 1.7B, 4B, 8B, 14B, 32B)
// None of the moe models are supported yet.
pub enum Qwen3Size {
    Size0_6B,
    Size1_7B,
    Size4B,
    Size8B,
    Size14B,
    Size32B,
}

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
    Phi4(Phi4Size),
    Qwen3(Qwen3Size),
}

impl ModelOptions {
    /// Map each ModelOptions variant to its HfConfig
    pub fn hf_config(&self) -> HfConfig {
        match self {
            ModelOptions::Gemma3(size) => match size {
                Gemma3Size::Size1B => HfConfig::new(
                    "google/gemma-3-1b-it",
                    "tokenizer.json",
                    "unsloth/gemma-3-1b-it-GGUF",
                    "gemma-3-1b-it-Q4_K_M.gguf",
                ),
                Gemma3Size::Size4B => HfConfig::new(
                    "google/gemma-3-4b-it",
                    "tokenizer.json",
                    "unsloth/gemma-3-4b-it-GGUF",
                    "gemma-3-4b-it-Q4_K_M.gguf",
                ),
                Gemma3Size::Size12B => HfConfig::new(
                    "google/gemma-3-12b-it",
                    "tokenizer.json",
                    "unsloth/gemma-3-12b-it-GGUF",
                    "gemma-3-12b-it-Q4_K_M.gguf",
                ),
                Gemma3Size::Size27B => HfConfig::new(
                    "google/gemma-3-27b-it",
                    "tokenizer.json",
                    "unsloth/gemma-3-27b-it-GGUF",
                    "gemma-3-27b-it-Q4_K_M.gguf",
                ),
            },
            ModelOptions::Phi4(size) => match size {
                Phi4Size::Size14B => HfConfig::new(
                    "microsoft/phi-4",
                    "tokenizer.json",
                    "microsoft/phi-4-gguf",
                    "phi-4-q4.gguf",
                ),
            },
            ModelOptions::Qwen3(size) => match size {
                Qwen3Size::Size0_6B => HfConfig::new(
                    "Qwen/Qwen3-0.6B",
                    "tokenizer.json",
                    "unsloth/Qwen3-0.6B-GGUF",
                    "Qwen3-0.6B-Q4_K_M.gguf",
                ),
                Qwen3Size::Size1_7B => HfConfig::new(
                    "Qwen/Qwen3-1.7B",
                    "tokenizer.json",
                    "unsloth/Qwen3-1.7B-GGUF",
                    "Qwen3-1.7B-Q4_K_M.gguf",
                ),
                Qwen3Size::Size4B => HfConfig::new(
                    "Qwen/Qwen3-4B",
                    "tokenizer.json",
                    "unsloth/Qwen3-4B-GGUF",
                    "Qwen3-4B-Q4_K_M.gguf",
                ),
                Qwen3Size::Size8B => HfConfig::new(
                    "Qwen/Qwen3-8B",
                    "tokenizer.json",
                    "unsloth/Qwen3-8B-GGUF",
                    "Qwen3-8B-Q4_K_M.gguf",
                ),
                Qwen3Size::Size14B => HfConfig::new(
                    "Qwen/Qwen3-14B",
                    "tokenizer.json",
                    "unsloth/Qwen3-14B-GGUF",
                    "Qwen3-14B-Q4_K_M.gguf",
                ),
                Qwen3Size::Size32B => HfConfig::new(
                    "Qwen/Qwen3-32B",
                    "tokenizer.json",
                    "unsloth/Qwen3-32B-GGUF",
                    "Qwen3-32B-Q4_K_M.gguf",
                ),
            },
        }
    }

    /// Construct the quantized model instance and return with its HfConfig
    pub(crate) fn build_model(
        self,
        params: GenerationParams,
    ) -> anyhow::Result<(HfConfig, Box<dyn LargeLanguageModel>)> {
        let hf = self.hf_config();
        let cfg = ModelConfig::new(params, hf.clone())?;
        let model: Box<dyn LargeLanguageModel> = match self {
            ModelOptions::Gemma3(_) => Box::new(QuantizedGemma3Model::new(cfg)?),
            ModelOptions::Phi4(_) => Box::new(QuantizedPhi4Model::new(cfg)?),
            ModelOptions::Qwen3(_) => Box::new(QuantizedQwen3Model::new(cfg)?),
        };
        Ok((hf, model))
    }
}

/// Builder for configuring and constructing a text generation pipeline.
///
/// Start by creating a builder with `new(ModelOptions)`, then chain optional settings:
/// - `.temperature(f64)`: sampling temperature (default: DEFAULT_TEMPERATURE)
/// - `.repeat_penalty(f32)`: penalty for repeated tokens (default: DEFAULT_REPEAT_PENALTY)
/// - `.repeat_last_n(usize)`: context length for repeat penalty (default: DEFAULT_REPEAT_LAST_N)
/// - `.seed(u64)`: random seed (default: DEFAULT_SEED)
/// - `.use_flash_attn(bool)`: enable flash attention for supported models (default: false)
///
/// Finally, call `.build()` to obtain a `TextGenerationPipeline`.
pub struct TextGenerationPipelineBuilder {
    model_choice: ModelOptions,
    temperature: Option<f64>,
    repeat_penalty: Option<f32>,
    repeat_last_n: Option<usize>,
    seed: Option<u64>,
    use_flash_attn: Option<bool>,
}

impl TextGenerationPipelineBuilder {
    pub fn new(model_choice: ModelOptions) -> Self {
        Self {
            model_choice,
            temperature: None,
            repeat_penalty: None,
            repeat_last_n: None,
            seed: None,
            use_flash_attn: None,
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

    pub fn use_flash_attn(mut self, use_flash_attn: bool) -> Self {
        self.use_flash_attn = Some(use_flash_attn);
        self
    }

    pub fn build(self) -> anyhow::Result<TextGenerationPipeline> {
        let temperature = self.temperature.unwrap_or(crate::DEFAULT_TEMPERATURE);
        let repeat_penalty = self.repeat_penalty.unwrap_or(crate::DEFAULT_REPEAT_PENALTY);
        let repeat_last_n = self.repeat_last_n.unwrap_or(crate::DEFAULT_REPEAT_LAST_N);
        let seed = self.seed.unwrap_or(crate::DEFAULT_SEED);
        let use_flash_attn = self.use_flash_attn.unwrap_or(false);

        let generation_params = GenerationParams::new(
            temperature,
            repeat_penalty,
            repeat_last_n,
            seed,
            use_flash_attn,
        );

        // Build the HfConfig and model in one go via the nested enum helper
        let (hf_config, model) = self.model_choice.build_model(generation_params)?;

        let tokenizer = crate::utils::load_tokenizer(&hf_config)?;

        Ok(TextGenerationPipeline { model, tokenizer })
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
    model: Box<dyn LargeLanguageModel>,
    tokenizer: Tokenizer,
}

impl TextGenerationPipeline {
    pub fn generate_text(&self, prompt: &str, max_length: usize) -> anyhow::Result<String> {
        self.model.prompt_model(&self.tokenizer, prompt, max_length)
    }
}

pub(crate) trait LargeLanguageModel {
    fn prompt_model(
        &self,
        tokenizer: &Tokenizer,
        prompt: &str,
        max_length: usize,
    ) -> anyhow::Result<String>;
}
