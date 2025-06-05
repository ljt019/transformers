// This file will contain the new generic builder once we implement the ModelOptionsType trait
// For now, we keep the size enums here temporarily until they're moved to model_options.rs

// Generic builder for text generation pipelines
use super::model_options::ModelOptionsType;
use crate::models::raw::generation::GenerationParams;

// Re-export size enums for convenience
pub use super::model_options::{Gemma3ModelOptions, Phi4ModelOptions, Qwen3ModelOptions};

// Size enums
#[derive(Clone)]
pub enum Qwen3Size {
    Size0_6B,
    Size1_7B,
    Size4B,
    Size8B,
    Size14B,
    Size32B,
}

#[derive(Clone)]
pub enum Gemma3Size {
    Size1B,
    Size4B,
    Size12B,
    Size27B,
}

#[derive(Clone)]
pub enum Phi4Size {
    Size14B,
}

/// Generic builder for configuring and constructing text generation pipelines.
///
/// The builder is generic over the model options type, which determines what
/// type of pipeline will be returned. This provides compile-time type safety
/// ensuring users only get methods their chosen model supports.
///
/// # Examples
///
/// ## Basic Model (Phi4)
/// ```rust
/// use transformers::pipelines::{TextGenerationPipelineBuilder, Phi4ModelOptions, Phi4Size};
///
/// let pipeline = TextGenerationPipelineBuilder::new(
///     Phi4ModelOptions::new(Phi4Size::Size14B)
/// )
/// .temperature(0.7)
/// .build()?;
///
/// // Only basic methods available
/// let response = pipeline.prompt_completion("Hello, world!", 100)?;
/// ```
///
/// ## Tool Calling Model (Gemma3)
/// ```rust
/// use transformers::pipelines::{TextGenerationPipelineBuilder, Gemma3ModelOptions, Gemma3Size};
///
/// let mut pipeline = TextGenerationPipelineBuilder::new(
///     Gemma3ModelOptions::new(Gemma3Size::Size4B)
/// )
/// .build()?;
///
/// // Tool methods are available
/// pipeline.register_tool(my_tool)?;
/// let result = pipeline.call_with_tools("What is 2+2?", 100)?;
/// ```
///
/// ## Full Featured Model (Qwen3)
/// ```rust
/// use transformers::pipelines::{TextGenerationPipelineBuilder, Qwen3ModelOptions, Qwen3Size};
///
/// let mut pipeline = TextGenerationPipelineBuilder::new(
///     Qwen3ModelOptions::new(Qwen3Size::Size8B)
/// )
/// .build()?;
///
/// // Both reasoning and tool methods available
/// pipeline.enable_reasoning();
/// pipeline.register_tool(my_tool)?;
/// let result = pipeline.call_with_tools("Solve this step by step: ...", 500)?;
/// ```
pub struct TextGenerationPipelineBuilder<M: ModelOptionsType> {
    model_options: M,
    temperature: Option<f64>,
    repeat_penalty: Option<f32>,
    repeat_last_n: Option<usize>,
    seed: Option<u64>,
}

impl<M: ModelOptionsType> TextGenerationPipelineBuilder<M> {
    /// Create a new builder with the specified model options.
    ///
    /// The model options determine what type of pipeline will be created
    /// and what capabilities it will have.
    pub fn new(model_options: M) -> Self {
        Self {
            model_options,
            temperature: None,
            repeat_penalty: None,
            repeat_last_n: None,
            seed: None,
        }
    }

    /// Set the sampling temperature (default: 0.7).
    ///
    /// Higher values make output more random, lower values more deterministic.
    pub fn temperature(mut self, temperature: f64) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Set the repeat penalty (default: 1.1).
    ///
    /// Values > 1.0 discourage repetition, values < 1.0 encourage it.
    pub fn repeat_penalty(mut self, repeat_penalty: f32) -> Self {
        self.repeat_penalty = Some(repeat_penalty);
        self
    }

    /// Set the repeat penalty context length (default: 64).
    ///
    /// Number of previous tokens to consider when applying repeat penalty.
    pub fn repeat_last_n(mut self, repeat_last_n: usize) -> Self {
        self.repeat_last_n = Some(repeat_last_n);
        self
    }

    /// Set the random seed for reproducible generation (default: 299792458).
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Build the text generation pipeline.
    ///
    /// The return type depends on the model options provided to the builder.
    /// Each model type returns a pipeline with the appropriate capabilities.
    pub fn build(self) -> anyhow::Result<M::Pipeline> {
        let temperature = self.temperature.unwrap_or(crate::DEFAULT_TEMPERATURE);
        let repeat_penalty = self.repeat_penalty.unwrap_or(crate::DEFAULT_REPEAT_PENALTY);
        let repeat_last_n = self.repeat_last_n.unwrap_or(crate::DEFAULT_REPEAT_LAST_N);
        let seed = self.seed.unwrap_or(crate::DEFAULT_SEED);

        let generation_params =
            GenerationParams::new(temperature, repeat_penalty, repeat_last_n, seed);

        // Delegate to the model options to build the appropriate pipeline
        self.model_options.build_pipeline(generation_params)
    }
}
