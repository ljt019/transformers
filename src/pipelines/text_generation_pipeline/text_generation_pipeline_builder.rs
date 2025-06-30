use crate::models::quantized_gemma3::{Gemma3Model, Gemma3Size};
use crate::models::quantized_qwen3::{Qwen3Model, Qwen3Size};
use crate::pipelines::utils::model_cache::global_cache;

use super::text_generation_model::TextGenerationModel;
use super::text_generation_pipeline::TextGenerationPipeline;

pub struct TextGenerationPipelineBuilder<M: TextGenerationModel> {
    model_options: M::Options,
    temperature: Option<f64>,
    repeat_penalty: Option<f32>,
    repeat_last_n: Option<usize>,
    seed: Option<u64>,
    max_len: Option<usize>,
    top_p: Option<f64>,
    top_k: Option<usize>,
    min_p: Option<f64>,
}

impl<M: TextGenerationModel> TextGenerationPipelineBuilder<M> {
    pub fn new(options: M::Options) -> Self {
        Self {
            model_options: options,
            temperature: None,
            repeat_penalty: None,
            repeat_last_n: None,
            seed: None,
            max_len: None,
            top_p: None,
            top_k: None,
            min_p: None,
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

    pub fn max_len(mut self, max_len: usize) -> Self {
        self.max_len = Some(max_len);
        self
    }

    pub fn top_p(mut self, top_p: f64) -> Self {
        self.top_p = Some(top_p.clamp(0.0, 1.0));
        self
    }

    pub fn top_k(mut self, top_k: usize) -> Self {
        self.top_k = Some(top_k);
        self
    }

    pub fn min_p(mut self, min_p: f64) -> Self {
        self.min_p = Some(min_p.clamp(0.0, 1.0));
        self
    }

    pub fn build(self) -> anyhow::Result<TextGenerationPipeline<M>>
    where
        M: Clone + Send + Sync + 'static,
        M::Options: std::fmt::Display,
    {
        // Always use the global cache to share models
        let cache_key = format!("{}", self.model_options);
        let model = global_cache().get_or_create(&cache_key, || Ok(M::new(self.model_options)))?;

        // Start with model-specific defaults
        let default_params = model.default_generation_params();

        // Override with any user-specified values
        let gen_params = crate::models::generation::GenerationParams::new(
            self.temperature.unwrap_or(default_params.temperature),
            self.repeat_penalty.unwrap_or(default_params.repeat_penalty),
            self.repeat_last_n.unwrap_or(default_params.repeat_last_n),
            self.seed.unwrap_or_else(|| rand::random::<u64>()),
            self.max_len.unwrap_or(default_params.max_len),
            self.top_p.unwrap_or(default_params.top_p),
            self.top_k.unwrap_or(default_params.top_k),
            self.min_p.unwrap_or(default_params.min_p),
        );

        Ok(TextGenerationPipeline::new(model, gen_params)?)
    }
}

impl TextGenerationPipelineBuilder<Qwen3Model> {
    pub fn qwen3(size: Qwen3Size) -> Self {
        Self::new(size)
    }
}

impl TextGenerationPipelineBuilder<Gemma3Model> {
    pub fn gemma3(size: Gemma3Size) -> Self {
        Self::new(size)
    }
}
