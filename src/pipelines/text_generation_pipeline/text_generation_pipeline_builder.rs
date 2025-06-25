use crate::models::quantized_gemma3::{Gemma3Model, Gemma3Size};
use crate::models::quantized_qwen3::{Qwen3Model, Qwen3Size};
use crate::pipelines::utils::model_cache::global_cache;

use super::text_generation_model::TextGenerationModel;
use super::text_generation_pipeline::TextGenerationPipeline;

pub struct TextGenerationPipelineBuilder<M: TextGenerationModel> {
    model_options: M::Options,
    temperature: f64,
    repeat_penalty: f32,
    repeat_last_n: usize,
    seed: u64,
    max_len: usize,
    top_p: f64,
    top_k: usize,
}

impl<M: TextGenerationModel> TextGenerationPipelineBuilder<M> {
    pub fn new(options: M::Options) -> Self {
        Self {
            model_options: options,
            temperature: 0.7,
            repeat_penalty: 1.0,
            repeat_last_n: 64,
            seed: 42,
            max_len: 1024,
            top_p: 1.0,
            top_k: 0,
        }
    }

    pub fn temperature(mut self, temperature: f64) -> Self {
        self.temperature = temperature;
        self
    }

    pub fn repeat_penalty(mut self, repeat_penalty: f32) -> Self {
        self.repeat_penalty = repeat_penalty;
        self
    }

    pub fn repeat_last_n(mut self, repeat_last_n: usize) -> Self {
        self.repeat_last_n = repeat_last_n;
        self
    }

    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    pub fn max_len(mut self, max_len: usize) -> Self {
        self.max_len = max_len;
        self
    }

    pub fn top_p(mut self, top_p: f64) -> Self {
        self.top_p = top_p.clamp(0.0, 1.0);
        self
    }

    pub fn top_k(mut self, top_k: usize) -> Self {
        self.top_k = top_k;
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

        let gen_params: crate::models::generation::GenerationParams =
            crate::models::generation::GenerationParams::new(
                self.temperature,
                1.0,
                64,
                42,
                self.max_len,
                self.top_p,
                self.top_k,
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
