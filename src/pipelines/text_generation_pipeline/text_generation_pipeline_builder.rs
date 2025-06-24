use crate::models::quantized_qwen3::{Qwen3Model, Qwen3Size};

use super::text_generation_model::TextGenerationModel;
use super::text_generation_pipeline::TextGenerationPipeline;

pub struct TextGenerationPipelineBuilder<M: TextGenerationModel> {
    model_options: M::Options,
    temperature: f64,
    repeat_penalty: f32,
    repeat_last_n: usize,
    seed: u64,
    max_len: usize,
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
        }
    }

    pub fn temperature(mut self, temperature: f64) -> Self {
        self.temperature = temperature;
        self
    }

    pub fn max_len(mut self, max_len: usize) -> Self {
        self.max_len = max_len;
        self
    }

    pub fn build(self) -> anyhow::Result<TextGenerationPipeline<M>> {
        let model = M::new(self.model_options);

        let gen_params: crate::models::generation::GenerationParams =
            crate::models::generation::GenerationParams::new(
                self.temperature,
                1.0,
                64,
                42,
                self.max_len,
            );

        Ok(TextGenerationPipeline::new(model, gen_params)?)
    }
}

impl TextGenerationPipelineBuilder<Qwen3Model> {
    pub fn qwen3(size: Qwen3Size) -> Self {
        Self::new(size)
    }
}
