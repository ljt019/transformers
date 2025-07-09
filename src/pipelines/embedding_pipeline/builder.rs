use super::model::EmbeddingModel;
use super::pipeline::EmbeddingPipeline;
use crate::core::ModelOptions;
use crate::pipelines::utils::{BasePipelineBuilder, StandardPipelineBuilder};
use std::sync::Arc;

/// Builder for embedding pipelines
pub type EmbeddingPipelineBuilder<M> = StandardPipelineBuilder<<M as EmbeddingModel>::Options>;

impl<M> BasePipelineBuilder<M> for StandardPipelineBuilder<M::Options>
where
    M: EmbeddingModel + Clone + Send + Sync + 'static,
    M::Options: ModelOptions + Clone,
{
    type Model = M;
    type Pipeline = EmbeddingPipeline<M>;
    type Options = M::Options;

    fn options(&self) -> &Self::Options {
        &self.options
    }
    
    fn device_request(&self) -> &crate::pipelines::utils::DeviceRequest {
        &self.device_request
    }

    fn create_model(options: Self::Options, device: candle_core::Device) -> anyhow::Result<M> {
        M::new(options, device)
    }
    
    fn get_tokenizer(options: Self::Options) -> anyhow::Result<tokenizers::Tokenizer> {
        M::get_tokenizer(options)
    }
    
    fn construct_pipeline(model: M, tokenizer: tokenizers::Tokenizer) -> anyhow::Result<Self::Pipeline> {
        Ok(EmbeddingPipeline {
            model: Arc::new(model),
            tokenizer,
        })
    }
}

impl StandardPipelineBuilder<crate::models::implementations::qwen3_embeddings::Qwen3EmbeddingSize> {
    pub fn qwen3(size: crate::models::implementations::qwen3_embeddings::Qwen3EmbeddingSize) -> Self {
        Self::new(size)
    }
}

