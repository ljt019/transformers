use super::model::EmbeddingModel;
use super::pipeline::EmbeddingPipeline;
use std::sync::Arc;
use crate::core::{global_cache, ModelOptions};
use crate::pipelines::utils::{build_cache_key, DeviceRequest, DeviceSelectable};

pub struct EmbeddingPipelineBuilder<M: EmbeddingModel> {
    options: M::Options,
    device_request: DeviceRequest,
}

impl<M: EmbeddingModel> EmbeddingPipelineBuilder<M> {
    pub fn new(options: M::Options) -> Self {
        Self {
            options,
            device_request: DeviceRequest::Default,
        }
    }

    pub async fn build(self) -> anyhow::Result<EmbeddingPipeline<M>>
    where
        M: Clone + Send + Sync + 'static,
        M::Options: ModelOptions + Clone,
    {
        let device = self.device_request.resolve()?;
        let key = build_cache_key(&self.options, &device);
        let model = global_cache()
            .get_or_create(&key, || M::new(self.options.clone(), device.clone()))
            .await?;
        let tokenizer = M::get_tokenizer(self.options)?;
        Ok(EmbeddingPipeline { model: Arc::new(model), tokenizer })
    }
}

impl<M: EmbeddingModel> DeviceSelectable for EmbeddingPipelineBuilder<M> {
    fn device_request_mut(&mut self) -> &mut DeviceRequest {
        &mut self.device_request
    }
}

impl EmbeddingPipelineBuilder<crate::models::implementations::qwen3_embeddings::Qwen3EmbeddingModel> {
    pub fn qwen3(size: crate::models::implementations::qwen3_embeddings::Qwen3EmbeddingSize) -> Self {
        Self::new(size)
    }
}

