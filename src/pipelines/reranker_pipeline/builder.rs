use super::reranker_model::RerankModel;
use super::reranker_pipeline::RerankPipeline;
use std::sync::Arc;
use crate::core::{global_cache, ModelOptions};
use crate::pipelines::utils::DeviceRequest;

pub struct RerankPipelineBuilder<M: RerankModel> {
    options: M::Options,
    device_request: DeviceRequest,
}

impl<M: RerankModel> RerankPipelineBuilder<M> {
    pub fn new(options: M::Options) -> Self {
        Self {
            options,
            device_request: DeviceRequest::Default,
        }
    }

    pub fn cpu(mut self) -> Self {
        self.device_request = DeviceRequest::Cpu;
        self
    }

    pub fn cuda_device(mut self, index: usize) -> Self {
        self.device_request = DeviceRequest::Cuda(index);
        self
    }

    pub fn device(mut self, device: candle_core::Device) -> Self {
        self.device_request = DeviceRequest::Explicit(device);
        self
    }

    pub async fn build(self) -> anyhow::Result<RerankPipeline<M>>
    where
        M: Clone + Send + Sync + 'static,
        M::Options: ModelOptions + Clone,
    {
        let device = self.device_request.resolve()?;
        let key = format!("{}-{:?}", self.options.cache_key(), device.location());
        let model = global_cache()
            .get_or_create(&key, || M::new(self.options.clone(), device.clone()))
            .await?;
        let tokenizer = M::get_tokenizer(self.options)?;
        Ok(RerankPipeline { model: Arc::new(model), tokenizer })
    }
}

impl RerankPipelineBuilder<crate::models::implementations::qwen3_reranker::Qwen3RerankModel> {
    pub fn qwen3(size: crate::models::implementations::qwen3_reranker::Qwen3RerankSize) -> Self {
        Self::new(size)
    }
}