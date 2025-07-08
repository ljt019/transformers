use super::model::RerankModel;
use super::pipeline::RerankPipeline;
use std::sync::Arc;
use crate::core::{global_cache, ModelOptions};
use crate::pipelines::utils::{build_cache_key, DeviceRequest, DeviceSelectable};

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

    pub async fn build(self) -> anyhow::Result<RerankPipeline<M>>
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
        Ok(RerankPipeline { model: Arc::new(model), tokenizer })
    }
}

impl<M: RerankModel> DeviceSelectable for RerankPipelineBuilder<M> {
    fn device_request_mut(&mut self) -> &mut DeviceRequest {
        &mut self.device_request
    }
}

impl RerankPipelineBuilder<crate::models::implementations::qwen3_reranker::Qwen3RerankModel> {
    pub fn qwen3(size: crate::models::implementations::qwen3_reranker::Qwen3RerankSize) -> Self {
        Self::new(size)
    }
}
