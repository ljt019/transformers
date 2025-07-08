use super::model::SentimentAnalysisModel;
use super::pipeline::SentimentAnalysisPipeline;
use crate::core::{global_cache, ModelOptions};
use crate::pipelines::utils::{build_cache_key, DeviceRequest, DeviceSelectable};

pub struct SentimentAnalysisPipelineBuilder<M: SentimentAnalysisModel> {
    options: M::Options,
    device_request: DeviceRequest,
}

impl<M: SentimentAnalysisModel> SentimentAnalysisPipelineBuilder<M> {
    pub fn new(options: M::Options) -> Self {
        Self {
            options,
            device_request: DeviceRequest::Default,
        }
    }

    pub async fn build(self) -> anyhow::Result<SentimentAnalysisPipeline<M>>
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
        Ok(SentimentAnalysisPipeline { model, tokenizer })
    }
}

impl<M: SentimentAnalysisModel> DeviceSelectable for SentimentAnalysisPipelineBuilder<M> {
    fn device_request_mut(&mut self) -> &mut DeviceRequest {
        &mut self.device_request
    }
}

impl
    SentimentAnalysisPipelineBuilder<
        crate::models::implementations::modernbert::SentimentModernBertModel,
    >
{
    pub fn modernbert(size: crate::models::ModernBertSize) -> Self {
        Self::new(size)
    }
}

