use super::model::SentimentAnalysisModel;
use super::pipeline::SentimentAnalysisPipeline;
use crate::core::{global_cache, ModelOptions};
use crate::pipelines::utils::{build_cache_key, DeviceRequest, DeviceSelectable, BasePipelineBuilder};

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
}

impl<M: SentimentAnalysisModel> DeviceSelectable for SentimentAnalysisPipelineBuilder<M> {
    fn device_request_mut(&mut self) -> &mut DeviceRequest {
        &mut self.device_request
    }
}

impl<M: SentimentAnalysisModel> BasePipelineBuilder<M> for SentimentAnalysisPipelineBuilder<M>
where
    M: Clone + Send + Sync + 'static,
    M::Options: ModelOptions + Clone,
{
    type Model = M;
    type Pipeline = SentimentAnalysisPipeline<M>;
    type Options = M::Options;

    fn options(&self) -> &Self::Options {
        &self.options
    }
    
    fn device_request(&self) -> &DeviceRequest {
        &self.device_request
    }

    fn create_model(options: Self::Options, device: candle_core::Device) -> anyhow::Result<M> {
        M::new(options, device)
    }
    
    fn get_tokenizer(options: Self::Options) -> anyhow::Result<tokenizers::Tokenizer> {
        M::get_tokenizer(options)
    }
    
    fn construct_pipeline(model: M, tokenizer: tokenizers::Tokenizer) -> anyhow::Result<Self::Pipeline> {
        Ok(SentimentAnalysisPipeline { model, tokenizer })
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

