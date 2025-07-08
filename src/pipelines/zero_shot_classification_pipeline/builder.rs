use super::model::ZeroShotClassificationModel;
use super::pipeline::ZeroShotClassificationPipeline;
use crate::core::{global_cache, ModelOptions};
use crate::pipelines::utils::{build_cache_key, DeviceRequest, DeviceSelectable, BasePipelineBuilder};

pub struct ZeroShotClassificationPipelineBuilder<M: ZeroShotClassificationModel> {
    options: M::Options,
    device_request: DeviceRequest,
}

impl<M: ZeroShotClassificationModel> ZeroShotClassificationPipelineBuilder<M> {
    pub fn new(options: M::Options) -> Self {
        Self {
            options,
            device_request: DeviceRequest::Default,
        }
    }
}

impl<M: ZeroShotClassificationModel> DeviceSelectable for ZeroShotClassificationPipelineBuilder<M> {
    fn device_request_mut(&mut self) -> &mut DeviceRequest {
        &mut self.device_request
    }
}

impl<M: ZeroShotClassificationModel> BasePipelineBuilder<M> for ZeroShotClassificationPipelineBuilder<M>
where
    M: Clone + Send + Sync + 'static,
    M::Options: ModelOptions + Clone,
{
    type Model = M;
    type Pipeline = ZeroShotClassificationPipeline<M>;
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
        Ok(ZeroShotClassificationPipeline { model, tokenizer })
    }
}

impl
    ZeroShotClassificationPipelineBuilder<
        crate::models::implementations::modernbert::ZeroShotModernBertModel,
    >
{
    pub fn modernbert(size: crate::models::ModernBertSize) -> Self {
        Self::new(size)
    }
}

