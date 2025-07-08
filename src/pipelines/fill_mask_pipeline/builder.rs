use super::model::FillMaskModel;
use super::pipeline::FillMaskPipeline;
use crate::core::{global_cache, ModelOptions};
use crate::pipelines::utils::{build_cache_key, DeviceRequest, DeviceSelectable, BasePipelineBuilder};

pub struct FillMaskPipelineBuilder<M: FillMaskModel> {
    options: M::Options,
    device_request: DeviceRequest,
}

impl<M: FillMaskModel> FillMaskPipelineBuilder<M> {
    pub fn new(options: M::Options) -> Self {
        Self {
            options,
            device_request: DeviceRequest::Default,
        }
    }
}

impl<M: FillMaskModel> DeviceSelectable for FillMaskPipelineBuilder<M> {
    fn device_request_mut(&mut self) -> &mut DeviceRequest {
        &mut self.device_request
    }
}

impl<M: FillMaskModel> BasePipelineBuilder<M> for FillMaskPipelineBuilder<M>
where
    M: Clone + Send + Sync + 'static,
    M::Options: ModelOptions + Clone,
{
    type Model = M;
    type Pipeline = FillMaskPipeline<M>;
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
        Ok(FillMaskPipeline { model, tokenizer })
    }
}

impl FillMaskPipelineBuilder<crate::models::implementations::modernbert::FillMaskModernBertModel> {
    pub fn modernbert(size: crate::models::ModernBertSize) -> Self {
        Self::new(size)
    }
}

