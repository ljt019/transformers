use super::model::FillMaskModel;
use super::pipeline::FillMaskPipeline;
use crate::core::ModelOptions;
use crate::pipelines::utils::{BasePipelineBuilder, StandardPipelineBuilder};

/// Builder for fill-mask pipelines
pub type FillMaskPipelineBuilder<M> = StandardPipelineBuilder<<M as FillMaskModel>::Options>;

impl<M> BasePipelineBuilder<M> for StandardPipelineBuilder<M::Options>
where
    M: FillMaskModel + Clone + Send + Sync + 'static,
    M::Options: ModelOptions + Clone,
{
    type Model = M;
    type Pipeline = FillMaskPipeline<M>;
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
        Ok(FillMaskPipeline { model, tokenizer })
    }
}

impl StandardPipelineBuilder<crate::models::ModernBertSize> {
    pub fn modernbert(size: crate::models::ModernBertSize) -> Self {
        Self::new(size)
    }
}

