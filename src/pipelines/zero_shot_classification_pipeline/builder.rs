use super::model::ZeroShotClassificationModel;
use super::pipeline::ZeroShotClassificationPipeline;
use crate::core::ModelOptions;
use crate::pipelines::utils::{BasePipelineBuilder, StandardPipelineBuilder};

/// Builder for zero-shot classification pipelines
pub type ZeroShotClassificationPipelineBuilder<M> = StandardPipelineBuilder<<M as ZeroShotClassificationModel>::Options>;

impl<M> BasePipelineBuilder<M> for StandardPipelineBuilder<M::Options>
where
    M: ZeroShotClassificationModel + Clone + Send + Sync + 'static,
    M::Options: ModelOptions + Clone,
{
    type Model = M;
    type Pipeline = ZeroShotClassificationPipeline<M>;
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
        Ok(ZeroShotClassificationPipeline { model, tokenizer })
    }
}

impl StandardPipelineBuilder<crate::models::ModernBertSize> {
    pub fn modernbert(size: crate::models::ModernBertSize) -> Self {
        Self::new(size)
    }
}

