use super::fill_mask_model::FillMaskModel;
use super::fill_mask_pipeline::FillMaskPipeline;
use crate::pipelines::utils::model_cache::global_cache;

pub struct FillMaskPipelineBuilder<M: FillMaskModel> {
    options: M::Options,
}

impl<M: FillMaskModel> FillMaskPipelineBuilder<M> {
    pub fn new(options: M::Options) -> Self {
        Self { options }
    }

    pub fn build(self) -> anyhow::Result<FillMaskPipeline<M>>
    where
        M: Clone + Send + Sync + 'static,
    {
        let key = format!("{:?}", self.options);
        let model = global_cache().get_or_create(&key, || M::new(self.options.clone()))?;
        let tokenizer = M::get_tokenizer(self.options)?;
        Ok(FillMaskPipeline { model, tokenizer })
    }
}

impl FillMaskPipelineBuilder<crate::models::modernbert::FillMaskModernBertModel> {
    pub fn modernbert(size: crate::models::modernbert::ModernBertSize) -> Self {
        Self::new(size)
    }
}
