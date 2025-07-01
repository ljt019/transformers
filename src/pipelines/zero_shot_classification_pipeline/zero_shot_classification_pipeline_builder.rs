use super::zero_shot_classification_model::ZeroShotClassificationModel;
use super::zero_shot_classification_pipeline::ZeroShotClassificationPipeline;
use crate::pipelines::utils::model_cache::global_cache;

pub struct ZeroShotClassificationPipelineBuilder<M: ZeroShotClassificationModel> {
    options: M::Options,
}

impl<M: ZeroShotClassificationModel> ZeroShotClassificationPipelineBuilder<M> {
    pub fn new(options: M::Options) -> Self {
        Self { options }
    }

    pub fn build(self) -> anyhow::Result<ZeroShotClassificationPipeline<M>>
    where
        M: Clone + Send + Sync + 'static,
    {
        let key = format!("{:?}", self.options);
        let model = global_cache().get_or_create(&key, || M::new(self.options.clone()))?;
        let tokenizer = M::get_tokenizer(self.options)?;
        Ok(ZeroShotClassificationPipeline { model, tokenizer })
    }
}

impl ZeroShotClassificationPipelineBuilder<crate::models::modernbert::ZeroShotModernBertModel> {
    pub fn modernbert(size: crate::models::modernbert::ModernBertSize) -> Self {
        Self::new(size)
    }
}
