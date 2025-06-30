use super::zero_shot_classification_pipeline::ZeroShotClassificationPipeline;
use crate::models::modernbert::{ZeroShotModernBertModel, ZeroShotModernBertSize};
use crate::pipelines::utils::model_cache::global_cache;

pub struct ZeroShotClassificationPipelineBuilder {
    size: ZeroShotModernBertSize,
}

impl ZeroShotClassificationPipelineBuilder {
    pub fn new(size: ZeroShotModernBertSize) -> Self {
        Self { size }
    }

    pub fn build(self) -> anyhow::Result<ZeroShotClassificationPipeline> {
        let key = format!("{:?}", self.size);
        let model = global_cache().get_or_create(&key, || ZeroShotModernBertModel::new(self.size))?;
        let tokenizer = model.get_tokenizer(self.size)?;
        Ok(ZeroShotClassificationPipeline { model, tokenizer })
    }
}
