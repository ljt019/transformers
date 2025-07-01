use super::fill_mask_pipeline::FillMaskPipeline;
use crate::models::modernbert::{FillMaskModernBertModel, ModernBertSize};
use crate::pipelines::utils::model_cache::global_cache;

pub struct FillMaskPipelineBuilder {
    size: ModernBertSize,
}

impl FillMaskPipelineBuilder {
    pub fn new(size: ModernBertSize) -> Self {
        Self { size }
    }

    pub fn build(self) -> anyhow::Result<FillMaskPipeline> {
        let key = format!("{:?}", self.size);
        let model =
            global_cache().get_or_create(&key, || FillMaskModernBertModel::new(self.size))?;
        let tokenizer = FillMaskModernBertModel::get_tokenizer(self.size)?;
        Ok(FillMaskPipeline { model, tokenizer })
    }
}
