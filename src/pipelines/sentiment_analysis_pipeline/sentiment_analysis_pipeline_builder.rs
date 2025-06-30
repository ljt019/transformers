use super::sentiment_analysis_pipeline::SentimentAnalysisPipeline;
use crate::models::modernbert::{SentimentModernBertModel, SentimentModernBertSize};
use crate::pipelines::utils::model_cache::global_cache;

pub struct SentimentAnalysisPipelineBuilder {
    size: SentimentModernBertSize,
}

impl SentimentAnalysisPipelineBuilder {
    pub fn new(size: SentimentModernBertSize) -> Self {
        Self { size }
    }

    pub fn build(self) -> anyhow::Result<SentimentAnalysisPipeline> {
        let key = format!("{:?}", self.size);
        let model = global_cache().get_or_create(&key, || SentimentModernBertModel::new(self.size))?;
        let tokenizer = model.get_tokenizer(self.size)?;
        Ok(SentimentAnalysisPipeline { model, tokenizer })
    }
}
