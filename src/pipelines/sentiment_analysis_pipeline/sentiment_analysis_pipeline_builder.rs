use super::sentiment_analysis_model::SentimentAnalysisModel;
use super::sentiment_analysis_pipeline::SentimentAnalysisPipeline;
use crate::pipelines::utils::model_cache::global_cache;

pub struct SentimentAnalysisPipelineBuilder<M: SentimentAnalysisModel> {
    options: M::Options,
}

impl<M: SentimentAnalysisModel> SentimentAnalysisPipelineBuilder<M> {
    pub fn new(options: M::Options) -> Self {
        Self { options }
    }

    pub fn build(self) -> anyhow::Result<SentimentAnalysisPipeline<M>>
    where
        M: Clone + Send + Sync + 'static,
    {
        let key = format!("{:?}", self.options);
        let model = global_cache().get_or_create(&key, || M::new(self.options.clone()))?;
        let tokenizer = M::get_tokenizer(self.options)?;
        Ok(SentimentAnalysisPipeline { model, tokenizer })
    }
}

impl SentimentAnalysisPipelineBuilder<crate::models::modernbert::SentimentModernBertModel> {
    pub fn modernbert(size: crate::models::modernbert::ModernBertSize) -> Self {
        Self::new(size)
    }
}
