use super::sentiment_analysis_model::SentimentAnalysisModel;
use tokenizers::Tokenizer;

pub struct SentimentAnalysisPipeline<M: SentimentAnalysisModel> {
    pub(crate) model: M,
    pub(crate) tokenizer: Tokenizer,
}

impl<M: SentimentAnalysisModel> SentimentAnalysisPipeline<M> {
    pub fn predict(&self, text: &str) -> anyhow::Result<String> {
        self.model.predict(&self.tokenizer, text)
    }

    pub fn device(&self) -> &candle_core::Device {
        self.model.device()
    }
}
