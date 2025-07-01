use crate::models::modernbert::{SentimentModernBertModel, SentimentModernBertSize};
use tokenizers::Tokenizer;

pub struct SentimentAnalysisPipeline {
    pub(crate) model: SentimentModernBertModel,
    pub(crate) tokenizer: Tokenizer,
}

impl SentimentAnalysisPipeline {
    pub fn predict(&self, text: &str) -> anyhow::Result<String> {
        self.model.predict(&self.tokenizer, text)
    }
}
