use crate::models::modernbert::{ZeroShotModernBertModel, ZeroShotModernBertSize};
use tokenizers::Tokenizer;

pub struct ZeroShotClassificationPipeline {
    pub(crate) model: ZeroShotModernBertModel,
    pub(crate) tokenizer: Tokenizer,
}

impl ZeroShotClassificationPipeline {
    pub fn predict(&self, text: &str, candidate_labels: &[&str]) -> anyhow::Result<Vec<(String, f32)>> {
        self.model.predict(&self.tokenizer, text, candidate_labels)
    }

    pub fn get_tokenizer(&self, size: ZeroShotModernBertSize) -> anyhow::Result<Tokenizer> {
        self.model.get_tokenizer(size)
    }
}
