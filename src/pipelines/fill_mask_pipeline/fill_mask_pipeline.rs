use crate::models::modernbert::{FillMaskModernBertModel, ModernBertSize};
use tokenizers::Tokenizer;

pub struct FillMaskPipeline {
    pub(crate) model: FillMaskModernBertModel,
    pub(crate) tokenizer: Tokenizer,
}

impl FillMaskPipeline {
    pub fn fill_mask(&self, text: &str) -> anyhow::Result<String> {
        self.model.predict(&self.tokenizer, text)
    }

    pub fn get_tokenizer(&self, size: ModernBertSize) -> anyhow::Result<Tokenizer> {
        self.model.get_tokenizer(size)
    }
}
