use crate::models::modernbert::FillMaskModernBertModel;
use tokenizers::Tokenizer;

pub struct FillMaskPipeline {
    pub(crate) model: FillMaskModernBertModel,
    pub(crate) tokenizer: Tokenizer,
}

impl FillMaskPipeline {
    pub fn fill_mask(&self, text: &str) -> anyhow::Result<String> {
        self.model.predict(&self.tokenizer, text)
    }
}
