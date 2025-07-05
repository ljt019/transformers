use super::fill_mask_model::FillMaskModel;
use tokenizers::Tokenizer;

pub struct FillMaskPipeline<M: FillMaskModel> {
    pub(crate) model: M,
    pub(crate) tokenizer: Tokenizer,
}

impl<M: FillMaskModel> FillMaskPipeline<M> {
    pub fn fill_mask(&self, text: &str) -> anyhow::Result<String> {
        self.model.predict(&self.tokenizer, text)
    }

    pub fn device(&self) -> &candle_core::Device {
        self.model.device()
    }
}
