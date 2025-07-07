use super::embedding_model::EmbeddingModel;
use tokenizers::Tokenizer;

pub struct EmbeddingPipeline<M: EmbeddingModel> {
    pub(crate) model: M,
    pub(crate) tokenizer: Tokenizer,
}

impl<M: EmbeddingModel> EmbeddingPipeline<M> {
    pub fn embed(&self, text: &str) -> anyhow::Result<Vec<f32>> {
        self.model.embed(&self.tokenizer, text)
    }

    pub fn device(&self) -> &candle_core::Device {
        self.model.device()
    }
}
