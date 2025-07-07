use super::embedding_model::EmbeddingModel;
use tokenizers::Tokenizer;

pub struct EmbeddingPipeline<M: EmbeddingModel> {
    pub(crate) model: M,
    pub(crate) tokenizer: Tokenizer,
}

impl<M: EmbeddingModel> EmbeddingPipeline<M> {
    pub fn embed(&self, text: &str) -> anyhow::Result<Vec<f32>> {
        let mut out = self.embed_batch(&[text])?;
        Ok(out.pop().unwrap())
    }

    pub fn embed_batch(&self, texts: &[&str]) -> anyhow::Result<Vec<Vec<f32>>> {
        self.model.embed_batch(&self.tokenizer, texts)
    }

    pub fn device(&self) -> &candle_core::Device {
        self.model.device()
    }
}
