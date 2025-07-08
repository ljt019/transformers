use super::model::ZeroShotClassificationModel;
use tokenizers::Tokenizer;

pub struct ZeroShotClassificationPipeline<M: ZeroShotClassificationModel> {
    pub(crate) model: M,
    pub(crate) tokenizer: Tokenizer,
}

impl<M: ZeroShotClassificationModel> ZeroShotClassificationPipeline<M> {
    /// Predict with normalized probabilities for single-label classification (probabilities sum to 1)
    pub fn predict(
        &self,
        text: &str,
        candidate_labels: &[&str],
    ) -> anyhow::Result<Vec<(String, f32)>> {
        self.model.predict(&self.tokenizer, text, candidate_labels)
    }

    /// Predict with raw entailment probabilities for multi-label classification
    pub fn predict_multi_label(
        &self,
        text: &str,
        candidate_labels: &[&str],
    ) -> anyhow::Result<Vec<(String, f32)>> {
        self.model
            .predict_multi_label(&self.tokenizer, text, candidate_labels)
    }

    pub fn device(&self) -> &candle_core::Device {
        self.model.device()
    }
}
