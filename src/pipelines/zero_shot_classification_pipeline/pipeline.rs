use super::model::ZeroShotClassificationModel;
use tokenizers::Tokenizer;

#[derive(Debug, Clone)]
pub struct ClassificationResult {
    pub label: String,
    pub score: f32,
}

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
    ) -> anyhow::Result<Vec<ClassificationResult>> {
        let results = self.model.predict(&self.tokenizer, text, candidate_labels)?;
        Ok(results
            .into_iter()
            .map(|(label, score)| ClassificationResult { label, score })
            .collect())
    }

    /// Predict with raw entailment probabilities for multi-label classification
    pub fn predict_multi_label(
        &self,
        text: &str,
        candidate_labels: &[&str],
    ) -> anyhow::Result<Vec<ClassificationResult>> {
        let results = self.model
            .predict_multi_label(&self.tokenizer, text, candidate_labels)?;
        Ok(results
            .into_iter()
            .map(|(label, score)| ClassificationResult { label, score })
            .collect())
    }

    /// Legacy method for backward compatibility - single-label
    pub fn predict_legacy(
        &self,
        text: &str,
        candidate_labels: &[&str],
    ) -> anyhow::Result<Vec<(String, f32)>> {
        self.model.predict(&self.tokenizer, text, candidate_labels)
    }

    /// Legacy method for backward compatibility - multi-label
    pub fn predict_multi_label_legacy(
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
