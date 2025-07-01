use super::zero_shot_classification_model::ZeroShotClassificationModel;
use tokenizers::Tokenizer;

pub struct ZeroShotClassificationPipeline<M: ZeroShotClassificationModel> {
    pub(crate) model: M,
    pub(crate) tokenizer: Tokenizer,
}

impl<M: ZeroShotClassificationModel> ZeroShotClassificationPipeline<M> {
    pub fn predict(&self, text: &str, candidate_labels: &[&str]) -> anyhow::Result<Vec<(String, f32)>> {
        self.model.predict(&self.tokenizer, text, candidate_labels)
    }
}
