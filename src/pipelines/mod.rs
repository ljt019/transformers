pub mod fill_mask_pipeline;
pub mod sentiment_analysis_pipeline;
pub mod text_generation_pipeline;
pub mod zero_shot_classification_pipeline;

use crate::utils::loaders::LoadedGgufModelWeights;

pub trait TextGenerationModel {
    fn load_tokenizer(&self) -> anyhow::Result<tokenizers::Tokenizer>;

    fn load_model_weights(&self) -> anyhow::Result<LoadedGgufModelWeights>;

    fn prompt(&self, prompt: &str, max_length: usize) -> anyhow::Result<String>;
}

pub trait FillMaskModel {
    fn fill_mask(prompt: &str) -> anyhow::Result<String>;
}

pub trait SentimentAnalysisModel {
    fn predict(text: &str) -> anyhow::Result<String>;
}

pub trait ZeroShotClassificationModel {
    fn predict(
        &self,
        premise: &str,
        candidate_labels: &[&str],
    ) -> anyhow::Result<Vec<(String, f32)>>;
}
