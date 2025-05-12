pub mod fill_mask_pipeline;
pub mod sentiment_analysis_pipeline;
pub mod text_generation_pipeline;
pub mod zero_shot_classification_pipeline;

pub trait TextGenerationModel {
    fn load_tokenizer(&self) -> anyhow::Result<tokenizers::Tokenizer>;

    fn get_eos_token_str(&self) -> &str;

    fn format_prompt(&self, prompt: &str) -> String;

    fn prompt_with_tokens(
        &self,
        tokens: &[u32],
        max_length: usize,
        eos_token: u32,
    ) -> anyhow::Result<Vec<u32>>;
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
