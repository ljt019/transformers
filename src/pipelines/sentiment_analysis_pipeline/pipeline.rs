use super::model::SentimentAnalysisModel;
use tokenizers::Tokenizer;

#[derive(Debug, Clone)]
pub struct SentimentResult {
    pub label: String,
    pub score: f32,
}

pub struct SentimentAnalysisPipeline<M: SentimentAnalysisModel> {
    pub(crate) model: M,
    pub(crate) tokenizer: Tokenizer,
}

impl<M: SentimentAnalysisModel> SentimentAnalysisPipeline<M> {
    /// Predict sentiment with structured result containing label and confidence score
    pub fn predict(&self, text: &str) -> anyhow::Result<SentimentResult> {
        // For now, return mock structured data - this needs to be implemented based on the actual model
        // The model.predict method needs to be updated to return structured data
        let result = self.model.predict(&self.tokenizer, text)?;
        
        // TODO: Parse the string result and extract label and score
        // This is a temporary implementation
        let label = if result.to_lowercase().contains("positive") {
            "POSITIVE".to_string()
        } else if result.to_lowercase().contains("negative") {
            "NEGATIVE".to_string()
        } else {
            "NEUTRAL".to_string()
        };
        
        Ok(SentimentResult {
            label,
            score: 0.95, // Mock score
        })
    }

    /// Legacy method for backward compatibility
    pub fn predict_legacy(&self, text: &str) -> anyhow::Result<String> {
        self.model.predict(&self.tokenizer, text)
    }

    pub fn device(&self) -> &candle_core::Device {
        self.model.device()
    }
}
