use super::model::FillMaskModel;
use tokenizers::Tokenizer;

#[derive(Debug, Clone)]
pub struct FillMaskPrediction {
    pub word: String,
    pub score: f32,
}

pub struct FillMaskPipeline<M: FillMaskModel> {
    pub(crate) model: M,
    pub(crate) tokenizer: Tokenizer,
}

impl<M: FillMaskModel> FillMaskPipeline<M> {
    /// Return the top prediction for the masked token
    pub fn fill_mask(&self, text: &str) -> anyhow::Result<FillMaskPrediction> {
        let predictions = self.fill_mask_top_k(text, 1)?;
        predictions.into_iter().next()
            .ok_or_else(|| anyhow::anyhow!("No predictions returned"))
    }

    /// Return top-k predictions with scores for ranking/choice
    pub fn fill_mask_top_k(&self, text: &str, _k: usize) -> anyhow::Result<Vec<FillMaskPrediction>> {
        // For now, return mock data - this needs to be implemented based on the actual model
        // The model.predict method needs to be updated to return structured data
        let result = self.model.predict(&self.tokenizer, text)?;
        
        // TODO: Parse the string result and extract multiple predictions with scores
        // This is a temporary implementation
        Ok(vec![FillMaskPrediction {
            word: result.trim().to_string(),
            score: 0.95, // Mock score
        }])
    }

    /// Legacy method for backward compatibility
    pub fn fill_mask_legacy(&self, text: &str) -> anyhow::Result<String> {
        self.model.predict(&self.tokenizer, text)
    }

    pub fn device(&self) -> &candle_core::Device {
        self.model.device()
    }
}
