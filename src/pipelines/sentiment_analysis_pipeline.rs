use crate::models::sentiment_modern_bert::SentimentModernBertModel;
// Re-export the size enum for users of the pipeline
pub use crate::models::sentiment_modern_bert::SentimentModernBertSize;
use crate::utils::{load_tokenizer, HfConfig};
use anyhow::Result;
use tokenizers::{PaddingParams, Tokenizer};

/// Builder for configuring and constructing a sentiment analysis pipeline using ModernBERT.
///
/// Finally, call `.build()` to obtain a `SentimentAnalysisPipeline`.
#[derive(Debug)]
pub struct SentimentAnalysisPipelineBuilder {
    size: SentimentModernBertSize,
}

impl SentimentAnalysisPipelineBuilder {
    /// Creates a new builder for the specified ModernBERT sentiment model size.
    pub fn new(size: SentimentModernBertSize) -> Self {
        Self { size }
    }

    /// Constructs the `SentimentAnalysisPipeline`.
    ///
    /// This involves downloading model/tokenizer assets (if not provided locally) and
    /// loading the model and tokenizer.
    pub fn build(self) -> Result<SentimentAnalysisPipeline> {
        // Build the SentimentModernBertModel
        let model = SentimentModernBertModel::new(self.size)?;

        // Get tokenizer info (model_id might differ from the one used for model weights)
        let tokenizer_model_id = SentimentModernBertModel::get_tokenizer_repo_info(self.size);

        // Create HfConfig for the tokenizer
        let tokenizer_config = HfConfig::new(
            &tokenizer_model_id, // model repo id (borrowed)
            "tokenizer.json",    // tokenizer filename on hub
            "",                  // gguf repo (not needed for tokenizer)
            "",                  // gguf filename (not needed for tokenizer)
        );

        // Load tokenizer using HfConfig
        let mut tokenizer = load_tokenizer(&tokenizer_config)?;

        // Configure padding (get pad_token_id from tokenizer config or a default)
        // Assuming the tokenizer for sentiment model has standard BERT padding tokens
        let pad_token_id = tokenizer.get_padding().map_or(0, |p| p.pad_id); // Default to 0 if no padding
        let pp = PaddingParams {
            strategy: tokenizers::PaddingStrategy::BatchLongest,
            pad_id: pad_token_id,
            pad_token: tokenizer
                .id_to_token(pad_token_id)
                .unwrap_or_else(|| "[PAD]".to_string()),
            ..Default::default()
        };
        tokenizer.with_padding(Some(pp));

        Ok(SentimentAnalysisPipeline { model, tokenizer })
    }
}

/// A ready-to-use pipeline for sentiment analysis using ModernBERT.
///
/// After building with `SentimentAnalysisPipelineBuilder`, call
/// `predict(text)` to get the sentiment label.
///
/// Example:
/// ```rust,no_run
/// use transformers::pipelines::sentiment_analysis_pipeline::{SentimentAnalysisPipelineBuilder, SentimentModernBertSize};
///
/// # fn run() -> anyhow::Result<()> {
/// let pipeline = SentimentAnalysisPipelineBuilder::new(SentimentModernBertSize::Base)
///    .build()?;
///
/// let sentiment = pipeline.predict("This movie was fantastic!")?;
/// // sentiment might be "positive"
/// println!("Sentiment: {}", sentiment);
/// # Ok(())
/// # }
/// ```
pub struct SentimentAnalysisPipeline {
    model: SentimentModernBertModel,
    tokenizer: Tokenizer,
}

impl SentimentAnalysisPipeline {
    /// Predicts the sentiment (e.g., "positive", "negative", "neutral") of the input text.
    pub fn predict(&self, text: &str) -> Result<String> {
        self.model.predict(&self.tokenizer, text)
    }
}
