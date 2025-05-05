use crate::models::sentiment_modern_bert::SentimentModernBertModel;
// Re-export the size enum for users of the pipeline
pub use crate::models::sentiment_modern_bert::SentimentModernBertSize;
use crate::utils::{load_tokenizer, HfConfig};
use anyhow::Result;
use std::path::PathBuf;
use tokenizers::{PaddingParams, Tokenizer};

/// Builder for configuring and constructing a sentiment analysis pipeline using ModernBERT.
///
/// Start by creating a builder with `new(SentimentModernBertSize)`, then chain optional settings:
/// - `.model_id(String)`: Override the Hugging Face model repository ID.
/// - `.revision(String)`: Specify a git revision (branch, tag, commit hash).
/// - `.tokenizer_file(PathBuf)`: Provide a local path to `tokenizer.json`.
/// - `.config_file(PathBuf)`: Provide a local path to `config.json`.
/// - `.weight_files(PathBuf)`: Provide a local path to model weights (`.safetensors` or `.bin`).
/// - `.cpu()`: Force execution on CPU even if CUDA is available.
///
/// Finally, call `.build()` to obtain a `SentimentAnalysisPipeline`.
#[derive(Debug)]
pub struct SentimentAnalysisPipelineBuilder {
    size: SentimentModernBertSize,
    model_id: Option<String>,
    revision: String,
    tokenizer_file: Option<PathBuf>,
    config_file: Option<PathBuf>,
    weight_files: Option<PathBuf>,
    cpu: bool,
}

impl SentimentAnalysisPipelineBuilder {
    /// Creates a new builder for the specified ModernBERT sentiment model size.
    pub fn new(size: SentimentModernBertSize) -> Self {
        Self {
            size,
            model_id: None,
            revision: "main".to_string(),
            tokenizer_file: None,
            config_file: None,
            weight_files: None,
            cpu: false,
        }
    }

    /// Overrides the default Hugging Face model repository ID.
    pub fn model_id(mut self, id: impl Into<String>) -> Self {
        self.model_id = Some(id.into());
        self
    }

    /// Sets the git revision (branch, tag, commit hash) to use.
    pub fn revision(mut self, rev: impl Into<String>) -> Self {
        self.revision = rev.into();
        self
    }

    /// Sets a local path for the tokenizer configuration file.
    pub fn tokenizer_file(mut self, p: impl Into<PathBuf>) -> Self {
        self.tokenizer_file = Some(p.into());
        self
    }

    /// Sets a local path for the model configuration file.
    pub fn config_file(mut self, p: impl Into<PathBuf>) -> Self {
        self.config_file = Some(p.into());
        self
    }

    /// Sets a local path for the model weight file(s).
    pub fn weight_files(mut self, p: impl Into<PathBuf>) -> Self {
        self.weight_files = Some(p.into());
        self
    }

    /// Forces the pipeline to run on the CPU.
    pub fn cpu(mut self) -> Self {
        self.cpu = true;
        self
    }

    /// Constructs the `SentimentAnalysisPipeline`.
    ///
    /// This involves downloading model/tokenizer assets (if not provided locally) and
    /// loading the model and tokenizer.
    pub fn build(self) -> Result<SentimentAnalysisPipeline> {
        // Build the SentimentModernBertModel
        let model = SentimentModernBertModel::new(
            self.size,
            self.model_id.clone(), // Pass clone for potential later use
            self.revision.clone(),
            self.tokenizer_file.clone(),
            self.config_file,
            self.weight_files,
            self.cpu,
        )?;

        // Get tokenizer info (model_id might differ from the one used for model weights)
        let (tokenizer_model_id, tokenizer_revision) =
            SentimentModernBertModel::get_tokenizer_repo_info(
                self.size,
                self.model_id, // Pass the original override if present
                self.revision,
            );

        // Create HfConfig for the tokenizer
        let tokenizer_config = HfConfig::new(
            &tokenizer_model_id, // model repo id (borrowed)
            "tokenizer.json",    // tokenizer filename on hub
            "",                  // gguf repo (not needed for tokenizer)
            "",                  // gguf filename (not needed for tokenizer)
        );

        // Load tokenizer using HfConfig
        let mut tokenizer = load_tokenizer(&tokenizer_config)?;

        // Handle local tokenizer file override if provided
        if let Some(local_tokenizer_path) = self.tokenizer_file {
            println!(
                "Overriding tokenizer with local file: {:?}",
                local_tokenizer_path
            );
            tokenizer = Tokenizer::from_file(local_tokenizer_path)
                .map_err(|e| anyhow::anyhow!("Failed to load local tokenizer file: {}", e))?;
        }

        // Configure padding (get pad_token_id from tokenizer config or a default)
        // Assuming the tokenizer for sentiment model has standard BERT padding tokens
        // TODO: Potentially get pad_token_id from the model's config if necessary
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
