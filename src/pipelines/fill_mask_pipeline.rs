use crate::models::modern_bert::ModernBertModel;
use anyhow::Result;
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::{PaddingParams, Tokenizer};

#[derive(Debug, Clone)]
pub enum ModernBertSize {
    Base,
    Large,
}

/// Builder for configuring and constructing a fill-mask pipeline using ModernBERT.
///
/// Finally, call `.build()` to obtain a `FillMaskPipeline`.
#[derive(Debug)]
pub struct FillMaskPipelineBuilder {
    size: ModernBertSize,
}

impl FillMaskPipelineBuilder {
    /// Creates a new builder for the specified ModernBERT size.
    pub fn new(size: ModernBertSize) -> Self {
        Self { size }
    }

    /// Constructs the `FillMaskPipeline`.
    ///
    /// This involves downloading model assets (if not provided locally) and
    /// loading the model and tokenizer.
    pub fn build(self) -> Result<FillMaskPipeline> {
        // Calculate model_id for both model and tokenizer use
        let model_id = match self.size {
            ModernBertSize::Base => "answerdotai/ModernBERT-base".to_string(),
            ModernBertSize::Large => "answerdotai/ModernBERT-large".to_string(),
        };

        // Fetch tokenizer first
        let api = Api::new()?;
        let repo = api.repo(Repo::new(model_id.clone(), RepoType::Model));

        let tokenizer_filename = repo.get("tokenizer.json")?;

        // Build the ModernBertModel using the provided configuration
        let model = ModernBertModel::new(self.size)?;

        let mut tokenizer = Tokenizer::from_file(&tokenizer_filename).map_err(|e| {
            anyhow::Error::msg(format!(
                "Failed to load tokenizer from {:?}: {}",
                tokenizer_filename, e
            ))
        })?;

        // Configure padding
        let pad_token_id = model.get_pad_token_id();
        let pp = PaddingParams {
            strategy: tokenizers::PaddingStrategy::BatchLongest,
            pad_id: pad_token_id,
            pad_token: tokenizer
                .id_to_token(pad_token_id)
                .unwrap_or_else(|| "[PAD]".to_string()),
            ..Default::default()
        };
        tokenizer.with_padding(Some(pp));

        Ok(FillMaskPipeline { model, tokenizer })
    }
}

/// A ready-to-use pipeline for filling masked tokens using ModernBERT.
///
/// After building with `FillMaskPipelineBuilder`, call
/// `fill_mask(text)` to predict the masked token.
///
/// Example:
/// ```rust,no_run
/// use transformers::pipelines::fill_mask_pipeline::{FillMaskPipelineBuilder, ModernBertSize};
///
/// # fn run() -> anyhow::Result<()> {
/// let pipeline = FillMaskPipelineBuilder::new(ModernBertSize::Base)
///    .build()?;
///
/// let output = pipeline.fill_mask("The capital of France is [MASK].")?;
/// // output might be "The capital of France is Paris."
/// println!("{}", output);
/// # Ok(())
/// # }
/// ```
pub struct FillMaskPipeline {
    model: ModernBertModel,
    tokenizer: Tokenizer,
}

impl FillMaskPipeline {
    /// Fills the `[MASK]` token in a given text string.
    ///
    /// Expects the input `text` to contain exactly one instance of `"[MASK]"`.
    /// Returns the text with the mask replaced by the predicted token.
    pub fn fill_mask(&self, text: &str) -> Result<String> {
        self.model.fill_mask(&self.tokenizer, text)
    }
}
