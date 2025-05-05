use crate::models::modern_bert::ModernBertModel;
use crate::utils::load_device;
use anyhow::{Error as E, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use hf_hub::{api::sync::Api, Repo, RepoType};
use std::path::PathBuf;
use tokenizers::{PaddingParams, Tokenizer};

#[derive(Debug, Clone)]
pub enum ModernBertSize {
    Base,
    Large,
}

/// Builder for configuring and constructing a fill-mask pipeline using ModernBERT.
///
/// Start by creating a builder with `new(ModernBertSize)`, then chain optional settings:
/// - `.model_id(String)`: Override the Hugging Face model repository ID.
/// - `.revision(String)`: Specify a git revision (branch, tag, commit hash).
/// - `.tokenizer_file(PathBuf)`: Provide a local path to `tokenizer.json`.
/// - `.config_file(PathBuf)`: Provide a local path to `config.json`.
/// - `.weight_files(PathBuf)`: Provide a local path to model weights (`.safetensors` or `.bin`).
/// - `.cpu()`: Force execution on CPU even if CUDA is available.
///
/// Finally, call `.build()` to obtain a `FillMaskPipeline`.
#[derive(Debug)]
pub struct FillMaskPipelineBuilder {
    size: ModernBertSize,
    model_id: Option<String>,
    revision: String,
    tokenizer_file: Option<PathBuf>,
    config_file: Option<PathBuf>,
    weight_files: Option<PathBuf>,
    cpu: bool,
}

impl FillMaskPipelineBuilder {
    /// Creates a new builder for the specified ModernBERT size.
    pub fn new(size: ModernBertSize) -> Self {
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

    /// Constructs the `FillMaskPipeline`.
    ///
    /// This involves downloading model assets (if not provided locally) and
    /// loading the model and tokenizer.
    pub fn build(self) -> Result<FillMaskPipeline> {
        // Calculate model_id for both model and tokenizer use
        let model_id = self.model_id.clone().unwrap_or_else(|| match self.size {
            ModernBertSize::Base => "answerdotai/ModernBERT-base".to_string(),
            ModernBertSize::Large => "answerdotai/ModernBERT-large".to_string(),
        });

        // Fetch tokenizer first
        let api = Api::new()?;
        let repo = api.repo(Repo::with_revision(
            model_id.clone(),
            RepoType::Model,
            self.revision.clone(),
        ));

        let tokenizer_filename = match &self.tokenizer_file {
            Some(file) => file.clone(),
            None => {
                println!("Fetching tokenizer...");
                repo.get("tokenizer.json")?
            }
        };

        // Build the ModernBertModel using the provided configuration
        let model = ModernBertModel::new(
            self.size,
            Some(model_id),
            self.revision,
            self.tokenizer_file,
            self.config_file,
            self.weight_files,
            self.cpu,
        )?;

        println!("Loading tokenizer...");
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

// Note: The original example included helper functions `tokenize_batch` and `get_attention_mask`.
// For this pipeline, which processes one string at a time, the logic is integrated directly
// into the `fill_mask` method for simplicity.
