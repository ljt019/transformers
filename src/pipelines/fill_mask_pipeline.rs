use crate::utils::load_device;
use anyhow::{Error as E, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::modernbert;
use candle_transformers::models::modernbert::ModernBertForMaskedLM;
use hf_hub::{api::sync::Api, Repo, RepoType};
use std::path::PathBuf;
use tokenizers::{PaddingParams, Tokenizer};

/// Available ModernBERT model sizes.
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
    device: Option<Device>, // Added to store device after loading
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
            device: None, // Initialize device as None
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
    pub fn build(mut self) -> Result<FillMaskPipeline> {
        let device = if self.cpu {
            Device::Cpu
        } else {
            load_device()?
        };
        self.device = Some(device.clone()); // Store the loaded device

        let api = Api::new()?;
        let model_id = self.model_id.clone().unwrap_or_else(|| match self.size {
            ModernBertSize::Base => "answerdotai/ModernBERT-base".to_string(),
            ModernBertSize::Large => "answerdotai/ModernBERT-large".to_string(),
        });

        println!("Using model: {}", model_id);
        println!("Using revision: {}", self.revision);
        println!("Using device: {:?}", device);

        let repo = api.repo(Repo::with_revision(
            model_id,
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

        let config_filename = match &self.config_file {
            Some(file) => file.clone(),
            None => {
                println!("Fetching config...");
                repo.get("config.json")?
            }
        };

        let weights_filename = match &self.weight_files {
            Some(files) => files.clone(),
            None => {
                println!("Fetching model weights...");
                match repo.get("model.safetensors") {
                    Ok(safetensors) => safetensors,
                    Err(_) => match repo.get("pytorch_model.bin") {
                        Ok(pytorch_model) => pytorch_model,
                        Err(e) => {
                            anyhow::bail!("Model weights not found in repo. Expected `model.safetensors` or `pytorch_model.bin`. Error: {e}")
                        }
                    },
                }
            }
        };

        println!("Loading configuration...");
        let config_content = std::fs::read_to_string(&config_filename).map_err(|e| {
            E::msg(format!(
                "Failed to read config file {:?}: {}",
                config_filename, e
            ))
        })?;
        let config: modernbert::Config = serde_json::from_str(&config_content).map_err(|e| {
            E::msg(format!(
                "Failed to parse config file {:?}: {}",
                config_filename, e
            ))
        })?;

        println!("Loading tokenizer...");
        let mut tokenizer = Tokenizer::from_file(&tokenizer_filename).map_err(|e| {
            E::msg(format!(
                "Failed to load tokenizer from {:?}: {}",
                tokenizer_filename, e
            ))
        })?;

        // Configure padding
        let pad_token_id = config.pad_token_id; // Use pad_token_id from config
        let pp = PaddingParams {
            strategy: tokenizers::PaddingStrategy::BatchLongest, // Or Fixed(512) or specify
            pad_id: pad_token_id,
            pad_token: tokenizer
                .id_to_token(pad_token_id)
                .unwrap_or_else(|| "[PAD]".to_string()), // Use the actual token if available
            ..Default::default()
        };
        tokenizer.with_padding(Some(pp));
        // No truncation typically needed for fill-mask, but could be added:
        // tokenizer.with_truncation(None);

        println!("Loading model weights...");
        let vb = if weights_filename
            .extension()
            .map_or(false, |ext| ext == "safetensors")
        {
            println!("Loading weights from safetensors...");
            unsafe {
                VarBuilder::from_mmaped_safetensors(&[weights_filename], DType::F32, &device)?
            }
        } else if weights_filename
            .extension()
            .map_or(false, |ext| ext == "bin")
        {
            println!("Loading weights from pytorch_model.bin...");
            VarBuilder::from_pth(&weights_filename, DType::F32, &device)?
        } else {
            anyhow::bail!("Unsupported weight file format: {:?}", weights_filename);
        };

        println!("Building model...");
        let model = ModernBertForMaskedLM::load(vb, &config)?;
        println!("Model built successfully.");

        Ok(FillMaskPipeline {
            model,
            tokenizer,
            device,
            config,
        }) // Pass device and config
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
    model: ModernBertForMaskedLM,
    tokenizer: Tokenizer,
    device: Device,
    config: modernbert::Config, // Store config to access mask_token_id later
}

impl FillMaskPipeline {
    /// Fills the `[MASK]` token in a given text string.
    ///
    /// Expects the input `text` to contain exactly one instance of `"[MASK]"`.
    /// Returns the text with the mask replaced by the predicted token.
    pub fn fill_mask(&self, text: &str) -> Result<String> {
        if text.matches("[MASK]").count() != 1 {
            anyhow::bail!("Input text must contain exactly one '[MASK]' token.");
        }

        // 1. Tokenize
        let tokens = self.tokenizer.encode(text, true).map_err(E::msg)?;
        let token_ids = tokens.get_ids();
        let attention_mask_vals = tokens.get_attention_mask();

        // Find mask token index using the tokenizer
        let mask_token_id = self
            .tokenizer
            .token_to_id("[MASK]")
            .ok_or_else(|| E::msg("Tokenizer does not contain a '[MASK]' token."))?;

        let mask_index = token_ids
            .iter()
            .position(|&id| id == mask_token_id)
            .ok_or_else(|| {
                E::msg(format!(
                    "Could not find mask token ID {} in tokenized input.",
                    mask_token_id
                ))
            })?;

        // 2. Prepare tensors
        let input_ids_tensor = Tensor::new(&token_ids[..], &self.device)?.unsqueeze(0)?;
        let attention_mask_tensor =
            Tensor::new(&attention_mask_vals[..], &self.device)?.unsqueeze(0)?;

        // 3. Forward pass
        let output = self
            .model
            .forward(&input_ids_tensor, &attention_mask_tensor)?
            .squeeze(0)? // Remove batch dim
            .to_dtype(DType::F32)?; // Ensure F32 for argmax

        // 4. Get prediction at mask position
        let logits_at_mask = output.i(mask_index)?;
        let predicted_token_id = logits_at_mask.argmax(0)?.to_scalar::<u32>()?;

        // 5. Decode the predicted token
        let predicted_token = self
            .tokenizer
            .decode(&[predicted_token_id], true)
            .map_err(E::msg)?
            .trim() // Often decoded tokens have extra spaces
            .to_string();

        // 6. Replace "[MASK]" in the original string
        let result = text.replace("[MASK]", &predicted_token);

        Ok(result)
    }
}

// Note: The original example included helper functions `tokenize_batch` and `get_attention_mask`.
// For this pipeline, which processes one string at a time, the logic is integrated directly
// into the `fill_mask` method for simplicity.
