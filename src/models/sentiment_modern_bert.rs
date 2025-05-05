use crate::models::raw::models::modernbert::{self, ModernBertForSequenceClassification};
use crate::utils::load_device;
use anyhow::{Error as E, Result};
use candle_core::{DType, Device, Tensor, D};
use candle_nn::VarBuilder;
use hf_hub::{api::sync::Api, Repo, RepoType};
use std::{collections::HashMap, path::PathBuf};
use tokenizers::Tokenizer;

/// Available ModernBERT Sentiment model sizes.
#[derive(Debug, Clone, Copy)] // Added Copy
pub enum SentimentModernBertSize {
    Base,
    Large,
}

pub struct SentimentModernBertModel {
    model: ModernBertForSequenceClassification,
    device: Device,
    id2label: HashMap<String, String>,
}

impl SentimentModernBertModel {
    pub fn new(
        size: SentimentModernBertSize,
        model_id_override: Option<String>,
        revision: String,
        tokenizer_file: Option<PathBuf>,
        config_file: Option<PathBuf>,
        weight_files: Option<PathBuf>,
        cpu: bool,
    ) -> Result<Self> {
        let device = if cpu { Device::Cpu } else { load_device()? };

        let default_model_id = match size {
            SentimentModernBertSize::Base => {
                "clapAI/modernBERT-base-multilingual-sentiment".to_string()
            }
            SentimentModernBertSize::Large => {
                "clapAI/modernBERT-large-multilingual-sentiment".to_string()
            }
        };
        let model_id = model_id_override.unwrap_or(default_model_id);

        println!("Using sentiment model: {}", model_id);
        println!("Using revision: {}", revision);
        println!("Using device: {:?}", device);

        let api = Api::new()?;
        let repo = api.repo(Repo::with_revision(model_id, RepoType::Model, revision));

        let config_filename = match config_file {
            Some(file) => file,
            None => {
                println!("Fetching config...");
                repo.get("config.json")?
            }
        };

        let weights_filename = match weight_files {
            Some(files) => files,
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
        println!(
            "--- Raw Config Content ---\n{}\n-------------------------",
            config_content
        );

        // Extract classification metadata (id2label) from the same JSON
        #[derive(serde::Deserialize)]
        struct ClassifierConfigRaw {
            id2label: HashMap<String, String>,
        }
        let class_cfg: ClassifierConfigRaw = serde_json::from_str(&config_content)
            .map_err(|e| E::msg(format!("Failed to parse classifier config: {}", e)))?;
        let id2label = class_cfg.id2label;

        // Deserialize the full model config for masked-LM weights
        let mut config: modernbert::Config = serde_json::from_str(&config_content)
            .map_err(|e| E::msg(format!("Failed to parse model config: {}", e)))?;
        // Inject classification metadata so head builds with correct class size
        use crate::models::raw::models::modernbert::{ClassifierConfig, ClassifierPooling};
        let label2id = id2label
            .iter()
            .map(|(id, label)| (label.clone(), id.clone()))
            .collect();
        let pooling = match config
            .classifier_config
            .as_ref()
            .and_then(|c| Some(c.classifier_pooling))
        {
            Some(cp) => cp,
            None => ClassifierPooling::MEAN,
        };
        config.classifier_config = Some(ClassifierConfig {
            id2label: id2label.clone(),
            label2id,
            classifier_pooling: pooling,
        });
        println!(
            "--- Deserialized Config Struct ---
{:?}\n--------------------------------",
            config
        );
        println!(
            "--- Deserialized config.classifier_config ---
{:?}\n--------------------------------",
            config.classifier_config
        );

        println!("Loading model weights...");
        // Determine DType based on filename or config (assuming float16 for safetensors/bin if specified)
        // Defaulting to F32 for now, adjust if needed based on actual model weights
        // The config specifies torch_dtype: "float16", but candle needs explicit handling
        // TODO: Properly handle F16 loading if VarBuilder supports it easily or convert after loading.
        let dtype = DType::F32;
        let vb = if weights_filename
            .extension()
            .map_or(false, |ext| ext == "safetensors")
        {
            println!("Loading weights from safetensors ({:?})...", dtype);
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename], dtype, &device)? }
        } else if weights_filename
            .extension()
            .map_or(false, |ext| ext == "bin")
        {
            println!("Loading weights from pytorch_model.bin ({:?})...", dtype);
            VarBuilder::from_pth(&weights_filename, dtype, &device)?
        } else {
            anyhow::bail!("Unsupported weight file format: {:?}", weights_filename);
        };

        println!("Building model...");
        let model = ModernBertForSequenceClassification::load(vb, &config)?;
        println!("Sentiment model built successfully.");

        Ok(Self {
            model,
            device,
            id2label,
        })
    }

    pub fn predict(&self, tokenizer: &Tokenizer, text: &str) -> Result<String> {
        // 1. Tokenize
        let tokens = tokenizer
            .encode(text, true)
            .map_err(|e| E::msg(format!("Tokenization error: {}", e)))?;
        let token_ids = tokens.get_ids();
        let attention_mask_vals = tokens.get_attention_mask();

        // 2. Prepare tensors
        let input_ids_tensor = Tensor::new(&token_ids[..], &self.device)?.unsqueeze(0)?;
        let attention_mask_tensor =
            Tensor::new(&attention_mask_vals[..], &self.device)?.unsqueeze(0)?;

        // 3. Forward pass
        let output_logits = self
            .model
            .forward(&input_ids_tensor, &attention_mask_tensor)?;

        // 4. Get prediction
        // Assuming output_logits shape is [batch_size, num_labels]
        let predictions = output_logits
            .argmax(D::Minus1)?
            .squeeze(0)?
            .to_scalar::<u32>()?;
        let predicted_label = self
            .id2label
            .get(&predictions.to_string())
            .ok_or_else(|| {
                E::msg(format!(
                    "Predicted ID '{}' not found in id2label map",
                    predictions
                ))
            })?
            .clone();

        Ok(predicted_label)
    }

    pub fn get_tokenizer_repo_info(
        size: SentimentModernBertSize,
        model_id_override: Option<String>,
        revision: String,
    ) -> (String, String) {
        let default_model_id = match size {
            SentimentModernBertSize::Base => {
                "clapAI/modernBERT-base-multilingual-sentiment".to_string()
            }
            SentimentModernBertSize::Large => {
                "clapAI/modernBERT-large-multilingual-sentiment".to_string()
            }
        };
        let model_id = model_id_override.unwrap_or(default_model_id);
        (model_id, revision)
    }
}
