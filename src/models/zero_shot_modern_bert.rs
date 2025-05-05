/*
Zero-Shot Modern BERT Finetune

https://huggingface.co/MoritzLaurer/ModernBERT-large-zeroshot-v2.0

MoritzLaurer/ModernBERT-large-zeroshot-v2.0

CONFIG.JSON CONTENTS:
```
{
  "_name_or_path": "answerdotai/ModernBERT-large",
  "architectures": [
    "ModernBertForSequenceClassification"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 50281,
  "classifier_activation": "gelu",
  "classifier_bias": false,
  "classifier_dropout": 0.0,
  "classifier_pooling": "mean",
  "cls_token_id": 50281,
  "decoder_bias": true,
  "deterministic_flash_attn": false,
  "embedding_dropout": 0.0,
  "eos_token_id": 50282,
  "global_attn_every_n_layers": 3,
  "global_rope_theta": 160000.0,
  "gradient_checkpointing": false,
  "hidden_activation": "gelu",
  "hidden_size": 1024,
  "id2label": {
    "0": "entailment",
    "1": "not_entailment"
  },
  "initializer_cutoff_factor": 2.0,
  "initializer_range": 0.02,
  "intermediate_size": 2624,
  "label2id": {
    "entailment": 0,
    "not_entailment": 1
  },
  "layer_norm_eps": 1e-05,
  "local_attention": 128,
  "local_rope_theta": 10000.0,
  "max_position_embeddings": 8192,
  "mlp_bias": false,
  "mlp_dropout": 0.0,
  "model_type": "modernbert",
  "norm_bias": false,
  "norm_eps": 1e-05,
  "num_attention_heads": 16,
  "num_hidden_layers": 28,
  "pad_token_id": 50283,
  "position_embedding_type": "absolute",
  "problem_type": "single_label_classification",
  "reference_compile": true,
  "sep_token_id": 50282,
  "sparse_pred_ignore_index": -100,
  "sparse_prediction": false,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.48.0.dev0",
  "vocab_size": 50368
}
```


*/

use crate::models::raw::models::modernbert::{
    self, ClassifierConfig, ClassifierPooling, ModernBertForSequenceClassification,
};
use crate::utils::load_device;
use anyhow::{Error as E, Result};
use candle_core::{DType, Device, IndexOp, Tensor, D};
use candle_nn::{ops::softmax, VarBuilder};
use hf_hub::{api::sync::Api, Repo, RepoType};
use std::collections::HashMap;
use tokenizers::Tokenizer;

/// Available sizes of the zero-shot ModernBERT model
#[derive(Debug, Clone, Copy)]
pub enum ZeroShotModernBertSize {
    Base, // Base model is available
    Large,
}

pub struct ZeroShotModernBertModel {
    model: ModernBertForSequenceClassification,
    device: Device,
    label2id: HashMap<String, u32>, // Store ID as u32
}

impl ZeroShotModernBertModel {
    pub fn new(size: ZeroShotModernBertSize) -> Result<Self> {
        let device = load_device()?;

        let model_id = match size {
            ZeroShotModernBertSize::Base => {
                "MoritzLaurer/ModernBERT-base-zeroshot-v2.0".to_string()
            }
            ZeroShotModernBertSize::Large => {
                "MoritzLaurer/ModernBERT-large-zeroshot-v2.0".to_string()
            }
        };

        let api = Api::new()?;
        let repo = api.repo(Repo::new(model_id.clone(), RepoType::Model)); // Clone model_id for tokenizer info

        let config_filename = repo.get("config.json")?;

        let weights_filename = {
            match repo.get("model.safetensors") {
                Ok(safetensors) => safetensors,
                Err(_) => match repo.get("pytorch_model.bin") {
                    Ok(pytorch_model) => pytorch_model,
                    Err(e) => {
                        anyhow::bail!("Model weights not found in repo {}. Expected `model.safetensors` or `pytorch_model.bin`. Error: {e}", model_id)
                    }
                },
            }
        };

        let config_content = std::fs::read_to_string(&config_filename).map_err(|e| {
            E::msg(format!(
                "Failed to read config file {:?}: {}",
                config_filename, e
            ))
        })?;

        // Extract classification metadata (id2label, label2id) from the same JSON
        #[derive(serde::Deserialize)]
        struct ClassifierConfigRaw {
            id2label: HashMap<String, String>,
            label2id: HashMap<String, u32>, // Expect u32 integer values
            classifier_pooling: Option<ClassifierPooling>, // Make optional to handle missing field
        }
        let class_cfg: ClassifierConfigRaw = serde_json::from_str(&config_content)
            .map_err(|e| E::msg(format!("Failed to parse classifier config: {}", e)))?;
        let id2label = class_cfg.id2label;
        let label2id = class_cfg.label2id; // Now HashMap<String, u32>
                                           // Use specified pooling or default if missing (e.g. older config might not have it)
        let classifier_pooling = class_cfg
            .classifier_pooling
            .unwrap_or(ClassifierPooling::MEAN); // Default based on model card if missing

        // Deserialize the full model config
        let mut config: modernbert::Config = serde_json::from_str(&config_content)
            .map_err(|e| E::msg(format!("Failed to parse model config: {}", e)))?;

        // Inject classification metadata so head builds with correct class size
        // Need to convert label2id values back to String for ClassifierConfig, if needed by the raw model code.
        // Let's check if ClassifierConfig actually needs String IDs.
        // Looking at raw/models/modernbert.rs, ClassifierConfig is mostly informational, except maybe for sizing.
        // The ModernBertForSequenceClassification::load seems to mainly use config.classifier_config.id2label.len()
        // Let's try cloning the u32 map directly first.
        // UPDATE: No, the `ClassifierConfig` struct itself expects `HashMap<String, String>` for label2id.
        // We need to convert the parsed u32 values back to strings for *that* struct, while keeping the u32 map for our own use.
        let label2id_for_config: HashMap<String, String> = label2id
            .iter()
            .map(|(k, v)| (k.clone(), v.to_string()))
            .collect();

        config.classifier_config = Some(ClassifierConfig {
            id2label: id2label.clone(),    // Clone for model struct
            label2id: label2id_for_config, // Use the converted map
            classifier_pooling,
        });

        // Determine DType - check config first, default to F32
        // Note: The config has "torch_dtype": "bfloat16", but VarBuilder might not handle bf16 directly from PTH/safetensors easily.
        // We'll load as F32 for broader compatibility for now. If performance requires BF16, specific handling would be needed.
        let dtype = DType::F32; // Defaulting to F32

        let vb = if weights_filename
            .extension()
            .map_or(false, |ext| ext == "safetensors")
        {
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights_filename], dtype, &device)? }
        } else if weights_filename
            .extension()
            .map_or(false, |ext| ext == "bin")
        {
            // VarBuilder::from_pth might require feature flags enabled in candle-nn
            VarBuilder::from_pth(&weights_filename, dtype, &device)?
        } else {
            anyhow::bail!("Unsupported weight file format: {:?}", weights_filename);
        };

        let model = ModernBertForSequenceClassification::load(vb, &config)?;

        Ok(Self {
            model,
            device,
            label2id, // Keep the parsed HashMap<String, u32>
        })
    }

    pub fn predict(
        &self,
        tokenizer: &Tokenizer,
        premise: &str,
        candidate_labels: &[&str],
    ) -> Result<Vec<(String, f32)>> {
        if candidate_labels.is_empty() {
            return Ok(vec![]);
        }

        // Find the ID for the "entailment" label from the config
        let entailment_id = *self // Dereference to get u32
            .label2id
            .get("entailment")
            .ok_or_else(|| E::msg("Config's label2id map does not contain 'entailment' key"))?;
        // No longer need to parse string, it's already u32
        let entailment_id_idx = entailment_id as usize; // Cast u32 to usize for indexing

        let mut encodings = Vec::new();
        for &label in candidate_labels {
            let hypothesis = format!("This example is {}.", label); // Standard NLI template
            let encoding = tokenizer
                .encode((premise, hypothesis.as_str()), true)
                .map_err(|e| E::msg(format!("Tokenization error: {}", e)))?;
            encodings.push(encoding);
        }

        // Pad the batch
        // Determine max length
        let max_len = encodings.iter().map(|e| e.len()).max().unwrap_or(0);

        // Get pad token id from tokenizer
        let pad_token_id = tokenizer
            .get_padding()
            .map(|p| p.pad_id)
            .or_else(|| tokenizer.token_to_id("<pad>")) // Common pad tokens
            .or_else(|| tokenizer.token_to_id("[PAD]"))
            .unwrap_or(0); // Default to 0 if no pad token found

        let mut all_token_ids: Vec<u32> = Vec::new();
        let mut all_attention_masks: Vec<u32> = Vec::new();

        for encoding in encodings {
            let mut token_ids = encoding.get_ids().to_vec();
            let mut attention_mask = encoding.get_attention_mask().to_vec();
            let current_len = token_ids.len();

            // Pad
            token_ids.resize(max_len, pad_token_id);
            attention_mask.resize(max_len, 0); // Pad mask is 0

            all_token_ids.extend(token_ids);
            all_attention_masks.extend(attention_mask);
        }

        // Create tensors
        let input_ids_tensor = Tensor::from_vec(
            all_token_ids,
            (candidate_labels.len(), max_len),
            &self.device,
        )?;
        let attention_mask_tensor = Tensor::from_vec(
            all_attention_masks,
            (candidate_labels.len(), max_len),
            &self.device,
        )?;

        // Forward pass
        let logits = self
            .model
            .forward(&input_ids_tensor, &attention_mask_tensor)?; // Logits shape: [batch_size, num_labels(2)]

        // Apply softmax to get probabilities
        let probabilities = softmax(&logits, D::Minus1)?; // Probabilities shape: [batch_size, 2]

        // Extract entailment probability (index depends on the loaded model's config)
        // We found entailment_id_idx earlier based on label2id
        let entailment_probs = probabilities.i((.., entailment_id_idx))?.to_vec1::<f32>()?;

        // Combine labels and scores
        let mut results: Vec<(String, f32)> = candidate_labels
            .iter()
            .map(|&label| label.to_string())
            .zip(entailment_probs.into_iter())
            .collect();

        // Sort by score descending
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        Ok(results)
    }

    pub fn get_tokenizer_repo_info(size: ZeroShotModernBertSize) -> String {
        match size {
            ZeroShotModernBertSize::Base => {
                "MoritzLaurer/ModernBERT-base-zeroshot-v2.0".to_string()
            }
            ZeroShotModernBertSize::Large => {
                "MoritzLaurer/ModernBERT-large-zeroshot-v2.0".to_string()
            }
        }
    }
}
