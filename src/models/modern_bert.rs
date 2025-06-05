use crate::models::raw::models::modernbert;
use crate::models::raw::models::modernbert::ModernBertForMaskedLM;
use crate::pipelines::fill_mask::pipeline::ModernBertSize;
use crate::utils::load_device;
use anyhow::{Error as E, Result};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::VarBuilder;
use hf_hub::{api::sync::Api, Repo, RepoType};
use tokenizers::Tokenizer;

pub struct ModernBertModel {
    model: ModernBertForMaskedLM,
    device: Device,
    config: modernbert::Config,
}

impl ModernBertModel {
    pub fn new(size: ModernBertSize) -> Result<Self> {
        let device = load_device()?;

        let api = Api::new()?;
        let model_id = match size {
            ModernBertSize::Base => "answerdotai/ModernBERT-base".to_string(),
            ModernBertSize::Large => "answerdotai/ModernBERT-large".to_string(),
        };

        let repo = api.repo(Repo::new(model_id, RepoType::Model));

        let config_filename = repo.get("config.json")?;

        let weights_filename = {
            match repo.get("model.safetensors") {
                Ok(safetensors) => safetensors,
                Err(_) => match repo.get("pytorch_model.bin") {
                    Ok(pytorch_model) => pytorch_model,
                    Err(e) => {
                        anyhow::bail!("Model weights not found in repo. Expected `model.safetensors` or `pytorch_model.bin`. Error: {e}")
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

        let config: modernbert::Config = serde_json::from_str(&config_content).map_err(|e| {
            E::msg(format!(
                "Failed to parse config file {:?}: {}",
                config_filename, e
            ))
        })?;

        let vb = if weights_filename
            .extension()
            .map_or(false, |ext| ext == "safetensors")
        {
            unsafe {
                VarBuilder::from_mmaped_safetensors(&[weights_filename], DType::F32, &device)?
            }
        } else if weights_filename
            .extension()
            .map_or(false, |ext| ext == "bin")
        {
            VarBuilder::from_pth(&weights_filename, DType::F32, &device)?
        } else {
            anyhow::bail!("Unsupported weight file format: {:?}", weights_filename);
        };

        let model = ModernBertForMaskedLM::load(vb, &config)?;

        Ok(Self {
            model,
            device,
            config,
        })
    }

    pub fn forward(
        &self,
        input_ids: &Tensor,
        attention_mask: &Tensor,
    ) -> candle_core::Result<Tensor> {
        self.model.forward(input_ids, attention_mask)
    }

    pub fn fill_mask(&self, tokenizer: &Tokenizer, text: &str) -> Result<String> {
        if text.matches("[MASK]").count() != 1 {
            anyhow::bail!("Input text must contain exactly one '[MASK]' token.");
        }

        // 1. Tokenize
        let tokens = tokenizer.encode(text, true).map_err(E::msg)?;
        let token_ids = tokens.get_ids();
        let attention_mask_vals = tokens.get_attention_mask();

        // Find mask token index using the tokenizer
        let mask_token_id = tokenizer
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
            .forward(&input_ids_tensor, &attention_mask_tensor)?
            .squeeze(0)? // Remove batch dim
            .to_dtype(DType::F32)?; // Ensure F32 for argmax

        // 4. Get prediction at mask position
        let logits_at_mask = output.i(mask_index)?;
        let predicted_token_id = logits_at_mask.argmax(0)?.to_scalar::<u32>()?;

        // 5. Decode the predicted token
        let predicted_token = tokenizer
            .decode(&[predicted_token_id], true)
            .map_err(E::msg)?
            .trim() // Often decoded tokens have extra spaces
            .to_string();

        // 6. Replace "[MASK]" in the original string
        let result = text.replace("[MASK]", &predicted_token);

        Ok(result)
    }

    pub fn get_pad_token_id(&self) -> u32 {
        self.config.pad_token_id
    }
}
