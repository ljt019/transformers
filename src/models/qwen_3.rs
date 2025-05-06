use crate::models::raw::models::quantized_qwen3;
use crate::utils::configs::ModelConfig;
use crate::utils::loaders::{GgufModelLoader, LoadedGgufModelWeights, TokenizerLoader};
use std::cell::RefCell;

use crate::pipelines::TextGenerationModel;

pub enum Qwen3Size {
    Size0_6B,
    Size1_7B,
    Size4B,
    Size8B,
    Size14B,
    Size32B,
}

// Use the generic QuantizedModel with Qwen3 specific types and configuration
pub type QuantizedQwen3Model = crate::utils::QuantizedModel<quantized_qwen3::ModelWeights>;

impl QuantizedQwen3Model {
    pub fn new(config: ModelConfig) -> anyhow::Result<Self> {
        // Load weights using the new loader
        let loaded_weights = config.model_loader.load_weights()?;

        // Extract the Qwen3 weights from the enum
        let qwen3_weights = match loaded_weights {
            LoadedGgufModelWeights::Qwen3(weights) => Ok(weights),
            _ => Err(anyhow::anyhow!(
                "Loaded unexpected model type for Qwen3, expected Qwen3 weights"
            )),
        }?;

        // Wrap the weights in RefCell
        let weights = RefCell::new(qwen3_weights);

        // Define the prompt formatter for Qwen3
        let format_prompt = |prompt: &str| -> String {
            format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n")
        };

        Ok(Self {
            name: "Qwen3",
            weights,
            format_prompt,
            eos_token: "<|endoftext|>",
            config,
        })
    }
}

impl TextGenerationModel for QuantizedQwen3Model {
    fn load_tokenizer(&self) -> anyhow::Result<tokenizers::Tokenizer> {
        // The tokenizer is the same for all sizes, so we can just use the 0.6B model
        let tokenizer_loader = TokenizerLoader::new("Qwen/Qwen3-0.6B", "tokenizer.json");

        let tokenizer = tokenizer_loader.load_tokenizer()?;

        Ok(tokenizer)
    }

    fn load_model_weights(&self, size: Qwen3Size) -> anyhow::Result<LoadedGgufModelWeights> {
        let (repo, file_name) = match size {
            Qwen3Size::Size0_6B => ("unsloth/Qwen3-0.6B-GGUF", "Qwen3-0.6B-Q4_K_M.gguf"),
            Qwen3Size::Size1_7B => ("unsloth/Qwen3-1.7B-GGUF", "Qwen3-1.7B-Q4_K_M.gguf"),
            Qwen3Size::Size4B => ("unsloth/Qwen3-4B-GGUF", "Qwen3-4B-Q4_K_M.gguf"),
            Qwen3Size::Size8B => ("unsloth/Qwen3-8B-GGUF", "Qwen3-8B-Q4_K_M.gguf"),
            Qwen3Size::Size14B => ("unsloth/Qwen3-14B-GGUF", "Qwen3-14B-Q4_K_M.gguf"),
        };
    }

    fn prompt(&self, prompt: &str, max_len: usize) -> anyhow::Result<String> {
        todo!()
    }
}
