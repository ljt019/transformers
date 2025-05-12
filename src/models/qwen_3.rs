use crate::models::raw::models::quantized_qwen3;
use crate::utils::configs::ModelConfig;
use crate::utils::loaders::{GgufModelLoader, TokenizerLoader};
use std::cell::RefCell;

use crate::pipelines::TextGenerationModel;

use crate::models::generate_tokens_from_prompt;

#[derive(Clone)]
pub enum Qwen3Size {
    Size0_6B,
    Size1_7B,
    Size4B,
    Size8B,
    Size14B,
    Size32B,
}

pub struct QuantizedQwen3Model {
    weights: RefCell<quantized_qwen3::ModelWeights>,
    config: ModelConfig,
}

impl QuantizedQwen3Model {
    pub fn new(config: ModelConfig, size: Qwen3Size) -> anyhow::Result<Self> {
        let specific_weights =
            QuantizedQwen3Model::load_model_weights(config.device.clone(), size.clone())?;
        let weights_refcell = RefCell::new(specific_weights);

        Ok(Self {
            weights: weights_refcell,
            config,
        })
    }

    pub fn load_model_weights(
        device: candle_core::Device,
        size: Qwen3Size,
    ) -> anyhow::Result<quantized_qwen3::ModelWeights> {
        let (repo, file_name) = match size {
            Qwen3Size::Size0_6B => ("unsloth/Qwen3-0.6B-GGUF", "Qwen3-0.6B-Q4_K_M.gguf"),
            Qwen3Size::Size1_7B => ("unsloth/Qwen3-1.7B-GGUF", "Qwen3-1.7B-Q4_K_M.gguf"),
            Qwen3Size::Size4B => ("unsloth/Qwen3-4B-GGUF", "Qwen3-4B-Q4_K_M.gguf"),
            Qwen3Size::Size8B => ("unsloth/Qwen3-8B-GGUF", "Qwen3-8B-Q4_K_M.gguf"),
            Qwen3Size::Size14B => ("unsloth/Qwen3-14B-GGUF", "Qwen3-14B-Q4_K_M.gguf"),
            Qwen3Size::Size32B => ("unsloth/Qwen3-32B-GGUF", "Qwen3-32B-Q4_K_M.gguf"),
        };

        let gguf_loader = GgufModelLoader::new(repo, file_name);

        let (mut gguf_file, gguf_content) = gguf_loader.load()?;

        let qwen3_model_weights =
            quantized_qwen3::ModelWeights::from_gguf(gguf_content, &mut gguf_file, &device)?;

        Ok(qwen3_model_weights)
    }
}

impl TextGenerationModel for QuantizedQwen3Model {
    fn load_tokenizer(&self) -> anyhow::Result<tokenizers::Tokenizer> {
        // The tokenizer is the same for all sizes, so we can just use the 0.6B model
        let tokenizer_loader = TokenizerLoader::new("Qwen/Qwen3-0.6B", "tokenizer.json");

        let tokenizer = tokenizer_loader.load()?;

        Ok(tokenizer)
    }

    fn get_eos_token_str(&self) -> &str {
        "<|im_end|>"
    }

    fn format_prompt(&self, prompt: &str) -> String {
        format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n")
    }

    fn prompt_with_tokens(
        &self,
        prompt_tokens: &[u32],
        max_len: usize,
        eos_token: u32,
    ) -> anyhow::Result<Vec<u32>> {
        let mut specific_weights_ref_mut = self.weights.borrow_mut();

        let response_tokens = generate_tokens_from_prompt(
            prompt_tokens,
            &self.config.params,
            &mut *specific_weights_ref_mut,
            max_len,
            &self.config.device,
            eos_token,
        )?;

        Ok(response_tokens)
    }
}
