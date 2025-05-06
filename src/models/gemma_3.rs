use crate::models::raw::models::quantized_gemma3;
use crate::pipelines::TextGenerationModel;
use crate::utils::configs::ModelConfig;
use crate::utils::loaders::{GgufModelLoader, LoadedGgufModelWeights, TokenizerLoader};
use std::cell::RefCell;

pub enum Gemma3Size {
    Size1B,
    Size4B,
    Size12B,
    Size27B,
}

pub struct QuantizedGemma3Model {
    weights: RefCell<quantized_gemma3::ModelWeights>,
    format_prompt: fn(&str) -> String,
    eos_token: String,
    config: ModelConfig,
}

impl QuantizedGemma3Model {
    pub fn new(config: ModelConfig) -> anyhow::Result<Self> {
        let weights = RefCell::new(quantized_gemma3::ModelWeights::default());
        let format_prompt = |prompt: &str| -> String {
            format!("<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n")
        };
    }
}

impl TextGenerationModel for QuantizedGemma3Model {
    fn load_tokenizer(&self) -> anyhow::Result<tokenizers::Tokenizer> {
        // The tokenizer is the same for all sizes, so we can just use the 1B model
        let tokenizer_loader = TokenizerLoader::new("google/gemma-3-1b-it", "tokenizer.json");

        let tokenizer = tokenizer_loader.load_tokenizer()?;

        Ok(tokenizer)
    }

    fn load_model_weights(&self, size: Gemma3Size) -> anyhow::Result<LoadedGgufModelWeights> {
        let (repo, file_name) = match size {
            Gemma3Size::Size1B => ("unsloth/gemma-3-1b-it-GGUF", "gemma-3-1b-it-Q4_K_M.gguf"),
            Gemma3Size::Size4B => ("unsloth/gemma-3-4b-it-GGUF", "gemma-3-4b-it-Q4_K_M.gguf"),
            Gemma3Size::Size12B => ("unsloth/gemma-3-12b-it-GGUF", "gemma-3-12b-it-Q4_K_M.gguf"),
            Gemma3Size::Size27B => ("unsloth/gemma-3-27b-it-GGUF", "gemma-3-27b-it-Q4_K_M.gguf"),
        };

        let gguf_loader = GgufModelLoader::new(self.config.device, repo, file_name);

        let (gguf_file, gguf_content) = gguf_loader.load()?;

        let weights = quantized_gemma3::ModelWeights::from_gguf(
            gguf_content,
            &mut gguf_file,
            &self.config.device,
        )?;

        Ok(weights)
    }

    fn prompt_with_tokens(&self, prompt_tokens: &[u32], max_len: usize) -> anyhow::Result<String> {}
}
