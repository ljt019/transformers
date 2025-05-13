use crate::models::raw::models::quantized_gemma3;
use crate::pipelines::TextGenerationModel;
use crate::utils::configs::ModelConfig;
use crate::utils::loaders::{GgufModelLoader, HfLoader, TokenizerLoader};
use minijinja::{context, Environment};
use serde_json::Value;
use std::cell::RefCell;

use super::generate_tokens_from_prompt;
use crate::Message;

#[derive(Clone)]
pub enum Gemma3Size {
    Size1B,
    Size4B,
    Size12B,
    Size27B,
}

pub struct QuantizedGemma3Model {
    weights: RefCell<quantized_gemma3::ModelWeights>,
    config: ModelConfig,
}

impl QuantizedGemma3Model {
    pub fn new(config: ModelConfig, size: Gemma3Size) -> anyhow::Result<Self> {
        let specific_weights =
            QuantizedGemma3Model::load_model_weights(config.device.clone(), size.clone())?;
        let weights_refcell = RefCell::new(specific_weights);

        Ok(Self {
            weights: weights_refcell,
            config: config,
        })
    }

    pub fn load_model_weights(
        device: candle_core::Device,
        size: Gemma3Size,
    ) -> anyhow::Result<quantized_gemma3::ModelWeights> {
        let (repo, file_name) = match size {
            Gemma3Size::Size1B => ("unsloth/gemma-3-1b-it-GGUF", "gemma-3-1b-it-Q4_K_M.gguf"),
            Gemma3Size::Size4B => ("unsloth/gemma-3-4b-it-GGUF", "gemma-3-4b-it-Q4_K_M.gguf"),
            Gemma3Size::Size12B => ("unsloth/gemma-3-12b-it-GGUF", "gemma-3-12b-it-Q4_K_M.gguf"),
            Gemma3Size::Size27B => ("unsloth/gemma-3-27b-it-GGUF", "gemma-3-27b-it-Q4_K_M.gguf"),
        };

        let gguf_loader = GgufModelLoader::new(repo, file_name);

        let (mut gguf_file, gguf_content) = gguf_loader.load()?;

        let gemma3_model_weights =
            quantized_gemma3::ModelWeights::from_gguf(gguf_content, &mut gguf_file, &device)?;

        Ok(gemma3_model_weights)
    }
}

impl TextGenerationModel for QuantizedGemma3Model {
    fn load_tokenizer(&self) -> anyhow::Result<tokenizers::Tokenizer> {
        // The tokenizer is the same for all sizes, so we can just use the 1B model
        let tokenizer_loader = TokenizerLoader::new("google/gemma-3-1b-it", "tokenizer.json");

        let tokenizer = tokenizer_loader.load()?;

        Ok(tokenizer)
    }

    fn get_eos_token_str(&self) -> &str {
        "<end_of_turn>"
    }

    fn format_messages(&self, messages: Vec<Message>) -> String {
        // Create a loader for the tokenizer config (using the 1B model's repo)
        let tokenizer_config_loader =
            HfLoader::new("google/gemma-3-1b-it", "tokenizer_config.json");

        // Loads the tokenizer_config.json file
        let tokenizer_config_path = tokenizer_config_loader.load().unwrap();
        let tokenizer_config_content = std::fs::read_to_string(tokenizer_config_path).unwrap();

        // Parse JSON and get the 'chat_template'
        let config_json: Value = serde_json::from_str(&tokenizer_config_content).unwrap();
        let chat_template = config_json["chat_template"].as_str().unwrap();

        // Create a minijinja environment
        let mut env = Environment::new();
        env.add_template("chat", chat_template).unwrap();

        let tmpl = env.get_template("chat").unwrap();

        // Render the template
        let rendered = tmpl
            .render(context! {
                messages => messages,
                add_generation_prompt => true, // Common practice, adjust if needed
            })
            .unwrap();

        rendered
    }

    fn format_prompt(&self, prompt: &str) -> String {
        format!("<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>assistant\n")
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
