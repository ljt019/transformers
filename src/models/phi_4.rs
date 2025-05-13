use crate::models::raw::models::quantized_phi3;
use crate::utils::configs::ModelConfig;
use crate::utils::loaders::{GgufModelLoader, HfLoader, TokenizerLoader};
use minijinja::{context, Environment};
use serde_json::Value;
use std::cell::RefCell;

use crate::models::generate_tokens_from_prompt;
use crate::pipelines::TextGenerationModel;
use crate::Message;

#[derive(Clone)]
pub enum Phi4Size {
    Size14B,
}

pub struct QuantizedPhi4Model {
    weights: RefCell<quantized_phi3::ModelWeights>,
    config: ModelConfig,
}

impl QuantizedPhi4Model {
    pub fn new(config: ModelConfig, size: Phi4Size) -> anyhow::Result<Self> {
        let specific_weights =
            QuantizedPhi4Model::load_model_weights(config.device.clone(), size.clone())?;
        let weights_refcell = RefCell::new(specific_weights);

        Ok(Self {
            weights: weights_refcell,
            config,
        })
    }

    pub fn load_model_weights(
        device: candle_core::Device,
        size: Phi4Size,
    ) -> anyhow::Result<quantized_phi3::ModelWeights> {
        let (repo, file_name) = match size {
            Phi4Size::Size14B => ("microsoft/phi-4-gguf", "phi-4-q4.gguf"),
        };

        let gguf_loader = GgufModelLoader::new(repo, file_name);

        let (mut gguf_file, gguf_content) = gguf_loader.load()?;

        let phi3_model_weights =
            quantized_phi3::ModelWeights::from_gguf(false, gguf_content, &mut gguf_file, &device)?;

        Ok(phi3_model_weights)
    }
}

impl TextGenerationModel for QuantizedPhi4Model {
    fn load_tokenizer(&self) -> anyhow::Result<tokenizers::Tokenizer> {
        let tokenizer_loader = TokenizerLoader::new("microsoft/phi-4", "tokenizer.json");

        let tokenizer = tokenizer_loader.load()?;

        Ok(tokenizer)
    }

    fn get_eos_token_str(&self) -> &str {
        "<|im_end|>"
    }

    fn format_prompt(&self, prompt: &str) -> String {
        format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n")
    }

    fn format_messages(&self, messages: Vec<Message>) -> String {
        // Create a loader for the tokenizer config
        let tokenizer_config_loader = HfLoader::new("microsoft/phi-4", "tokenizer_config.json");

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
