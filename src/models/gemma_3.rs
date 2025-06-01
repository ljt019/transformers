use crate::models::raw::models::quantized_gemma3;
use crate::models::shared::get_global_shared_model_cache;
use crate::pipelines::TextGenerationModel;
use crate::utils::configs::ModelConfig;
use crate::utils::gguf_cache::create_model_weights_from_cache;
use crate::utils::loaders::{HfLoader, TokenizerLoader};
use crate::utils::model_cache::ModelCacheKey;
use minijinja::{context, Environment};
use parking_lot::RwLock;
use serde_json::Value;
use std::sync::Arc;

use super::generate_tokens_from_prompt;
use crate::Message;

// Use the canonical Gemma3Size from the pipeline module
pub use crate::pipelines::text_generation_pipeline::Gemma3Size;

pub struct QuantizedGemma3Model {
    pipeline_state: Arc<RwLock<quantized_gemma3::PipelineState>>,
    config: ModelConfig,
}

impl QuantizedGemma3Model {
    pub fn new(config: ModelConfig, size: Gemma3Size) -> anyhow::Result<Self> {
        // Create cache key for this model configuration using a string identifier
        let model_identifier = match size {
            Gemma3Size::Size1B => "unsloth/gemma-3-1b-it-GGUF/gemma-3-1b-it-Q4_K_M.gguf",
            Gemma3Size::Size4B => "unsloth/gemma-3-4b-it-GGUF/gemma-3-4b-it-Q4_K_M.gguf",
            Gemma3Size::Size12B => "unsloth/gemma-3-12b-it-GGUF/gemma-3-12b-it-Q4_K_M.gguf",
            Gemma3Size::Size27B => "unsloth/gemma-3-27b-it-GGUF/gemma-3-27b-it-Q4_K_M.gguf",
        };
        let cache_key = ModelCacheKey::new(model_identifier, &config.device)?;

        // Get shared weights or load new ones
        let shared_cache = get_global_shared_model_cache();
        let shared_weights = shared_cache.get_or_load_gemma3_weights(cache_key, || {
            QuantizedGemma3Model::load_model_weights(config.device.clone(), size.clone())
        })?;

        // Create pipeline state with shared weights and individual KV caches
        let pipeline_state = quantized_gemma3::PipelineState::new(shared_weights);
        let pipeline_state_arc = Arc::new(RwLock::new(pipeline_state));

        Ok(Self {
            pipeline_state: pipeline_state_arc,
            config,
        })
    }

    pub fn load_model_weights(
        device: candle_core::Device,
        size: Gemma3Size,
    ) -> anyhow::Result<quantized_gemma3::Weights> {
        let (repo, file_name) = match size {
            Gemma3Size::Size1B => ("unsloth/gemma-3-1b-it-GGUF", "gemma-3-1b-it-Q4_K_M.gguf"),
            Gemma3Size::Size4B => ("unsloth/gemma-3-4b-it-GGUF", "gemma-3-4b-it-Q4_K_M.gguf"),
            Gemma3Size::Size12B => ("unsloth/gemma-3-12b-it-GGUF", "gemma-3-12b-it-Q4_K_M.gguf"),
            Gemma3Size::Size27B => ("unsloth/gemma-3-27b-it-GGUF", "gemma-3-27b-it-Q4_K_M.gguf"),
        };

        create_model_weights_from_cache(
            repo,
            file_name,
            &device,
            |gguf_content, gguf_file, device| {
                quantized_gemma3::Weights::from_gguf(gguf_content, gguf_file, device)
                    .map_err(|e| anyhow::anyhow!("Failed to create Gemma3 model weights: {}", e))
            },
        )
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

    fn format_messages(&self, messages: Vec<Message>) -> anyhow::Result<String> {
        // Create a loader for the tokenizer config (using the 1B model's repo)
        let tokenizer_config_loader =
            HfLoader::new("google/gemma-3-1b-it", "tokenizer_config.json");

        // Loads the tokenizer_config.json file
        let tokenizer_config_path = tokenizer_config_loader
            .load()
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer config: {}", e))?;
        let tokenizer_config_content = std::fs::read_to_string(tokenizer_config_path)
            .map_err(|e| anyhow::anyhow!("Failed to read tokenizer config file: {}", e))?;

        // Parse JSON and get the 'chat_template'
        let config_json: Value = serde_json::from_str(&tokenizer_config_content)
            .map_err(|e| anyhow::anyhow!("Failed to parse tokenizer config JSON: {}", e))?;
        let chat_template = config_json["chat_template"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Missing 'chat_template' field in tokenizer config"))?;

        // Create a minijinja environment
        let mut env = Environment::new();
        env.add_template("chat", chat_template)
            .map_err(|e| anyhow::anyhow!("Failed to add chat template: {}", e))?;

        let tmpl = env
            .get_template("chat")
            .map_err(|e| anyhow::anyhow!("Failed to get chat template: {}", e))?;

        // Render the template
        let rendered = tmpl
            .render(context! {
                messages => messages,
                add_generation_prompt => true, // Common practice, adjust if needed
            })
            .map_err(|e| anyhow::anyhow!("Failed to render chat template: {}", e))?;

        Ok(rendered)
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
        let mut pipeline_state_guard = self.pipeline_state.write();

        let response_tokens = generate_tokens_from_prompt(
            prompt_tokens,
            &self.config.params,
            &mut *pipeline_state_guard,
            max_len,
            &self.config.device,
            eos_token,
        )?;

        Ok(response_tokens)
    }
}
