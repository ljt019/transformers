use crate::models::raw::models::quantized_phi3;
use crate::models::shared::get_global_shared_model_cache;
use crate::utils::configs::ModelConfig;
use crate::utils::gguf_cache::create_model_weights_from_cache;
use crate::utils::loaders::{HfLoader, TokenizerLoader};
use crate::utils::model_cache::ModelCacheKey;
use minijinja::{context, Environment};
use once_cell::sync::Lazy;
use parking_lot::RwLock;
use serde_json::Value;
use std::sync::Arc;

use crate::models::generate_tokens_from_prompt;
use crate::pipelines::TextGenerationModel;
use crate::Message;

// Use the canonical Phi4Size from the pipeline module
pub use crate::pipelines::text_generation::builder::Phi4Size;

// Cache the chat template content to avoid repeated disk I/O and parsing
static CHAT_TEMPLATE_CONTENT: Lazy<anyhow::Result<String>> = Lazy::new(|| {
    // Load the tokenizer config once
    let tokenizer_config_loader = HfLoader::new("microsoft/phi-4", "tokenizer_config.json");
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

    Ok(chat_template.to_owned())
});

pub struct QuantizedPhi4Model {
    pipeline_state: Arc<RwLock<quantized_phi3::PipelineState>>,
    config: ModelConfig,
}

impl QuantizedPhi4Model {
    pub fn new(config: ModelConfig, size: Phi4Size) -> anyhow::Result<Self> {
        // Create cache key for this model configuration using the exact same identifier as load_model_weights
        let model_identifier = match size {
            Phi4Size::Size14B => "microsoft/phi-4-gguf/phi-4-Q4_K.gguf",
        };
        let cache_key = ModelCacheKey::new(model_identifier, &config.device)?;

        // Get shared weights or load new ones
        let shared_cache = get_global_shared_model_cache();
        let shared_weights = shared_cache.get_or_load_phi4_weights(cache_key, || {
            QuantizedPhi4Model::load_model_weights(config.device.clone(), size.clone())
        })?;

        // Create pipeline state with shared weights and individual KV caches
        let pipeline_state = quantized_phi3::PipelineState::new(shared_weights);
        let pipeline_state_arc = Arc::new(RwLock::new(pipeline_state));

        Ok(Self {
            pipeline_state: pipeline_state_arc,
            config,
        })
    }

    pub fn load_model_weights(
        device: candle_core::Device,
        size: Phi4Size,
    ) -> anyhow::Result<quantized_phi3::Weights> {
        let (repo, file_name) = match size {
            Phi4Size::Size14B => ("microsoft/phi-4-gguf", "phi-4-Q4_K.gguf"),
        };

        create_model_weights_from_cache(
            repo,
            file_name,
            &device,
            |gguf_content, gguf_file, device| {
                quantized_phi3::Weights::from_gguf(false, gguf_content, gguf_file, device)
                    .map_err(|e| anyhow::anyhow!("Failed to create Phi4 model weights: {}", e))
            },
        )
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

    fn format_messages(&self, messages: Vec<Message>) -> anyhow::Result<String> {
        // Get the cached template content
        let binding = CHAT_TEMPLATE_CONTENT.as_ref();
        let template_content = binding
            .as_ref()
            .map_err(|e| anyhow::anyhow!("Failed to get cached chat template content: {}", e))?;

        // Create environment and add template (this is lightweight compared to file I/O)
        let mut env = Environment::new();
        env.add_template("chat", template_content)
            .map_err(|e| anyhow::anyhow!("Failed to add chat template: {}", e))?;

        let template = env
            .get_template("chat")
            .map_err(|e| anyhow::anyhow!("Failed to get chat template: {}", e))?;

        // Render the template
        let rendered = template
            .render(context! {
                messages => messages,
                add_generation_prompt => true, // Common practice, adjust if needed
            })
            .map_err(|e| anyhow::anyhow!("Failed to render chat template: {}", e))?;

        Ok(rendered)
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
