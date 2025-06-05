use crate::models::raw::models::quantized_qwen3;
use crate::models::shared::get_global_shared_model_cache;
use crate::utils::configs::ModelConfig;
use crate::utils::gguf_cache::create_model_weights_from_cache;
use crate::utils::loaders::{HfLoader, TokenizerLoader};
use crate::utils::model_cache::ModelCacheKey;
use minijinja::{context, Environment};
use parking_lot::RwLock;
use serde_json::Value;
use std::sync::Arc;

use crate::models::generate_tokens_from_prompt;
use crate::pipelines::TextGenerationModel;
use crate::Message;

// Use the canonical Qwen3Size from the pipeline module
pub use crate::pipelines::text_generation::builder::Qwen3Size;

pub struct QuantizedQwen3Model {
    pipeline_state: Arc<RwLock<quantized_qwen3::PipelineState>>,
    config: ModelConfig,
}

impl QuantizedQwen3Model {
    pub fn new(config: ModelConfig, size: Qwen3Size) -> anyhow::Result<Self> {
        // Create cache key for this model configuration using a string identifier
        let model_identifier = match size {
            Qwen3Size::Size0_6B => {
                "Qwen/Qwen2.5-0.5B-Instruct-GGUF/qwen2.5-0.5b-instruct-q4_k_m.gguf"
            }
            Qwen3Size::Size1_7B => {
                "Qwen/Qwen2.5-1.5B-Instruct-GGUF/qwen2.5-1.5b-instruct-q4_k_m.gguf"
            }
            Qwen3Size::Size4B => "Qwen/Qwen2.5-3B-Instruct-GGUF/qwen2.5-3b-instruct-q4_k_m.gguf",
            Qwen3Size::Size8B => "Qwen/Qwen2.5-7B-Instruct-GGUF/qwen2.5-7b-instruct-q4_k_m.gguf",
            Qwen3Size::Size14B => "Qwen/Qwen2.5-14B-Instruct-GGUF/qwen2.5-14b-instruct-q4_k_m.gguf",
            Qwen3Size::Size32B => "Qwen/Qwen2.5-32B-Instruct-GGUF/qwen2.5-32b-instruct-q4_k_m.gguf",
        };
        let cache_key = ModelCacheKey::new(model_identifier, &config.device)?;

        // Get shared weights or load new ones
        let shared_cache = get_global_shared_model_cache();
        let shared_weights = shared_cache.get_or_load_qwen3_weights(cache_key, || {
            QuantizedQwen3Model::load_model_weights(config.device.clone(), size.clone())
        })?;

        // Create pipeline state with shared weights and individual KV caches
        let pipeline_state = quantized_qwen3::PipelineState::new(shared_weights);
        let pipeline_state_arc = Arc::new(RwLock::new(pipeline_state));

        Ok(Self {
            pipeline_state: pipeline_state_arc,
            config,
        })
    }

    pub fn load_model_weights(
        device: candle_core::Device,
        size: Qwen3Size,
    ) -> anyhow::Result<quantized_qwen3::Weights> {
        let (repo, file_name) = match size {
            Qwen3Size::Size0_6B => ("unsloth/Qwen3-0.6B-GGUF", "Qwen3-0.6B-Q4_K_M.gguf"),
            Qwen3Size::Size1_7B => ("unsloth/Qwen3-1.7B-GGUF", "Qwen3-1.7B-Q4_K_M.gguf"),
            Qwen3Size::Size4B => ("unsloth/Qwen3-4B-GGUF", "Qwen3-4B-Q4_K_M.gguf"),
            Qwen3Size::Size8B => ("unsloth/Qwen3-8B-GGUF", "Qwen3-8B-Q4_K_M.gguf"),
            Qwen3Size::Size14B => ("unsloth/Qwen3-14B-GGUF", "Qwen3-14B-Q4_K_M.gguf"),
            Qwen3Size::Size32B => ("unsloth/Qwen3-32B-GGUF", "Qwen3-32B-Q4_K_M.gguf"),
        };

        create_model_weights_from_cache(
            repo,
            file_name,
            &device,
            |gguf_content, gguf_file, device| {
                quantized_qwen3::Weights::from_gguf(gguf_content, gguf_file, device)
                    .map_err(|e| anyhow::anyhow!("Failed to create Qwen3 weights: {}", e))
            },
        )
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

    fn format_messages(&self, messages: Vec<Message>) -> anyhow::Result<String> {
        // Create a loader for the tokenizer config
        let tokenizer_config_loader = HfLoader::new("Qwen/Qwen3-0.6B", "tokenizer_config.json");

        // Loads the tokenizer_config.json file and returns the path to the json file as a PathBuf
        let tokenizer_config_path = tokenizer_config_loader
            .load()
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer config: {}", e))?;
        let tokenizer_config_content = std::fs::read_to_string(tokenizer_config_path)
            .map_err(|e| anyhow::anyhow!("Failed to read tokenizer config: {}", e))?;

        // Parse JSON and get the 'chat_template'
        let config_json: Value = serde_json::from_str(&tokenizer_config_content)
            .map_err(|e| anyhow::anyhow!("Failed to parse tokenizer config: {}", e))?;
        let mut chat_template_str = config_json["chat_template"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Missing 'chat_template' field in tokenizer config"))?
            .to_string();

        // Perform targeted template fixes:
        // 1) Reverse list usage messages[::-1] -> messages|reverse
        chat_template_str = chat_template_str.replace("messages[::-1]", "messages|reverse");

        // 2) Convert Python method calls .startswith and .endswith to Jinja tests
        chat_template_str = chat_template_str.replace(
            ".startswith('<tool_response>')",
            " is startingwith \"<tool_response>\"",
        );
        chat_template_str = chat_template_str.replace(
            ".endswith('</tool_response>')",
            " is endingwith \"<tool_response>\"",
        );

        // 3) Targeted fixes for known problematic lines based on the debug output of chat_template_str

        // Fixing: {%- set content = message.content.split('</think>')[-1].lstrip('\n') %}
        chat_template_str = chat_template_str.replace(
            "{%- set content = message.content.split('</think>')[-1].lstrip('\\n') %}",
            "{%- set content = lstrip(last(split(message.content, \"</think>\")), \"\\n\") %}",
        );

        // Fixing: {%- set reasoning_content = message.content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n') %}
        chat_template_str = chat_template_str.replace(
            "{%- set reasoning_content = message.content.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n') %}",
            "{%- set reasoning_content = lstrip(last(split(rstrip(first(split(message.content, \"</think>\")), \"\\n\"), \"<think>\")), \"\\n\") %}"
        );

        // Fixing: reasoning_content.strip('\n') within the assistant message block
        chat_template_str = chat_template_str.replace(
            "reasoning_content.strip('\\n')",
            "strip(reasoning_content, '\\n')",
        );

        // Fixing: content.lstrip('\n') at the end of the assistant message block
        chat_template_str =
            chat_template_str.replace("+ content.lstrip('\\n') }}", "+ lstrip(content, '\\n') }}");

        // Create a minijinja environment
        let mut env = Environment::new();

        // Register filters and tests for string operations
        // Qwen3 is the only model i've had to do this with so for, it's weird
        // It's jinja template just has weird pythonic stuff in it
        env.add_test("startingwith", |value: &str, prefix: &str| {
            value.starts_with(prefix)
        });
        env.add_test("endingwith", |value: &str, suffix: &str| {
            value.ends_with(suffix)
        });
        env.add_function("split", |v: String, sep: String| -> Vec<String> {
            v.split(&sep).map(String::from).collect()
        });
        env.add_function("first", |v: Vec<String>| -> String {
            v.first().cloned().unwrap_or_default()
        });
        env.add_function("last", |v: Vec<String>| -> String {
            v.last().cloned().unwrap_or_default()
        });
        env.add_function("lstrip", |s: String, pat: String| -> String {
            let c = pat.chars().next().unwrap_or('\0'); // Get first char or null
            s.trim_start_matches(c).to_string()
        });
        env.add_function("rstrip", |s: String, pat: String| -> String {
            let c = pat.chars().next().unwrap_or('\0');
            s.trim_end_matches(c).to_string()
        });
        env.add_function("strip", |s: String, pat: String| -> String {
            let c = pat.chars().next().unwrap_or('\0');
            s.trim_matches(c).to_string()
        });

        // Add the patched template
        env.add_template("chat", &chat_template_str)
            .map_err(|e| anyhow::anyhow!("Failed to add chat template: {}", e))?;

        // Get the template
        let tmpl = env
            .get_template("chat")
            .map_err(|e| anyhow::anyhow!("Failed to get chat template: {}", e))?;

        // Convert messages to a format that works better with Jinja2
        let messages_dicts: Vec<serde_json::Value> = messages
            .into_iter()
            .map(|msg| {
                serde_json::json!({
                    "role": msg.role(),
                    "content": msg.content(),
                })
            })
            .collect();

        // Render the template
        let rendered = tmpl
            .render(context! {
                messages => messages_dicts,
                add_generation_prompt => true,
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
