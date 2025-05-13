use crate::models::raw::models::quantized_qwen3;
use crate::utils::configs::ModelConfig;
use crate::utils::loaders::{GgufModelLoader, HfLoader, TokenizerLoader};
use minijinja::{context, Environment};
use serde_json::Value;
use std::cell::RefCell;

use crate::models::generate_tokens_from_prompt;
use crate::pipelines::TextGenerationModel;
use crate::Message;
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

    fn format_messages(&self, messages: Vec<Message>) -> String {
        // Create a loader for the tokenizer config
        let tokenizer_config_loader = HfLoader::new("Qwen/Qwen3-0.6B", "tokenizer_config.json");

        // Loads the tokenizer_config.json file and returns the path to the json file as a PathBuf
        let tokenizer_config_path = tokenizer_config_loader.load().unwrap();
        let tokenizer_config_content = std::fs::read_to_string(tokenizer_config_path).unwrap();

        // Parse JSON and get the 'chat_template'
        let config_json: Value = serde_json::from_str(&tokenizer_config_content).unwrap();
        let mut chat_template_str = config_json["chat_template"].as_str().unwrap().to_string();

        // Perform targeted template fixes:
        // 1) Reverse list usage
        chat_template_str = chat_template_str.replace("messages[::-1]", "messages|reverse");
        // 2) Convert Python method calls to Jinja functions/tests
        chat_template_str = chat_template_str.replace(
            ".startswith('<tool_response>')",
            " is startingwith \"<tool_response>\"",
        );
        chat_template_str = chat_template_str.replace(
            ".endswith('</tool_response>')",
            " is endingwith \"</tool_response>\"",
        );
        // 3) Replace split and method chaining with function pipeline
        chat_template_str = chat_template_str.replace(
            "message.content.split('</think>')[-1].lstrip('\n')",
            "lstrip(last(split(message.content, \"</think>\")), \"\\n\")",
        );
        chat_template_str = chat_template_str.replace(
            "message.content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n')",
            "lstrip(last(split(rstrip(first(split(message.content, \"</think>\")), \"\\n\"), \"<think>\")), \"\\n\")"
        );

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
        env.add_template("chat", &chat_template_str).unwrap();

        // Get the template
        let tmpl = env.get_template("chat").unwrap();

        // Render the template
        let rendered = tmpl
            .render(context! {
                messages => messages,
                add_generation_prompt => true,
            })
            .unwrap();

        println!("{}", rendered);

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
