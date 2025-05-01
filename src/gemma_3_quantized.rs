use crate::pipelines::text_generation_pipeline::LargeLanguageModel;
use crate::token_output_stream::TokenOutputStream;
use candle_core::{Device, Tensor};
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::models::quantized_gemma3::ModelWeights;
use std::cell::RefCell;
use std::io::Write;
use tokenizers::Tokenizer;

use crate::utils::{load_device, load_model, HfConfig};

#[derive(Clone)]
pub struct ModelConfig {
    pub device: Device,
    pub hf_config: HfConfig,
    pub params: ModelParams,
}

impl ModelConfig {
    pub fn new(params: ModelParams, hf_config: HfConfig) -> anyhow::Result<Self> {
        let device = load_device()?;
        Ok(Self {
            device,
            hf_config,
            params,
        })
    }
}

#[derive(Clone)]
pub struct ModelParams {
    temperature: f64,
    repeat_penalty: f32,
    repeat_last_n: usize,
    seed: u64,
}

impl Default for ModelParams {
    fn default() -> Self {
        Self {
            temperature: 0.0,
            repeat_penalty: 1.1,
            repeat_last_n: 64,
            seed: 299792458,
        }
    }
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self::new(ModelParams::default(), HfConfig::default())
            .expect("Failed to create default model config")
    }
}

pub struct QuantizedGemma3Model {
    config: ModelConfig,
    weights: RefCell<ModelWeights>,
}

impl QuantizedGemma3Model {
    pub fn new(config: ModelConfig) -> anyhow::Result<Self> {
        println!("Loading model (this might take a while)...");
        let weights = load_model(&config.device, &config.hf_config)?;
        println!("Model loaded.");
        Ok(Self {
            config,
            weights: RefCell::new(weights),
        })
    }

    fn initialize_logits_processor(&self) -> LogitsProcessor {
        let sampling = if self.config.params.temperature <= 0. {
            Sampling::ArgMax
        } else {
            Sampling::All {
                temperature: self.config.params.temperature,
            }
        };
        LogitsProcessor::from_sampling(self.config.params.seed, sampling)
    }
}

impl LargeLanguageModel for QuantizedGemma3Model {
    fn prompt_model(
        &self,
        tokenizer: &Tokenizer,
        prompt: &str,
        max_length: usize,
    ) -> anyhow::Result<String> {
        let mut model_weights = self.weights.borrow_mut();

        let formatted_prompt =
            format!("<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n");
        let tokens = tokenizer
            .encode(formatted_prompt, true)
            .map_err(anyhow::Error::msg)?;
        let prompt_tokens = tokens.get_ids();
        let prompt_len = prompt_tokens.len();

        let mut logits_processor = self.initialize_logits_processor();
        let mut all_generated_tokens: Vec<u32> = Vec::with_capacity(max_length);

        let input = Tensor::new(prompt_tokens, &self.config.device)?.unsqueeze(0)?;
        let logits = model_weights.forward(&input, 0)?;
        let logits = logits.squeeze(0)?;
        let mut next_token = logits_processor.sample(&logits)?;
        all_generated_tokens.push(next_token);

        let eos_token_id = *tokenizer
            .get_vocab(true)
            .get("<end_of_turn>")
            .ok_or_else(|| anyhow::anyhow!("EOS token '<end_of_turn>' not found in vocabulary"))?;

        let mut sampled = 0;
        for index in 0..max_length {
            if next_token == eos_token_id {
                break;
            }

            let context_size = prompt_len + index;
            let input = Tensor::new(&[next_token], &self.config.device)?.unsqueeze(0)?;
            let logits = model_weights.forward(&input, context_size)?;
            let logits = logits.squeeze(0)?;

            let start_at = all_generated_tokens
                .len()
                .saturating_sub(self.config.params.repeat_last_n);
            let penalty_context = &all_generated_tokens[start_at..];

            let logits = if self.config.params.repeat_penalty <= 1. || penalty_context.is_empty() {
                logits
            } else {
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    self.config.params.repeat_penalty,
                    penalty_context,
                )?
            };

            next_token = logits_processor.sample(&logits)?;

            if next_token == eos_token_id {
                break;
            }
            all_generated_tokens.push(next_token);
            sampled += 1;
        }

        let generated_text = tokenizer
            .decode(&all_generated_tokens, true)
            .map_err(|e| anyhow::anyhow!("Failed to decode generated tokens: {}", e))?;

        Ok(generated_text)
    }
}

pub fn print_stats(
    prompt_dt: std::time::Duration,
    dt: std::time::Duration,
    prompt_tokens_len: usize,
    sampled: usize,
) {
    println!(
        "\n\nProcessed {} prompt tokens in {:.2}s ({:.2} token/s)",
        prompt_tokens_len,
        prompt_dt.as_secs_f64(),
        prompt_tokens_len as f64 / prompt_dt.as_secs_f64(),
    );
    println!(
        "Generated {sampled} tokens in {:.2}s ({:.2} token/s)",
        dt.as_secs_f64(),
        sampled as f64 / dt.as_secs_f64(),
    );
}
