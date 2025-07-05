use super::text_generation_model::LanguageModelContext;
use super::text_generation_model::TextGenerationModel;
use crate::models::generation::{
    apply_repeat_penalty, initialize_logits_processor, GenerationParams,
};
use crate::pipelines::utils::load_device;
use candle_core::Tensor;
use std::sync::{Arc, Mutex};
use tokenizers::Tokenizer;

/// Base structure containing common fields for both pipeline types
pub struct BasePipeline<M: TextGenerationModel> {
    pub model: Arc<Mutex<M>>,
    pub model_tokenizer: Tokenizer,
    pub context: Arc<Mutex<M::Context>>,
    pub gen_params: Arc<Mutex<GenerationParams>>,
    pub device: candle_core::Device,
    pub last_processed_tokens: Arc<Mutex<Vec<u32>>>,
    pub special_strings: std::collections::HashSet<String>,
}

impl<M: TextGenerationModel> BasePipeline<M> {
    pub fn new(model: M, gen_params: GenerationParams) -> anyhow::Result<Self> {
        let model_tokenizer = model.get_tokenizer()?;
        let context = model.new_context();
        let device = load_device()?;

        // Collect textual forms of special tokens for display filtering
        let mut special_strings: std::collections::HashSet<String> = model_tokenizer
            .get_added_tokens_decoder()
            .values()
            .filter(|tok| tok.special)
            .map(|tok| tok.content.clone())
            .collect();

        // Add standard special tokens that might not be in the decoder
        special_strings.insert("<|im_start|>".to_string());
        special_strings.insert("<|im_end|>".to_string());
        special_strings.insert("<|im_sep|>".to_string());

        Ok(Self {
            model: Arc::new(Mutex::new(model)),
            model_tokenizer,
            context: Arc::new(Mutex::new(context)),
            gen_params: Arc::new(Mutex::new(gen_params)),
            device,
            last_processed_tokens: Arc::new(Mutex::new(Vec::new())),
            special_strings,
        })
    }

    /// Get the current position in the context (number of cached tokens)
    pub fn context_position(&self) -> usize {
        self.context.lock().unwrap().position()
    }

    pub fn set_generation_params(&self, params: GenerationParams) {
        *self.gen_params.lock().unwrap() = params;
    }

    pub fn can_reuse_cache(&self, new_tokens: &[u32]) -> bool {
        // Cache can be reused if the new prompt begins with the exact token
        // sequence that is already cached.
        new_tokens.starts_with(&self.last_processed_tokens.lock().unwrap())
    }

    pub fn completion_from_tokens(&self, input_tokens: &[u32]) -> anyhow::Result<String> {
        const CHUNK_SIZE: usize = 64; // Must be <= initial kv cache size

        let params = self.gen_params.lock().unwrap().clone();

        let mut logits_processor = initialize_logits_processor(&params, params.seed);

        let mut generated_tokens: Vec<u32> = Vec::with_capacity(params.max_len);

        // Feed the initial prompt in manageable chunks to allow the KV cache to grow.
        let mut idx = 0;
        let mut last_logits = None;
        while idx < input_tokens.len() {
            let end = usize::min(idx + CHUNK_SIZE, input_tokens.len());
            let chunk = &input_tokens[idx..end];

            let input = Tensor::new(chunk, &self.device)?.unsqueeze(0)?;
            let logits = {
                let mut ctx = self.context.lock().unwrap();
                ctx.generate(&input)
            }?;
            last_logits = Some(logits.squeeze(0)?);
            idx = end;
        }

        // Safety: there is always at least one chunk, so last_logits is Some
        let mut next_token = logits_processor.sample(&last_logits.unwrap())?;
        generated_tokens.push(next_token);

        // Generate autoregressively
        let eos_tokens = self.model.lock().unwrap().get_eos_tokens();
        for _ in 0..params.max_len {
            if eos_tokens.contains(&next_token) {
                break;
            }

            let input = Tensor::new(&[next_token], &self.device)?.unsqueeze(0)?;
            let logits = {
                let mut ctx = self.context.lock().unwrap();
                ctx.generate(&input)
            }?;
            let logits = logits.squeeze(0)?;

            let start_at = generated_tokens.len().saturating_sub(params.repeat_last_n);
            let penalty_context = &generated_tokens[start_at..];

            let logits = if params.repeat_penalty <= 1. || penalty_context.is_empty() {
                logits
            } else {
                apply_repeat_penalty(&logits, params.repeat_penalty, penalty_context)?
            };

            next_token = logits_processor.sample(&logits)?;
            generated_tokens.push(next_token);
        }

        // Filter out EOS tokens before decoding
        let eos_tokens = self.model.lock().unwrap().get_eos_tokens();
        let filtered_tokens: Vec<u32> = generated_tokens
            .into_iter()
            .filter(|&token| !eos_tokens.contains(&token))
            .collect();

        // Fast multi-token decode in a single call instead of per-token concat
        let generated_tokens_str = self
            .model_tokenizer
            .decode(&filtered_tokens, /*skip_special_tokens=*/ true)
            .unwrap();

        Ok(generated_tokens_str)
    }
}
