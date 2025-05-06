use crate::models::raw::generation::{LogitsProcessor, Sampling};
use crate::models::raw::models::quantized_gemma3;
use crate::models::raw::models::quantized_phi3;
use crate::models::raw::models::quantized_qwen3;
use candle_core::{CudaDevice, Device, Tensor};
use std::cell::RefCell;
use tokenizers::Tokenizer;

pub mod configs;
pub mod loaders;

/// Loads a device to be used for the model. Uses CUDA by default, falling back to CPU if CUDA is not available.
pub fn load_device() -> anyhow::Result<Device> {
    match CudaDevice::new_with_stream(0) {
        Ok(cuda_device) => Ok(Device::Cuda(cuda_device)),
        Err(_) => Ok(Device::Cpu),
    }
}

/// Generic function to generate text using a quantized model.
pub fn generate_quantized_text<M: QuantizedModelWeights>(
    model_weights: &mut M,
    device: &Device,
    tokenizer: &Tokenizer,
    prompt_tokens: &[u32],
    eos_token_str: &str,
    params: &GenerationParams,
    max_length: usize,
) -> anyhow::Result<String> {
    let prompt_len = prompt_tokens.len();
    let mut logits_processor = initialize_logits_processor(params, params.seed);
    let mut all_generated_tokens: Vec<u32> = Vec::with_capacity(max_length);

    let input = Tensor::new(prompt_tokens, device)?.unsqueeze(0)?;
    let logits = model_weights.forward(&input, 0)?;
    let logits = logits.squeeze(0)?;
    let mut next_token = logits_processor.sample(&logits)?;
    all_generated_tokens.push(next_token);

    let eos_token_id = *tokenizer
        .get_vocab(true)
        .get(eos_token_str)
        .ok_or_else(|| anyhow::anyhow!("EOS token '{}' not found in vocabulary", eos_token_str))?;

    for index in 0..max_length {
        if next_token == eos_token_id {
            break;
        }

        let context_size = prompt_len + index;
        let input = Tensor::new(&[next_token], device)?.unsqueeze(0)?;
        let logits = model_weights.forward(&input, context_size)?;
        let logits = logits.squeeze(0)?;

        let start_at = all_generated_tokens
            .len()
            .saturating_sub(params.repeat_last_n);
        let penalty_context = &all_generated_tokens[start_at..];

        let logits = if params.repeat_penalty <= 1. || penalty_context.is_empty() {
            logits
        } else {
            apply_repeat_penalty(&logits, params.repeat_penalty, penalty_context)?
        };

        next_token = logits_processor.sample(&logits)?;

        if next_token == eos_token_id {
            break;
        }
        all_generated_tokens.push(next_token);
    }

    let generated_text = tokenizer
        .decode(&all_generated_tokens, true)
        .map_err(|e| anyhow::anyhow!("Failed to decode generated tokens: {}", e))?;

    Ok(generated_text)
}
