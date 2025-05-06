pub mod raw;

pub mod modern_bert;
pub mod sentiment_modern_bert;
pub mod zero_shot_modern_bert;

pub mod gemma_3;
pub mod phi_4;
pub mod qwen_3;

pub use raw::generation::{apply_repeat_penalty, initialize_logits_processor};

trait ModelWeights {
    fn forward(
        &self,
        xs: &candle_core::Tensor,
        start_pos: usize,
    ) -> candle_core::Result<candle_core::Tensor>;
}

use candle_core::{Device, Tensor};

fn generate_tokens_from_prompt<M: ModelWeights>(
    prompt_tokens: &[u32],
    params: &raw::generation::GenerationParams,
    model_weights: &mut M,
    max_len: usize,
    device: &Device,
    eos_token_id: u32,
) -> anyhow::Result<Vec<u32>> {
    let prompt_len = prompt_tokens.len();

    let mut logits_processor = initialize_logits_processor(params, params.seed);
    let mut all_generated_tokens: Vec<u32> = Vec::with_capacity(max_len);

    // 1 x L (batch and seq_len)
    let input = Tensor::new(prompt_tokens, device)?.unsqueeze(0)?;

    // 1 x 1 x V (batch, seq_len, vocab_size)
    let logits = model_weights.forward(&input, 0)?;

    // 1 x V (seq_len, vocab_size)
    let logits = logits.squeeze(0)?;

    // 1 (seq_len)
    let mut next_token = logits_processor.sample(&logits)?;
    all_generated_tokens.push(next_token);

    for index in 0..max_len {
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
        all_generated_tokens.push(next_token);
    }

    Ok(all_generated_tokens)
}
