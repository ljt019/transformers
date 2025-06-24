use super::text_generation_model::LanguageModelContext;
use super::text_generation_model::TextGenerationModel;
use super::text_generation_model::{ToggleableReasoning, ToolCalling};
use crate::models::generation::{
    apply_repeat_penalty, initialize_logits_processor, GenerationParams,
};
use crate::pipelines::utils::load_device;
use candle_core::Tensor;
use tokenizers::Tokenizer;

pub struct TextGenerationPipeline<M: TextGenerationModel> {
    model: M,
    model_tokenizer: Tokenizer,
    context: M::Context,
    gen_params: GenerationParams,
    device: candle_core::Device,
}

impl<M: TextGenerationModel> TextGenerationPipeline<M> {
    pub fn new(model: M, gen_params: GenerationParams) -> anyhow::Result<Self> {
        let model_tokenizer = model.get_tokenizer()?;
        let context = model.new_context();
        let device = load_device()?;

        Ok(Self {
            model,
            model_tokenizer,
            context,
            gen_params,
            device,
        })
    }
}

impl<M: TextGenerationModel> TextGenerationPipeline<M> {
    pub fn prompt_completion(&mut self, prompt: &str) -> anyhow::Result<String>
    where
        M::Context: LanguageModelContext,
    {
        let templated_prompt = self
            .model
            .apply_chat_template(&[crate::Message::user(prompt)])?;

        let prompt_tokens = self
            .model_tokenizer
            .encode(templated_prompt, true)
            .expect("Failed to encode prompt");

        let mut logits_processor =
            initialize_logits_processor(&self.gen_params, self.gen_params.seed);

        let mut generated_tokens: Vec<u32> = Vec::with_capacity(self.gen_params.max_len);

        // 1 x L (batch and seq_len)
        let input = Tensor::new(prompt_tokens.get_ids(), &self.device)?.unsqueeze(0)?;

        // 1 x 1 x V (batch, seq_len, vocab_size)
        let logits = self.context.generate(&input)?;

        // 1 x V (seq_len, vocab_size)
        let logits = logits.squeeze(0)?;

        // 1 (seq_len)
        let mut next_token = logits_processor.sample(&logits)?;
        generated_tokens.push(next_token);

        for _ in 0..self.gen_params.max_len {
            if next_token == self.model.get_eos_token() {
                break;
            }

            let input = Tensor::new(&[next_token], &self.device)?.unsqueeze(0)?;
            let logits = self.context.generate(&input)?;
            let logits = logits.squeeze(0)?;

            let start_at = generated_tokens
                .len()
                .saturating_sub(self.gen_params.repeat_last_n);
            let penalty_context = &generated_tokens[start_at..];

            let logits = if self.gen_params.repeat_penalty <= 1. || penalty_context.is_empty() {
                logits
            } else {
                apply_repeat_penalty(&logits, self.gen_params.repeat_penalty, penalty_context)?
            };

            next_token = logits_processor.sample(&logits)?;
            generated_tokens.push(next_token);
        }

        let generated_tokens_str = generated_tokens
            .iter()
            .map(|t| self.model_tokenizer.decode(&[*t], true).unwrap())
            .collect::<Vec<String>>()
            .join("");

        Ok(generated_tokens_str)
    }
}

impl<M: TextGenerationModel + ToggleableReasoning> TextGenerationPipeline<M> {
    pub fn set_reasoning(&mut self, enable: bool) -> anyhow::Result<()> {
        self.model.set_reasoning(enable)
    }
}

impl<M: TextGenerationModel + ToolCalling> TextGenerationPipeline<M> {
    pub fn register_tool(&mut self, tool: String) {
        self.model.register_tool(tool)
    }

    pub fn toggle_tools(&mut self, enable: bool) {
        self.model.toggle_tools(enable)
    }
}
