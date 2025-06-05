use super::{traits::TextGenerationPipeline, TextGenerationModel};
use crate::Message;
use tokenizers::Tokenizer;

/// A basic text generation pipeline with no special capabilities.
/// Used for models like Phi4 that only provide standard text generation.
pub struct BasicPipeline {
    model: Box<dyn TextGenerationModel>,
    tokenizer: Tokenizer,
    eos_token_id: u32,
}

impl BasicPipeline {
    pub fn new(
        model: Box<dyn TextGenerationModel>,
        tokenizer: Tokenizer,
        eos_token_id: u32,
    ) -> Self {
        Self {
            model,
            tokenizer,
            eos_token_id,
        }
    }
}

impl TextGenerationPipeline for BasicPipeline {
    fn prompt_completion(&self, prompt: &str, max_length: usize) -> anyhow::Result<String> {
        // Format the prompt
        let formatted_prompt = self.model.format_prompt(prompt);

        // Turn the prompt into tokens
        let prompt_tokens = self
            .tokenizer
            .encode(formatted_prompt, true)
            .map_err(|e| anyhow::anyhow!("Failed to encode prompt: {}", e))?;

        // Generate the response with the prompt tokens
        let response_as_tokens = self.model.prompt_with_tokens(
            prompt_tokens.get_ids(),
            max_length,
            self.eos_token_id,
        )?;

        // Turn the response tokens back into a string
        let response = self
            .tokenizer
            .decode(&response_as_tokens, true)
            .map_err(|e| anyhow::anyhow!("Failed to decode response tokens: {}", e))?;

        Ok(response)
    }

    fn message_completion(
        &self,
        messages: Vec<Message>,
        max_length: usize,
    ) -> anyhow::Result<String> {
        // Format the messages
        let formatted_messages = self.model.format_messages(messages)?;

        // Turn the prompt into tokens
        let prompt_tokens = self
            .tokenizer
            .encode(formatted_messages, true)
            .map_err(|e| anyhow::anyhow!("Failed to encode formatted messages: {}", e))?;

        // Generate the response with the prompt tokens
        let response_as_tokens = self.model.prompt_with_tokens(
            prompt_tokens.get_ids(),
            max_length,
            self.eos_token_id,
        )?;

        // Turn the response tokens back into a string
        let response = self
            .tokenizer
            .decode(&response_as_tokens, true)
            .map_err(|e| anyhow::anyhow!("Failed to decode response tokens: {}", e))?;

        Ok(response)
    }
}
