use super::{
    tools::{Tool, ToolCallResult},
    traits::{TextGenerationPipeline, TextGenerationPipelineWithTools},
    TextGenerationModel,
};
use crate::Message;
use tokenizers::Tokenizer;

/// A text generation pipeline with tool calling capabilities.
/// Used for models like Gemma3 that can register and use tools during generation.
pub struct ToolCallingPipeline {
    model: Box<dyn TextGenerationModel>,
    tokenizer: Tokenizer,
    eos_token_id: u32,
    tools: Vec<Tool>,
}

impl ToolCallingPipeline {
    pub fn new(
        model: Box<dyn TextGenerationModel>,
        tokenizer: Tokenizer,
        eos_token_id: u32,
    ) -> Self {
        Self {
            model,
            tokenizer,
            eos_token_id,
            tools: Vec::new(),
        }
    }
}

impl TextGenerationPipeline for ToolCallingPipeline {
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

impl TextGenerationPipelineWithTools for ToolCallingPipeline {
    fn register_tool(&mut self, tool: Tool) -> anyhow::Result<()> {
        todo!("Tool registration not yet implemented")
    }

    fn unregister_tool(&mut self, tool_name: &str) -> anyhow::Result<()> {
        todo!("Tool unregistration not yet implemented")
    }

    fn list_tools(&self) -> Vec<&Tool> {
        todo!("Tool listing not yet implemented")
    }

    fn call_with_tools(&self, prompt: &str, max_length: usize) -> anyhow::Result<ToolCallResult> {
        todo!("Tool calling not yet implemented")
    }

    fn call_with_tools_messages(
        &self,
        messages: Vec<Message>,
        max_length: usize,
    ) -> anyhow::Result<ToolCallResult> {
        todo!("Tool calling with messages not yet implemented")
    }
}
