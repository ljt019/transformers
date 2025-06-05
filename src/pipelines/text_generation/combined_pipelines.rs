use super::{
    tools::{Tool, ToolCallResult},
    traits::{
        TextGenerationPipeline, TextGenerationPipelineWithToggleableReasoning,
        TextGenerationPipelineWithTools,
    },
};
use crate::pipelines::TextGenerationModel;
use crate::Message;
use tokenizers::Tokenizer;

/// A text generation pipeline with both toggleable reasoning and tool calling capabilities.
/// Used for models like Qwen3 that support both reasoning control and tool use.
pub struct ToggleableReasoningToolsPipeline {
    model: Box<dyn TextGenerationModel>,
    tokenizer: Tokenizer,
    eos_token_id: u32,
    reasoning_enabled: bool,
    tools: Vec<Tool>,
    reasoning_trace: Option<String>,
}

impl ToggleableReasoningToolsPipeline {
    pub fn new(
        model: Box<dyn TextGenerationModel>,
        tokenizer: Tokenizer,
        eos_token_id: u32,
    ) -> Self {
        Self {
            model,
            tokenizer,
            eos_token_id,
            reasoning_enabled: false, // Default to disabled
            tools: Vec::new(),
            reasoning_trace: None,
        }
    }
}

impl TextGenerationPipeline for ToggleableReasoningToolsPipeline {
    fn prompt_completion(&self, prompt: &str, max_length: usize) -> anyhow::Result<String> {
        // Use reasoning-aware prompt formatting if reasoning is enabled
        let formatted_prompt = if self.reasoning_enabled {
            self.model.format_prompt_with_reasoning(prompt, true)
        } else {
            self.model.format_prompt(prompt)
        };

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

impl TextGenerationPipelineWithToggleableReasoning for ToggleableReasoningToolsPipeline {
    fn enable_reasoning(&mut self) {
        todo!("Reasoning enabling not yet implemented")
    }

    fn disable_reasoning(&mut self) {
        todo!("Reasoning disabling not yet implemented")
    }

    fn is_reasoning_enabled(&self) -> bool {
        todo!("Reasoning status check not yet implemented")
    }

    fn get_reasoning_trace(&self) -> Option<&str> {
        todo!("Reasoning trace access not yet implemented")
    }
}

impl TextGenerationPipelineWithTools for ToggleableReasoningToolsPipeline {
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
