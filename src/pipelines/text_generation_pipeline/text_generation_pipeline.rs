use super::text_generation_model::LanguageModelContext;
use super::text_generation_model::TextGenerationModel;
use super::text_generation_model::{IntoTool, ToggleableReasoning, Tool, ToolCalling};
use crate::models::generation::{
    apply_repeat_penalty, initialize_logits_processor, GenerationParams,
};
use crate::pipelines::utils::load_device;
use candle_core::Tensor;
use regex::Regex;
use serde::Deserialize;
use tokenizers::Tokenizer;

pub struct TextGenerationPipeline<M: TextGenerationModel> {
    model: M,
    model_tokenizer: Tokenizer,
    context: M::Context,
    gen_params: GenerationParams,
    device: candle_core::Device,
    last_processed_tokens: Vec<u32>,
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
            last_processed_tokens: Vec::new(),
        })
    }

    /// Get the current position in the context (number of cached tokens)
    pub fn context_position(&self) -> usize {
        self.context.position()
    }

    pub fn prompt_completion(&mut self, prompt: &str) -> anyhow::Result<String> {
        // Reset context for fresh generation
        self.context.reset();

        let templated_prompt = self
            .model
            .apply_chat_template(&[crate::Message::user(prompt)])?;

        let prompt_tokens = self
            .model_tokenizer
            .encode(templated_prompt, true)
            .expect("Failed to encode prompt");

        self.completion(&prompt_tokens.get_ids())
    }

    pub fn message_completion(&mut self, messages: &[crate::Message]) -> anyhow::Result<String> {
        let templated_prompt = self.model.apply_chat_template(messages)?;

        let new_tokens = self
            .model_tokenizer
            .encode(templated_prompt, true)
            .expect("Failed to encode prompt")
            .get_ids()
            .to_vec();

        // Debug logging
        eprintln!("Debug: new_tokens.len() = {}", new_tokens.len());
        eprintln!(
            "Debug: last_processed_tokens.len() = {}",
            self.last_processed_tokens.len()
        );
        eprintln!("Debug: context.position() = {}", self.context.position());

        // Check if we need to reset due to context overflow
        let max_seq_len = self.model.get_max_seq_len();
        let pending_tokens = new_tokens.len();

        if self.context.position() + pending_tokens > max_seq_len {
            // Context would overflow, reset and start fresh
            eprintln!("Debug: Resetting due to context overflow");
            self.context.reset();
            self.last_processed_tokens.clear();
        } else if self.can_reuse_cache(&new_tokens) {
            // Cache prefix matches, only feed the suffix
            let prefix_len = self.last_processed_tokens.len();
            let new_portion = &new_tokens[prefix_len..];
            let response = self.completion(new_portion)?;

            // Track only prompt tokens for next turn
            self.last_processed_tokens = new_tokens;
            return Ok(response);
        } else {
            // Cache is invalid (conversation changed), reset
            eprintln!("Debug: Cache invalid, resetting");
            eprintln!(
                "Debug: starts_with = {}",
                new_tokens.starts_with(&self.last_processed_tokens)
            );
            self.context.reset();
        }

        // Process all tokens from scratch
        eprintln!(
            "Debug: Processing all {} tokens from scratch",
            new_tokens.len()
        );
        let response = self.completion(&new_tokens)?;

        // Update tracking (prompt tokens only)
        self.last_processed_tokens = new_tokens;

        Ok(response)
    }

    fn can_reuse_cache(&self, new_tokens: &[u32]) -> bool {
        // Cache can be reused if the new prompt begins with the exact token
        // sequence that is already cached.
        new_tokens.starts_with(&self.last_processed_tokens)
    }

    fn completion(&mut self, input_tokens: &[u32]) -> anyhow::Result<String> {
        const CHUNK_SIZE: usize = 64; // Must be <= initial kv cache size

        let mut logits_processor =
            initialize_logits_processor(&self.gen_params, self.gen_params.seed);

        let mut generated_tokens: Vec<u32> = Vec::with_capacity(self.gen_params.max_len);

        // Feed the initial prompt in manageable chunks to allow the KV cache to grow.
        let mut idx = 0;
        let mut last_logits = None;
        while idx < input_tokens.len() {
            let end = usize::min(idx + CHUNK_SIZE, input_tokens.len());
            let chunk = &input_tokens[idx..end];

            let input = Tensor::new(chunk, &self.device)?.unsqueeze(0)?;
            let logits = self.context.generate(&input)?;
            last_logits = Some(logits.squeeze(0)?);
            idx = end;
        }

        // Safety: there is always at least one chunk, so last_logits is Some
        let mut next_token = logits_processor.sample(&last_logits.unwrap())?;
        generated_tokens.push(next_token);

        // Generate autoregressively
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
    pub fn register_tool<T: IntoTool>(&mut self, tool: T) -> anyhow::Result<()> {
        self.model.register_tool(tool.into_tool())
    }

    /// Register multiple tools at once.
    pub fn register_tools(&mut self, tools: Vec<Tool>) -> anyhow::Result<()> {
        for tool in tools {
            self.model.register_tool(tool)?;
        }
        Ok(())
    }

    pub fn registered_tools(&self) -> Vec<Tool> {
        self.model.registered_tools()
    }

    /// Same as [`prompt_completion`], but automatically handles tool calls emitted by the
    /// model. The function will repeatedly generate until the final assistant response no
    /// longer contains a `<tool_call>` block, at which point that response is returned.
    pub fn prompt_completion_with_tools(&mut self, prompt: &str) -> anyhow::Result<String> {
        // Accumulated chat history
        let mut messages = vec![crate::Message::user(prompt)];

        // Safety limit to avoid infinite loops in case of model issues
        const MAX_TOOL_ITERATIONS: usize = 8;

        for _ in 0..MAX_TOOL_ITERATIONS {
            // Generate assistant response for current state
            let assistant_response = if messages.len() == 1 {
                // First turn – no previous context to reuse
                self.prompt_completion(prompt)?
            } else {
                self.message_completion(&messages)?
            };

            // Check for tool calls
            let tool_calls = Self::extract_tool_calls(&assistant_response)?;

            // If no tool calls, return the assistant response as final answer
            if tool_calls.is_empty() {
                return Ok(assistant_response.trim().to_string());
            }

            // Record the assistant message that issued the tool call
            eprintln!(
                "Debug: Assistant Tool Call Response response: {}",
                assistant_response
            );
            messages.push(crate::Message::assistant(&assistant_response));

            // Execute each tool call and append the tool response messages
            for tc in tool_calls {
                let result = self
                    .model
                    .call_tool(tc.name.clone(), tc.arguments.clone())?;

                messages.push(crate::Message {
                    role: "tool".to_string(),
                    content: result,
                });
            }
        }

        Err(anyhow::anyhow!(
            "Maximum number of tool iterations reached without final response"
        ))
    }

    /// Same as [`message_completion`], but automatically handles tool calls emitted by the
    /// model. The function will repeatedly generate until the final assistant response no
    /// longer contains a `<tool_call>` block, at which point that response is returned.
    pub fn message_completion_with_tools(
        &mut self,
        messages: &[crate::Message],
    ) -> anyhow::Result<String> {
        // Accumulated chat history
        let mut messages = messages.to_vec();

        // Safety limit to avoid infinite loops in case of model issues
        const MAX_TOOL_ITERATIONS: usize = 8;

        for _ in 0..MAX_TOOL_ITERATIONS {
            // Generate assistant response for current state
            let assistant_response = self.message_completion(&messages)?;

            // Check for tool calls
            let tool_calls = Self::extract_tool_calls(&assistant_response)?;

            // If no tool calls, return the assistant response as final answer
            if tool_calls.is_empty() {
                return Ok(assistant_response.trim().to_string());
            }

            // Record the assistant message that issued the tool call
            eprintln!(
                "Debug: Assistant Tool Call Response response: {}",
                assistant_response
            );
            messages.push(crate::Message::assistant(&assistant_response));

            // Execute each tool call and append the tool response messages
            for tc in tool_calls {
                let result = self
                    .model
                    .call_tool(tc.name.clone(), tc.arguments.clone())?;

                messages.push(crate::Message {
                    role: "tool".to_string(),
                    content: result,
                });
            }
        }

        Err(anyhow::anyhow!(
            "Maximum number of tool iterations reached without final response"
        ))
    }

    /// Extract tool calls from the text returned by the model.
    fn extract_tool_calls(text: &str) -> anyhow::Result<Vec<ToolCallInvocation>> {
        // Precompiled regex to capture everything inside <tool_call>...</tool_call>
        // (?s) enables dot to match newlines.
        static TOOL_CALL_RE: once_cell::sync::Lazy<Regex> = once_cell::sync::Lazy::new(|| {
            Regex::new(r"(?s)<tool_call>\s*(.*?)\s*</tool_call>").unwrap()
        });

        let mut calls = Vec::new();
        for cap in TOOL_CALL_RE.captures_iter(text) {
            let inner = cap.get(1).unwrap().as_str().trim();

            // The inner text is expected to be JSON.
            let parsed: RawToolCall = serde_json::from_str(inner)
                .map_err(|e| anyhow::anyhow!("Failed to parse tool call JSON: {e}: {inner}"))?;

            calls.push(ToolCallInvocation {
                name: parsed.name,
                arguments: parsed.arguments.unwrap_or_default(),
            });
        }

        Ok(calls)
    }
}

#[derive(Deserialize)]
struct RawToolCall {
    name: String,
    #[serde(default)]
    arguments: Option<std::collections::HashMap<String, String>>,
}

#[derive(Clone)]
struct ToolCallInvocation {
    name: String,
    arguments: std::collections::HashMap<String, String>,
}
