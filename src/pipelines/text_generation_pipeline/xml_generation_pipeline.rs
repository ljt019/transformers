use super::base_pipeline::BasePipeline;
use super::text_generation_model::{IntoTool, ToggleableReasoning, Tool, ToolCalling, LanguageModelContext};
use super::text_generation_model::TextGenerationModel;
use super::text_generation_pipeline::Input;
use super::xml_parser::{Event, XmlParser};
use crate::models::generation::{initialize_logits_processor, apply_repeat_penalty, GenerationParams};
use candle_core::Tensor;
use std::sync::{Arc, Mutex};
use async_stream::try_stream;
use futures::Stream;
use std::pin::Pin;
use regex::Regex;
use serde::Deserialize;

/// XML generation pipeline that outputs parsed Events
pub struct XmlGenerationPipeline<M: TextGenerationModel> {
    base: BasePipeline<M>,
    xml_parser: XmlParser,
}

impl<M: TextGenerationModel> XmlGenerationPipeline<M> {
    pub fn new(model: M, gen_params: GenerationParams, xml_parser: XmlParser) -> anyhow::Result<Self> {
        Ok(Self {
            base: BasePipeline::new(model, gen_params)?,
            xml_parser,
        })
    }

    /// Get the current position in the context (number of cached tokens)
    pub fn context_position(&self) -> usize {
        self.base.context_position()
    }

    /// Get a reference to the XML parser
    pub fn xml_parser(&self) -> &XmlParser {
        &self.xml_parser
    }

    /// Generate a completion from either a prompt or a chat history.
    /// Returns a Vec<Event>.
    pub fn completion<'a>(&self, input: impl Into<Input<'a>>) -> anyhow::Result<Vec<Event>> {
        let text = match input.into() {
            Input::Prompt(p) => self.prompt_completion_internal(p)?,
            Input::Messages(m) => self.message_completion_internal(m)?,
        };

        Ok(self.xml_parser.parse_complete(&text))
    }

    fn prompt_completion_internal(&self, prompt: &str) -> anyhow::Result<String> {
        // Reset context for fresh generation
        self.base.context.lock().unwrap().reset();

        let templated_prompt = self
            .base.model
            .lock()
            .unwrap()
            .apply_chat_template(&[crate::Message::user(prompt)])?;

        let prompt_tokens = self
            .base.model_tokenizer
            .encode(templated_prompt, true)
            .map_err(|e| anyhow::anyhow!(e))?
            .get_ids()
            .to_vec();

        self.base.completion_from_tokens(&prompt_tokens)
    }

    fn message_completion_internal(&self, messages: &[crate::Message]) -> anyhow::Result<String> {
        let templated_prompt = self.base.model.lock().unwrap().apply_chat_template(messages)?;

        let new_tokens = self
            .base.model_tokenizer
            .encode(templated_prompt, true)
            .map_err(|e| anyhow::anyhow!(e))?
            .get_ids()
            .to_vec();

        // Check if we need to reset due to context overflow
        let max_seq_len = self.base.model.lock().unwrap().get_max_seq_len();
        let pending_tokens = new_tokens.len();

        if self.base.context.lock().unwrap().position() + pending_tokens > max_seq_len {
            // Context would overflow, reset and start fresh
            self.base.context.lock().unwrap().reset();
            self.base.last_processed_tokens.lock().unwrap().clear();
        } else if self.base.can_reuse_cache(&new_tokens) {
            // Cache prefix matches, only feed the suffix
            let prefix_len = self.base.last_processed_tokens.lock().unwrap().len();
            let new_portion = &new_tokens[prefix_len..];
            let response = self.base.completion_from_tokens(new_portion)?;

            // Track only prompt tokens for next turn
            *self.base.last_processed_tokens.lock().unwrap() = new_tokens;
            return Ok(response);
        } else {
            // Cache is invalid (conversation changed), reset
            self.base.context.lock().unwrap().reset();
        }

        // Process all tokens from scratch
        let response = self.base.completion_from_tokens(&new_tokens)?;

        // Update tracking (prompt tokens only)
        *self.base.last_processed_tokens.lock().unwrap() = new_tokens;

        Ok(response)
    }

    /// Streaming version of completion that yields Events
    pub fn completion_stream<'a>(
        &'a self,
        input: impl Into<Input<'a>>,
    ) -> anyhow::Result<Pin<Box<dyn Stream<Item = Event> + Send + 'a>>> {
        match input.into() {
            Input::Prompt(p) => self.prompt_completion_stream(p),
            Input::Messages(m) => self.message_completion_stream(m),
        }
    }

    fn prompt_completion_stream(
        &self,
        prompt: &str,
    ) -> anyhow::Result<Pin<Box<dyn Stream<Item = Event> + Send + '_>>> {
        // Fresh turn → reset context
        self.base.context.lock().unwrap().reset();

        let templated = self
            .base.model
            .lock()
            .unwrap()
            .apply_chat_template(&[crate::Message::user(prompt)])?;
        let tokens = self
            .base.model_tokenizer
            .encode(templated, true)
            .map_err(|e| anyhow::anyhow!(e))?
            .get_ids()
            .to_vec();

        use futures::StreamExt;
        let inner = self.raw_completion_stream(tokens);

        self.xml_parser.reset();
        let parser = self.xml_parser.clone();

        use async_stream::stream;
        Ok(Box::pin(stream! {
            futures::pin_mut!(inner);
            while let Some(result) = inner.next().await {
                let token = result.expect("stream generation failed");
                let events = parser.parse_token(&token);
                for event in events {
                    yield event;
                }
            }

            // Flush any remaining events
            let final_events = parser.flush();
            for event in final_events {
                yield event;
            }
        }))
    }

    fn message_completion_stream(
        &self,
        messages: &[crate::Message],
    ) -> anyhow::Result<Pin<Box<dyn Stream<Item = Event> + Send + '_>>> {
        let templated = self.base.model.lock().unwrap().apply_chat_template(messages)?;
        let new_tokens = self
            .base.model_tokenizer
            .encode(templated, true)
            .map_err(|e| anyhow::anyhow!(e))?
            .get_ids()
            .to_vec();

        // Same cache logic
        let max_seq = self.base.model.lock().unwrap().get_max_seq_len();
        if self.base.context.lock().unwrap().position() + new_tokens.len() > max_seq {
            self.base.context.lock().unwrap().reset();
            self.base.last_processed_tokens.lock().unwrap().clear();
        } else if self.base.can_reuse_cache(&new_tokens) {
            let suffix = new_tokens[self.base.last_processed_tokens.lock().unwrap().len()..].to_vec();
            *self.base.last_processed_tokens.lock().unwrap() = new_tokens;
            
            let inner = self.raw_completion_stream(suffix);
            self.xml_parser.reset();
            let parser = self.xml_parser.clone();

            use async_stream::stream;
            use futures::StreamExt;
            return Ok(Box::pin(stream! {
                futures::pin_mut!(inner);
                while let Some(result) = inner.next().await {
                    let token = result.expect("stream generation failed");
                    let events = parser.parse_token(&token);
                    for event in events {
                        yield event;
                    }
                }

                // Flush any remaining events
                let final_events = parser.flush();
                for event in final_events {
                    yield event;
                }
            }));
        } else {
            self.base.context.lock().unwrap().reset();
        }

        *self.base.last_processed_tokens.lock().unwrap() = new_tokens.clone();
        let inner = self.raw_completion_stream(new_tokens);
        
        self.xml_parser.reset();
        let parser = self.xml_parser.clone();

        use async_stream::stream;
        use futures::StreamExt;
        Ok(Box::pin(stream! {
            futures::pin_mut!(inner);
            while let Some(result) = inner.next().await {
                let token = result.expect("stream generation failed");
                let events = parser.parse_token(&token);
                for event in events {
                    yield event;
                }
            }

            // Flush any remaining events
            let final_events = parser.flush();
            for event in final_events {
                yield event;
            }
        }))
    }

    fn raw_completion_stream<'a>(&'a self, input_tokens: Vec<u32>) -> Pin<Box<dyn Stream<Item = anyhow::Result<String>> + Send + 'a>>
    where
        M: 'a,
    {
        // Capture everything the async generator needs **by value**
        let device = self.base.device.clone();
        let eos_tokens = self.base.model.lock().unwrap().get_eos_tokens();
        let tokenizer = self.base.model_tokenizer.clone();
        let context = Arc::clone(&self.base.context);
        let params = self.base.gen_params.clone();

        Box::pin(try_stream! {
            const CHUNK_SIZE: usize = 64;

            let mut logits_processor =
                initialize_logits_processor(&params, params.seed);

            // Send the whole prompt first
            let mut idx = 0;
            let mut last_logits = None;
            while idx < input_tokens.len() {
                let end   = usize::min(idx + CHUNK_SIZE, input_tokens.len());
                let chunk = &input_tokens[idx..end];

                let input  = Tensor::new(chunk, &device)?.unsqueeze(0)?;
                let logits = {
                    let mut ctx = context.lock().unwrap();
                    ctx.generate(&input)
                }?;
                last_logits = Some(logits.squeeze(0)?);
                idx = end;
            }

            // First sampled token
            let mut generated: Vec<u32> = Vec::with_capacity(params.max_len);

            // Incremental decoder that keeps special tokens
            let mut dec_full  = tokenizer.decode_stream(false);

            let mut next_token = logits_processor.sample(&last_logits.unwrap())?;
            generated.push(next_token);

            // Skip yielding if this token is an EOS token
            if !eos_tokens.contains(&next_token) {
                if let Some(chunk) = dec_full.step(next_token).map_err(|e| anyhow::anyhow!(e))? {
                    yield chunk;
                }
            } else {
                // Still need to step the decoder to keep state consistent, but don't yield
                let _ = dec_full.step(next_token).map_err(|e| anyhow::anyhow!(e))?;
            }

            // Autoregressive loop
            for _ in 0..params.max_len {
                if eos_tokens.contains(&next_token) {
                    break;
                }

                let input  = Tensor::new(&[next_token], &device)?.unsqueeze(0)?;
                let logits = {
                    let mut ctx = context.lock().unwrap();
                    ctx.generate(&input)
                }?;
                let logits = logits.squeeze(0)?;

                let start_at = generated.len().saturating_sub(params.repeat_last_n);
                let penalty_context = &generated[start_at..];

                let logits = if params.repeat_penalty <= 1. || penalty_context.is_empty() {
                    logits
                } else {
                    apply_repeat_penalty(&logits, params.repeat_penalty, penalty_context)?
                };

                next_token = logits_processor.sample(&logits)?;
                generated.push(next_token);

                // Skip yielding if this token is an EOS token
                if !eos_tokens.contains(&next_token) {
                    if let Some(chunk) = dec_full.step(next_token).map_err(|e| anyhow::anyhow!(e))? {
                        yield chunk;
                    }
                } else {
                    // Still need to step the decoder, but don't yield
                    let _ = dec_full.step(next_token).map_err(|e| anyhow::anyhow!(e))?;
                }
            }
        })
    }
}

// Implementations for models with ToggleableReasoning
impl<M: TextGenerationModel + ToggleableReasoning> XmlGenerationPipeline<M> {
    pub fn set_reasoning(&self, enable: bool) -> anyhow::Result<()> {
        self.base.model.lock().unwrap().set_reasoning(enable)
    }
}

// Implementations for models with ToolCalling
impl<M: TextGenerationModel + ToolCalling + Send> XmlGenerationPipeline<M> {
    pub fn register_tool<T: IntoTool>(&self, tool: T) -> anyhow::Result<()> {
        let tool = tool.into_tool();
        self.base.model.lock().unwrap().register_tool(tool)
    }

    pub fn register_tools(&self, tools: Vec<Tool>) -> anyhow::Result<()> {
        for tool in tools {
            self.base.model.lock().unwrap().register_tool(tool)?;
        }
        Ok(())
    }

    pub fn registered_tools(&self) -> Vec<Tool> {
        self.base.model.lock().unwrap().registered_tools()
    }

    pub fn completion_with_tools<'a>(&self, input: impl Into<Input<'a>>) -> anyhow::Result<Vec<Event>> {
        let text = match input.into() {
            Input::Prompt(p) => self.prompt_completion_with_tools(p)?,
            Input::Messages(m) => self.message_completion_with_tools(m)?,
        };

        Ok(self.xml_parser.parse_complete(&text))
    }

    pub fn completion_stream_with_tools<'a>(
        &'a self,
        input: impl Into<Input<'a>>,
    ) -> anyhow::Result<Pin<Box<dyn Stream<Item = Event> + Send + 'a>>> {
        match input.into() {
            Input::Prompt(p) => {
                let stream = self.prompt_completion_stream_with_tools(p)?;
                Ok(Box::pin(stream))
            }
            Input::Messages(m) => {
                let stream = self.message_completion_stream_with_tools(m)?;
                Ok(Box::pin(stream))
            }
        }
    }

    fn prompt_completion_with_tools(&self, prompt: &str) -> anyhow::Result<String> {
        // Reset for new generation
        self.base.context.lock().unwrap().reset();

        // Create messages with user prompt
        let mut messages = vec![crate::Message::user(prompt)];

        let tools = self.base.model.lock().unwrap().registered_tools();
        if tools.is_empty() {
            anyhow::bail!("No tools registered. Call register_tool() first.");
        }

        loop {
            // Generate response
            let templated = self.base.model.lock().unwrap().apply_chat_template(&messages)?;
            let tokens = self
                .base.model_tokenizer
                .encode(templated, true)
                .map_err(|e| anyhow::anyhow!(e))?
                .get_ids()
                .to_vec();

            let response = self.base.completion_from_tokens(&tokens)?;

            // Try to extract tool calls
            match Self::extract_tool_calls(&response) {
                Ok(tool_calls) if !tool_calls.is_empty() => {
                    // Add assistant message with tool calls
                    messages.push(crate::Message::assistant(&response));

                    // Execute tools and collect responses
                    let mut tool_responses = Vec::new();
                    for call in tool_calls {
                        // Find the tool
                        let tool = tools
                            .iter()
                            .find(|t| t.name == call.name)
                            .ok_or_else(|| anyhow::anyhow!("Tool '{}' not found", call.name))?;

                        // Execute the tool
                        match tool.execute(call.arguments) {
                            Ok(result) => {
                                tool_responses.push(format!(
                                    "<tool_response>\n{}: {}\n</tool_response>",
                                    call.name, result
                                ));
                            }
                            Err(e) => {
                                tool_responses.push(format!(
                                    "<tool_response>\n{}: Error: {}\n</tool_response>",
                                    call.name, e
                                ));
                            }
                        }
                    }

                    // Add tool response message
                    let tool_response_text = tool_responses.join("\n");
                    messages.push(crate::Message::user(&tool_response_text));

                    // Continue loop to generate final response
                }
                _ => {
                    // No tool calls, return the response
                    return Ok(response);
                }
            }
        }
    }

    fn message_completion_with_tools(
        &self,
        messages: &[crate::Message],
    ) -> anyhow::Result<String> {
        let tools = self.base.model.lock().unwrap().registered_tools();
        if tools.is_empty() {
            anyhow::bail!("No tools registered. Call register_tool() first.");
        }

        let mut messages = messages.to_vec();

        loop {
            // Generate response  
            let templated = self.base.model.lock().unwrap().apply_chat_template(&messages)?;
            let new_tokens = self
                .base.model_tokenizer
                .encode(templated, true)
                .map_err(|e| anyhow::anyhow!(e))?
                .get_ids()
                .to_vec();

            // Check if we need to reset due to context overflow
            let max_seq_len = self.base.model.lock().unwrap().get_max_seq_len();
            let pending_tokens = new_tokens.len();

            if self.base.context.lock().unwrap().position() + pending_tokens > max_seq_len {
                self.base.context.lock().unwrap().reset();
                self.base.last_processed_tokens.lock().unwrap().clear();
            } else if self.base.can_reuse_cache(&new_tokens) {
                let prefix_len = self.base.last_processed_tokens.lock().unwrap().len();
                let new_portion = &new_tokens[prefix_len..];
                let response = self.base.completion_from_tokens(new_portion)?;

                *self.base.last_processed_tokens.lock().unwrap() = new_tokens;

                // Check for tool calls
                match Self::extract_tool_calls(&response) {
                    Ok(tool_calls) if !tool_calls.is_empty() => {
                        messages.push(crate::Message::assistant(&response));

                        let mut tool_responses = Vec::new();
                        for call in tool_calls {
                            let tool = tools
                                .iter()
                                .find(|t| t.name == call.name)
                                .ok_or_else(|| anyhow::anyhow!("Tool '{}' not found", call.name))?;

                            match tool.execute(call.arguments) {
                                Ok(result) => {
                                    tool_responses.push(format!(
                                        "<tool_response>\n{}: {}\n</tool_response>",
                                        call.name, result
                                    ));
                                }
                                Err(e) => {
                                    tool_responses.push(format!(
                                        "<tool_response>\n{}: Error: {}\n</tool_response>",
                                        call.name, e
                                    ));
                                }
                            }
                        }

                        let tool_response_text = tool_responses.join("\n");
                        messages.push(crate::Message::user(&tool_response_text));
                        continue;
                    }
                    _ => return Ok(response),
                }
            } else {
                self.base.context.lock().unwrap().reset();
            }

            let response = self.base.completion_from_tokens(&new_tokens)?;
            *self.base.last_processed_tokens.lock().unwrap() = new_tokens;

            match Self::extract_tool_calls(&response) {
                Ok(tool_calls) if !tool_calls.is_empty() => {
                    messages.push(crate::Message::assistant(&response));

                    let mut tool_responses = Vec::new();
                    for call in tool_calls {
                        let tool = tools
                            .iter()
                            .find(|t| t.name == call.name)
                            .ok_or_else(|| anyhow::anyhow!("Tool '{}' not found", call.name))?;

                        match tool.execute(call.arguments) {
                            Ok(result) => {
                                tool_responses.push(format!(
                                    "<tool_response>\n{}: {}\n</tool_response>",
                                    call.name, result
                                ));
                            }
                            Err(e) => {
                                tool_responses.push(format!(
                                    "<tool_response>\n{}: Error: {}\n</tool_response>",
                                    call.name, e
                                ));
                            }
                        }
                    }

                    let tool_response_text = tool_responses.join("\n");
                    messages.push(crate::Message::user(&tool_response_text));
                }
                _ => return Ok(response),
            }
        }
    }

    fn prompt_completion_stream_with_tools(
        &self,
        prompt: &str,
    ) -> anyhow::Result<impl Stream<Item = Event> + Send + '_> {
        use async_stream::stream;
        use futures::StreamExt;

        let messages = vec![crate::Message::user(prompt)];
        Ok(stream! {
            let mut messages = messages;
            
            let stream = self.message_completion_stream_with_tools(&messages[..])?;
            futures::pin_mut!(stream);
            while let Some(event) = stream.next().await {
                yield event;
            }
        })
    }

    fn message_completion_stream_with_tools(
        &self,
        messages: &[crate::Message],
    ) -> anyhow::Result<impl Stream<Item = Event> + Send + '_> {
        use async_stream::stream;
        use futures::StreamExt;

        let tools = self.base.model.lock().unwrap().registered_tools();
        if tools.is_empty() {
            anyhow::bail!("No tools registered. Call register_tool() first.");
        }

        let initial_messages = messages.to_vec();
        let parser = self.xml_parser.clone();
        
        Ok(stream! {
            let mut messages = initial_messages;
            let mut response_buffer = String::new();

            loop {
                // Stream the response
                let stream = self.completion_stream(&messages[..])?;
                futures::pin_mut!(stream);

                // Collect full response while yielding events
                while let Some(event) = stream.next().await {
                    response_buffer.push_str(event.get_content());
                    yield event;
                }

                // Check for tool calls in the complete response
                match Self::extract_tool_calls(&response_buffer) {
                    Ok(tool_calls) if !tool_calls.is_empty() => {
                        // Add assistant message with tool calls
                        messages.push(crate::Message::assistant(&response_buffer));
                        response_buffer.clear();

                        // Execute tools
                        let mut tool_responses = Vec::new();
                        for call in tool_calls {
                            let tool = tools
                                .iter()
                                .find(|t| t.name == call.name);

                            match tool {
                                Some(tool) => {
                                    match tool.execute(call.arguments) {
                                        Ok(result) => {
                                            tool_responses.push(format!(
                                                "<tool_response>\n{}: {}\n</tool_response>",
                                                call.name, result
                                            ));
                                        }
                                        Err(e) => {
                                            tool_responses.push(format!(
                                                "<tool_response>\n{}: Error: {}\n</tool_response>",
                                                call.name, e
                                            ));
                                        }
                                    }
                                }
                                None => {
                                    tool_responses.push(format!(
                                        "<tool_response>\n{}: Error: Tool not found\n</tool_response>",
                                        call.name
                                    ));
                                }
                            }
                        }

                        let tool_response_text = tool_responses.join("\n");
                        messages.push(crate::Message::user(&tool_response_text));

                        // Continue to get the final response
                    }
                    _ => {
                        // No tool calls, we're done
                        break;
                    }
                }
            }
        })
    }

    fn extract_tool_calls(text: &str) -> anyhow::Result<Vec<ToolCallInvocation>> {
        let tool_regex = Regex::new(r"<tool_call>(.*?)</tool_call>")?;
        let mut tool_calls = Vec::new();

        for cap in tool_regex.captures_iter(text) {
            let json_str = cap.get(1).unwrap().as_str();
            match serde_json::from_str::<RawToolCall>(json_str) {
                Ok(raw_call) => {
                    tool_calls.push(ToolCallInvocation {
                        name: raw_call.name,
                        arguments: raw_call.arguments.unwrap_or_default(),
                    });
                }
                Err(e) => {
                    eprintln!("Failed to parse tool call JSON: {}", e);
                }
            }
        }

        Ok(tool_calls)
    }
}

#[derive(Deserialize)]
struct RawToolCall {
    name: String,
    #[serde(default)]
    arguments: Option<std::collections::HashMap<String, String>>,
}

struct ToolCallInvocation {
    name: String,
    arguments: std::collections::HashMap<String, String>,
}