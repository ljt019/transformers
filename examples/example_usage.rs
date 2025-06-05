// Example usage of the new trait-based text generation pipeline architecture
use transformers::{
    pipelines::{
        Gemma3ModelOptions, Gemma3Size, Phi4ModelOptions, Phi4Size, Qwen3ModelOptions, Qwen3Size,
        TextGenerationPipeline, TextGenerationPipelineBuilder,
        TextGenerationPipelineWithToggleableReasoning, TextGenerationPipelineWithTools, Tool,
    },
    Message,
};

fn main() -> anyhow::Result<()> {
    // Example 1: Basic Model (Phi4) - only basic text generation
    let basic_pipeline =
        TextGenerationPipelineBuilder::new(Phi4ModelOptions::new(Phi4Size::Size14B))
            .temperature(0.7)
            .repeat_penalty(1.1)
            .build()?;

    // Only basic methods are available
    let response = basic_pipeline.prompt_completion("What is the meaning of life?", 100)?;
    println!("Phi4 response: {}", response);

    // Example 2: Tool Calling Model (Gemma3) - basic + tool calling
    let mut tool_pipeline =
        TextGenerationPipelineBuilder::new(Gemma3ModelOptions::new(Gemma3Size::Size4B))
            .temperature(0.8)
            .build()?;

    // Register a tool
    let calculator_tool = Tool::new(
        "calculator".to_string(),
        "Performs basic mathematical calculations".to_string(),
        serde_json::json!({
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "Mathematical expression to evaluate"
                }
            },
            "required": ["expression"]
        }),
    );

    tool_pipeline.register_tool(calculator_tool)?;
    let tool_result = tool_pipeline.call_with_tools("What is 15 * 23?", 100)?;
    println!("Gemma3 tool result: {:?}", tool_result);

    // Example 3: Full Featured Model (Qwen3) - basic + reasoning + tools
    let mut full_pipeline =
        TextGenerationPipelineBuilder::new(Qwen3ModelOptions::new(Qwen3Size::Size8B))
            .temperature(0.7)
            .seed(42)
            .build()?;

    // Enable reasoning
    full_pipeline.enable_reasoning();

    // Register tools
    let weather_tool = Tool::new(
        "get_weather".to_string(),
        "Get current weather for a location".to_string(),
        serde_json::json!({
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "City and country"
                }
            },
            "required": ["location"]
        }),
    );

    full_pipeline.register_tool(weather_tool)?;

    // Use both reasoning and tools
    let complex_result = full_pipeline.call_with_tools(
        "Think step by step: Should I bring an umbrella if I'm visiting Paris tomorrow?",
        200,
    )?;
    println!("Qwen3 reasoning + tools result: {:?}", complex_result);

    // Check reasoning status
    if full_pipeline.is_reasoning_enabled() {
        println!("Reasoning is enabled");
        if let Some(trace) = full_pipeline.get_reasoning_trace() {
            println!("Reasoning trace: {}", trace);
        }
    }

    // Disable reasoning for faster generation
    full_pipeline.disable_reasoning();

    // Use with messages
    let messages = vec![
        Message::system("You are a helpful assistant."),
        Message::user("Hello, how are you?"),
    ];

    let message_response = full_pipeline.message_completion(messages, 50)?;
    println!("Message completion: {}", message_response);

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compile_time_type_safety() {
        // This test verifies that the type system works correctly
        // Different model options return different pipeline types

        let qwen3_options = Qwen3ModelOptions::new(Qwen3Size::Size1_7B);
        let gemma3_options = Gemma3ModelOptions::new(Gemma3Size::Size1B);
        let phi4_options = Phi4ModelOptions::new(Phi4Size::Size14B);

        // Verify capabilities are correctly declared
        assert!(qwen3_options.capabilities().tool_calling);
        assert!(matches!(
            qwen3_options.capabilities().reasoning,
            transformers::pipelines::ReasoningSupport::Toggleable
        ));

        assert!(gemma3_options.capabilities().tool_calling);
        assert!(matches!(
            gemma3_options.capabilities().reasoning,
            transformers::pipelines::ReasoningSupport::None
        ));

        assert!(!phi4_options.capabilities().tool_calling);
        assert!(matches!(
            phi4_options.capabilities().reasoning,
            transformers::pipelines::ReasoningSupport::None
        ));
    }
}
