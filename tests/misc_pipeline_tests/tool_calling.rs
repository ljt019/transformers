// Integration tests for tool calling functionality
// This is a separate crate that tests the public API

use transformers::pipelines::text_generation_pipeline::*;

#[tool]
fn get_weather(city: String) -> Result<String, ToolError> {
    Ok(format!("The weather in {city} is sunny."))
}

#[tokio::test]
async fn basic_tool_use() -> anyhow::Result<()> {
    let pipeline = TextGenerationPipelineBuilder::qwen3(Qwen3Size::Size0_6B)
        .seed(42)
        .max_len(150)
        .build()
        .await?;
    pipeline.register_tools(tools![get_weather]).await?;
    let out = pipeline
        .completion_with_tools("What's the weather like in Paris today?")
        .await?;
    println!("{}", out);
    assert!(out.contains(
        "<tool_result name=\"get_weather\">\nThe weather in Paris is sunny.\n</tool_result>"
    ));
    Ok(())
}
