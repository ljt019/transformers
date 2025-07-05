// Integration tests for tool error handling
// This is a separate crate that tests the public API

use transformers::pipelines::text_generation_pipeline::*;

#[tool(on_error = ErrorStrategy::Fail, retries = 1)]
fn fail_tool() -> Result<String, ToolError> {
    Err(ToolError::Message("boom".into()))
}

#[tokio::test]
async fn test_tool_fail_strategy() -> anyhow::Result<()> {
    // Fail strategy should propagate the error
    let pipeline = TextGenerationPipelineBuilder::qwen3(Qwen3Size::Size0_6B)
        .seed(0)
        .max_len(200)
        .build()
        .await?;

    pipeline.register_tools(tools![fail_tool]).await?;

    let res = pipeline.completion_with_tools("call fail_tool").await;

    println!("res: {:?}", res);

    assert!(res.is_err());

    Ok(())
}

#[tool(on_error = ErrorStrategy::ReturnToModel, retries = 1)]
fn fail_tool_model() -> Result<String, ToolError> {
    Err(ToolError::Message("boom".into()))
}

#[tokio::test]
async fn test_tool_return_to_model_strategy() -> anyhow::Result<()> {
    let pipeline = TextGenerationPipelineBuilder::qwen3(Qwen3Size::Size0_6B)
        .seed(0)
        .max_len(200)
        .build()
        .await?;

    pipeline.register_tools(tools![fail_tool_model]).await?;

    let res = pipeline.completion_with_tools("call fail_tool_model").await;

    println!("res: {:?}", res);

    // Kinda hacky, but set seed, and can't think of an easier way to test this
    assert!(res.unwrap().contains("Error"));

    Ok(())
}
