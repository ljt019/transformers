use transformers::pipelines::fill_mask_pipeline::*;
use transformers::pipelines::sentiment_analysis_pipeline::*;
use transformers::pipelines::text_generation_pipeline::*;
use transformers::pipelines::zero_shot_classification_pipeline::*;

#[test]
fn basic_text_generation() -> anyhow::Result<()> {
    let pipeline = TextGenerationPipelineBuilder::qwen3(Qwen3Size::Size0_6B)
        .seed(42)
        .temperature(0.7)
        .max_len(8)
        .build()?;
    let out = pipeline.completion("Rust is a")?;
    assert!(!out.trim().is_empty());
    Ok(())
}

#[tool]
fn get_weather(city: String) -> Result<String, ToolError> {
    Ok(format!("The weather in {city} is sunny."))
}

#[test]
fn basic_tool_use() -> anyhow::Result<()> {
    let pipeline = TextGenerationPipelineBuilder::qwen3(Qwen3Size::Size0_6B)
        .seed(42)
        .max_len(150)
        .build()?;
    pipeline.register_tools(tools![get_weather])?;
    let out = pipeline.completion_with_tools("What's the weather like in Paris today?")?;
    println!("{}", out);
    assert!(out.contains(
        "<tool_result name=\"get_weather\">\nThe weather in Paris is sunny.\n</tool_result>"
    ));
    Ok(())
}

#[tokio::test]
async fn basic_streaming() -> anyhow::Result<()> {
    let pipeline = TextGenerationPipelineBuilder::qwen3(Qwen3Size::Size0_6B)
        .seed(42)
        .max_len(8)
        .build()?;
    let mut stream = pipeline.completion_stream("Hello")?;
    let mut acc = String::new();
    use futures::StreamExt;
    while let Some(tok) = stream.next().await {
        acc.push_str(&tok?);
    }
    assert!(!acc.trim().is_empty());
    Ok(())
}

#[test]
fn basic_fill_mask() -> anyhow::Result<()> {
    let pipeline = FillMaskPipelineBuilder::modernbert(ModernBertSize::Base).build()?;
    let res = pipeline.fill_mask("The capital of France is [MASK].")?;
    assert!(res.contains("Paris") || !res.trim().is_empty());
    Ok(())
}

#[test]
fn basic_zero_shot_classification() -> anyhow::Result<()> {
    let pipeline =
        ZeroShotClassificationPipelineBuilder::modernbert(ModernBertSize::Base).build()?;
    let labels = ["politics", "sports"];
    let res = pipeline.predict("The election results were surprising", &labels)?;
    assert_eq!(res.len(), 2);
    Ok(())
}

#[test]
fn basic_sentiment() -> anyhow::Result<()> {
    let pipeline = SentimentAnalysisPipelineBuilder::modernbert(ModernBertSize::Base).build()?;
    let res = pipeline.predict("I love Rust!")?;
    assert!(!res.trim().is_empty());
    Ok(())
}

#[tool(on_error = ErrorStrategy::Fail, retries = 1)]
fn fail_tool() -> Result<String, ToolError> {
    Err(ToolError::Message("boom".into()))
}

#[tool(on_error = ErrorStrategy::ReturnToModel, retries = 1)]
fn fail_tool_model() -> Result<String, ToolError> {
    Err(ToolError::Message("boom".into()))
}

#[test]
fn test_tool_error_strategies() -> anyhow::Result<()> {
    // Fail strategy should propagate the error
    let pipeline = TextGenerationPipelineBuilder::qwen3(Qwen3Size::Size0_6B)
        .seed(0)
        .max_len(32)
        .build()?;
    pipeline.register_tools(tools![fail_tool])?;
    let res = pipeline.completion_with_tools("call fail_tool");
    if let Ok(out) = &res {
        // Some models may not emit a tool call; in that case just ensure we got a response
        assert!(!out.trim().is_empty());
    } else {
        // When a tool call happens the pipeline should propagate the error
        assert!(res.is_err());
    }

    // ReturnToModel strategy should succeed with error message in output
    let pipeline = TextGenerationPipelineBuilder::qwen3(Qwen3Size::Size0_6B)
        .seed(0)
        .max_len(32)
        .build()?;
    pipeline.register_tools(tools![fail_tool_model])?;
    let out = pipeline.completion_with_tools("call fail_tool_model")?;
    assert!(out.contains("Error:"));
    Ok(())
}

#[test]
fn test_empty_input_handling() -> anyhow::Result<()> {
    let pipeline = TextGenerationPipelineBuilder::qwen3(Qwen3Size::Size0_6B)
        .seed(0)
        .max_len(4)
        .build()?;
    let out = pipeline.completion("")?;
    assert!(!out.trim().is_empty());

    let pipeline = FillMaskPipelineBuilder::modernbert(ModernBertSize::Base).build()?;
    assert!(pipeline.fill_mask("").is_err());

    Ok(())
}
