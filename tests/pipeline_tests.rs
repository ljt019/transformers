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
        "<tool_response name=\"get_weather\">\nThe weather in Paris is sunny.\n</tool_response>"
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
        acc.push_str(&tok);
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
