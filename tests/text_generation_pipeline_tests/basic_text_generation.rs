// Integration tests for text generation pipeline
// This is a separate crate that tests the public API

use futures::StreamExt;
use transformers::pipelines::text_generation_pipeline::*;

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

#[tokio::test]
async fn basic_streaming() -> anyhow::Result<()> {
    let pipeline = TextGenerationPipelineBuilder::qwen3(Qwen3Size::Size0_6B)
        .seed(42)
        .max_len(8)
        .build()?;
    let mut stream = pipeline.completion_stream("Hello")?;
    let mut acc = String::new();
    while let Some(tok) = stream.next().await {
        acc.push_str(&tok?);
    }
    assert!(!acc.trim().is_empty());
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
    Ok(())
}
