// Integration tests for text generation pipeline
// This is a separate crate that tests the public API

use futures::StreamExt;
use transformers::pipelines::text_generation::*;

#[tokio::test]
async fn basic_text_generation() -> anyhow::Result<()> {
    let pipeline = TextGenerationPipelineBuilder::qwen3(Qwen3Size::Size0_6B)
        .seed(42)
        .temperature(0.7)
        .max_len(8)
        .build()
        .await?;
    let out = pipeline.completion("Rust is a").await?;
    assert!(!out.trim().is_empty());
    Ok(())
}

#[tokio::test]
async fn basic_streaming() -> anyhow::Result<()> {
    let pipeline = TextGenerationPipelineBuilder::qwen3(Qwen3Size::Size0_6B)
        .seed(42)
        .max_len(8)
        .build()
        .await?;
    let mut stream = pipeline.completion_stream("Hello").await?;
    let mut acc = String::new();
    while let Some(tok) = stream.next().await {
        acc.push_str(&tok?);
    }
    assert!(!acc.trim().is_empty());
    Ok(())
}

#[tokio::test]
async fn test_empty_input_handling() -> anyhow::Result<()> {
    let pipeline = TextGenerationPipelineBuilder::qwen3(Qwen3Size::Size0_6B)
        .seed(0)
        .max_len(4)
        .build()
        .await?;
    let out = pipeline.completion("").await?;
    assert!(!out.trim().is_empty());
    Ok(())
}

#[tokio::test]
async fn test_set_generation_params() -> anyhow::Result<()> {
    let pipeline = TextGenerationPipelineBuilder::qwen3(Qwen3Size::Size0_6B)
        .seed(42)
        .max_len(1)
        .build()
        .await?;

    let short = pipeline.completion("Rust is a").await?;

    let new_params = GenerationParams::new(0.7, 1.0, 64, 42, 8, 1.0, 0, 0.0);
    pipeline.set_generation_params(new_params).await;

    let longer = pipeline.completion("Rust is a").await?;

    assert!(longer.len() >= short.len());
    Ok(())
}
