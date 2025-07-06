// Integration tests for sentiment analysis pipeline
// This is a separate crate that tests the public API

use transformers::pipelines::sentiment_analysis_pipeline::*;
use candle_core::DeviceLocation;

#[tokio::test]
async fn basic_sentiment() -> anyhow::Result<()> {
    let pipeline = SentimentAnalysisPipelineBuilder::modernbert(ModernBertSize::Base)
        .build()
        .await?;
    let res = pipeline.predict("I love Rust!")?;
    assert!(!res.trim().is_empty());
    Ok(())
}

#[tokio::test]
async fn select_cpu_device() -> anyhow::Result<()> {
    let pipeline = SentimentAnalysisPipelineBuilder::modernbert(ModernBertSize::Base)
        .cpu()
        .build()
        .await?;
    assert!(pipeline.device().is_cpu());
    Ok(())
}

#[tokio::test]
async fn select_cuda_device() -> anyhow::Result<()> {
    let pipeline = SentimentAnalysisPipelineBuilder::modernbert(ModernBertSize::Base)
        .cuda_device(0)
        .build()
        .await?;
    match pipeline.device().location() {
        DeviceLocation::Cuda { gpu_id } => assert_eq!(gpu_id, 0),
        _ => {}
    }
    Ok(())
}
