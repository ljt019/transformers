// Integration tests for sentiment analysis pipeline
// This is a separate crate that tests the public API

use transformers::pipelines::sentiment_analysis_pipeline::*;

#[test]
fn basic_sentiment() -> anyhow::Result<()> {
    let pipeline = SentimentAnalysisPipelineBuilder::modernbert(ModernBertSize::Base).build()?;
    let res = pipeline.predict("I love Rust!")?;
    assert!(!res.trim().is_empty());
    Ok(())
}
