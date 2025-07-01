use anyhow::Result;
use transformers::pipelines::sentiment_analysis_pipeline::*;

fn main() -> Result<()> {
    println!("Building pipeline...");

    let pipeline = SentimentAnalysisPipelineBuilder::modernbert(ModernBertSize::Base).build()?;

    println!("Pipeline built successfully.");

    let text = "I love my new car";

    let result = pipeline.predict(text)?;
    println!("Result: {:?}", result);

    Ok(())
}
