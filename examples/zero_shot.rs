use anyhow::Result;
use transformers::pipelines::zero_shot_classification_pipeline::*;

fn main() -> Result<()> {
    println!("Building pipeline...");

    // 1. Create the pipeline, using the builder to configure the model
    let mut pipeline = ZeroShotClassificationPipelineBuilder::modernbert(ZeroShotModernBertSize::Base);

    println!("Pipeline built successfully.");

    Ok(())
}
