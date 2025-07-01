use anyhow::Result;
use transformers::pipelines::fill_mask_pipeline::*;

fn main() -> Result<()> {
    println!("Building pipeline...");

    let pipeline = FillMaskPipelineBuilder::modernbert(ModernBertSize::Base).build()?;

    println!("Pipeline built successfully.");

    let text = "I love my [MASK] car.";

    let result = pipeline.fill_mask(text)?;
    println!("Result: {:?}", result);

    Ok(())
}
