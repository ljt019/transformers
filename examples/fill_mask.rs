use anyhow::Result;
use transformers::pipelines::fill_mask_pipeline::*;

#[tokio::main]
async fn main() -> Result<()> {
    println!("Building pipeline...");

    let pipeline = FillMaskPipelineBuilder::modernbert(ModernBertSize::Base)
        .build()
        .await?;

    println!("Pipeline built successfully.");

    let text = "I love my [MASK] car.";

    let result = pipeline.fill_mask(text)?;
    
    println!("\n=== Fill Mask Results ===");
    println!("Text: \"{}\"", text);
    println!("Predictions:");
    for prediction in result {
        println!("  - {}: {:.4}", prediction.token, prediction.score);
    }

    Ok(())
}
