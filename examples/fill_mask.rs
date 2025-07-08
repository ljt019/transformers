use anyhow::Result;
use transformers::pipelines::fill_mask_pipeline::*;
use transformers::pipelines::utils::BasePipelineBuilder;

#[tokio::main]
async fn main() -> Result<()> {
    println!("Building pipeline...");

    let pipeline = FillMaskPipelineBuilder::modernbert(ModernBertSize::Base)
        .build()
        .await?;

    println!("Pipeline built successfully.");

    let text = "I love my [MASK] car.";

    let result = pipeline.predict(text)?;

    println!("\n=== Fill Mask Results ===");
    println!("Text: \"{}\"", text);
    println!(
        "Prediction: \"{}\" (confidence: {:.4})",
        result.word, result.score
    );

    // Show top-k predictions
    let top_predictions = pipeline.predict_top_k(text, 3)?;
    println!("\nTop 3 predictions:");
    for (i, prediction) in top_predictions.iter().enumerate() {
        println!(
            "  {}. \"{}\" (confidence: {:.4})",
            i + 1,
            prediction.word,
            prediction.score
        );
    }

    Ok(())
}
