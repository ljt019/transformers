use anyhow::Result;
use transformers::pipelines::zero_shot_classification_pipeline::*;

fn main() -> Result<()> {
    println!("Building pipeline...");

    let pipeline =
        ZeroShotClassificationPipelineBuilder::modernbert(ModernBertSize::Base).build()?;

    println!("Pipeline built successfully.");

    let text = "I love my new car";
    let candidate_labels = vec!["coding", "reading", "writing", "speaking", "cars"];

    let result = pipeline.predict_multi_label(text, &candidate_labels)?;
    println!("Result: {:?}", result);

    Ok(())
}
