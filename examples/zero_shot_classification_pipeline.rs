use anyhow::Result;
use transformers::pipelines::zero_shot_classification_pipeline::*;

fn main() -> Result<()> {
    println!("Building zero-shot classification pipeline...");

    // 1. Create the pipeline, selecting the desired model size.
    let pipeline =
        ZeroShotClassificationPipelineBuilder::new(ZeroShotModernBertSize::Base).build()?;

    println!("Pipeline built successfully.");

    // 2. Define a premise and candidate labels
    let premise = "Apple just announced the new iPhone 15 with USB-C.";
    let candidate_labels = &["technology", "business", "politics", "sports"];

    println!("\nClassifying premise: '{}'", premise);
    println!("With labels: {:?}", candidate_labels);

    // 3. Perform prediction
    let results = pipeline.predict(premise, candidate_labels)?;

    println!("\n--- Results ---");
    for (label, score) in results {
        println!("  - {}: {:.4}", label, score);
    }
    println!("--- End ---");

    // Example 2: Different topic
    let premise2 = "The local team won the championship game last night!";
    let candidate_labels2 = &["technology", "business", "politics", "sports"];

    println!("\nClassifying premise: '{}'", premise2);
    println!("With labels: {:?}", candidate_labels2);

    let results2 = pipeline.predict(premise2, candidate_labels2)?;

    println!("\n--- Results 2 ---");
    for (label, score) in results2 {
        println!("  - {}: {:.4}", label, score);
    }
    println!("--- End 2 ---");

    Ok(())
}
