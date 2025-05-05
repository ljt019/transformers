use anyhow::Result;
use transformers::pipelines::fill_mask_pipeline::*;

fn main() -> Result<()> {
    println!("Building fill-mask pipeline...");

    // 1. Create the pipeline, selecting the desired model size
    let pipeline = FillMaskPipelineBuilder::new(ModernBertSize::Base)
        // Optionally configure other settings like .revision("..."), .cpu(), etc.
        .build()?;
    println!("Pipeline built successfully.");

    // 2. Define the prompt with a [MASK] token
    let prompt = "The capital of France is [MASK].";
    println!("\nFilling mask for prompt: '{}'", prompt);

    // 3. Fill the mask
    let filled_text = pipeline.fill_mask(prompt)?;

    println!("\n--- Result ---");
    println!("{}", filled_text);
    println!("--- End ---\n");

    // Example 2
    let prompt2 = "I'm a [MASK] model.";
    println!("Filling mask for prompt: '{}'", prompt2);
    let filled_text2 = pipeline.fill_mask(prompt2)?;
    println!("\n--- Result 2 ---");
    println!("{}", filled_text2);
    println!("--- End 2 ---\n");

    Ok(())
}
