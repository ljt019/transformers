use anyhow::Result;
use transformers::pipelines::text_generation_pipeline::*;

fn main() -> Result<()> {
    println!("Building pipeline...");

    // 1. Create the pipeline, using the builder to configure the model
    let mut pipeline = TextGenerationPipelineBuilder::gemma3(Gemma3Size::Size1B)
        .temperature(0.7)
        .max_len(100)
        .build()?;

    println!("Pipeline built successfully.");

    // 3. Generate text
    let generated_text = pipeline
        .prompt_completion("Explain the concept of Large Language Models in simple terms.")?;

    println!("\n--- Generated Text ---");
    println!("{}", generated_text);

    let generated_text =
        pipeline.prompt_completion("Explain the fibonacci sequence in simple terms.")?;

    println!("\n--- Generated Text 2 ---");
    println!("{}", generated_text);

    Ok(())
}
