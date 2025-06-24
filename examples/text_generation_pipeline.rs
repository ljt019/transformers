use anyhow::Result;
use transformers::models::Qwen3Size;
use transformers::pipelines::text_generation_pipeline::*;

fn main() -> Result<()> {
    println!("Building pipeline...");

    // 1. Create the pipeline, using the builder to configure the model
    let mut pipeline = TextGenerationPipelineBuilder::qwen3(Qwen3Size::Size0_6B)
        .temperature(0.7)
        .max_len(15)
        .build()?;

    println!("Pipeline built successfully.");

    // 2. Define prompt and max length
    let prompt = "Explain the concept of Large Language Models in simple terms.";

    println!("Generating text for prompt: '{}'", prompt);

    // 3. Generate text
    let generated_text = pipeline.prompt_completion(prompt)?;

    println!("\n--- Generated Text ---");
    println!("{}", generated_text);
    println!("--- End of Text ---\n");

    let second_prompt = "Explain the fibonacci sequence in simple terms. /no_think";

    let generated_text = pipeline.prompt_completion(second_prompt)?;

    println!("\n--- Generated Text 2 ---");
    println!("{}", generated_text);
    println!("--- End of Text 2 ---");

    Ok(())
}
