use anyhow::Result;
use transformers::pipelines::text_generation_pipeline::*;

fn main() -> Result<()> {
    println!("Building pipeline...");

    // 1. Create the pipeline, using the builder to configure the model
    let pipeline = TextGenerationPipelineBuilder::new(ModelOptions::Qwen3(Qwen3Size::Size1_7B))
        .temperature(0.7)
        .build()?;
    println!("Pipeline built successfully.");

    // 2. Define prompt and max length
    let prompt = "Explain the concept of Large Language Models in simple terms.";
    let max_length = 250; // Maximum number of tokens to generate

    println!("Generating text for prompt: '{}'", prompt);

    // 3. Generate text
    let generated_text = pipeline.generate_text(prompt, max_length)?;

    println!("\n--- Generated Text ---");
    println!("{}", generated_text);
    println!("--- End of Text ---\n");

    let second_prompt = "Explain the fibonacci sequence in simple terms. /no_think";

    let generated_text = pipeline.generate_text(second_prompt, max_length)?;

    println!("\n--- Generated Text 2 ---");
    println!("{}", generated_text);
    println!("--- End of Text 2 ---");

    Ok(())
}
