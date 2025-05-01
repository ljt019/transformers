use anyhow::Result;
use rustformers::pipelines::text_generation_pipeline::*;

fn main() -> Result<()> {
    println!("Building pipeline...");

    // 1. Create the builder
    let builder = TextGenerationPipelineBuilder::new();

    // 2. Set the model choice
    let builder = builder.set_model_choice(ModelOptions::Gemma3_1b);

    // 3. Build the pipeline
    // This might take time as it loads the model weights
    let pipeline = builder.build()?;
    println!("Pipeline built successfully.");

    // 4. Define prompt and max length
    let prompt = "Explain the concept of Large Language Models in simple terms.";
    let max_length = 10000; // Maximum number of tokens to generate

    println!("Generating text for prompt: '{}'", prompt);

    // 5. Generate text
    let generated_text = pipeline.generate_text(prompt, max_length)?;

    // 6. Print the result
    println!("\n--- Generated Text ---");
    println!("{}", generated_text);
    println!("--- End of Text ---\n");

    let second_prompt = "Explain the fibbonnaci sequence in simple terms.";

    let generated_text = pipeline.generate_text(second_prompt, max_length)?;

    println!("\n--- Generated Text 2 ---");
    println!("{}", generated_text);
    println!("--- End of Text 2 ---");

    Ok(())
}
