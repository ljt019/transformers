use anyhow::Result;
use transformers::models::Qwen3Size;
use transformers::pipelines::text_generation_pipeline::*;
use transformers::prelude::*;

fn main() -> Result<()> {
    println!("Building pipeline...");

    // 1. Create the pipeline, using the builder to configure the model
    let mut pipeline = TextGenerationPipelineBuilder::qwen3(Qwen3Size::Size0_6B)
        .temperature(0.7)
        .max_len(100)
        .build()?;
    println!("Pipeline built successfully.");

    // 2. Define messages and max length
    let messages = vec![
        Message::system("You are a helpful assistant."),
        Message::user("Explain the concept of Large Language Models in simple terms."),
    ];

    // Get the last user message using the convenient method
    let prompt = messages.last_user().unwrap();

    println!("Generating text for prompt: '{}'", prompt);

    // 3. Generate text
    let generated_text = pipeline.message_completion(&messages)?;

    println!("\n--- Generated Text ---");
    println!("{}", generated_text);
    println!("--- End of Text ---\n");

    let generated_text =
        pipeline.message_completion(&[Message::user("What was my last message about?")])?;

    println!("\n--- Generated Text 2 ---");
    println!("{}", generated_text);
    println!("--- End of Text 2 ---");

    Ok(())
}
