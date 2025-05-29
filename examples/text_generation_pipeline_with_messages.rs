use anyhow::Result;
use transformers::pipelines::text_generation_pipeline::*;
use transformers::Message;

fn main() -> Result<()> {
    println!("Building pipeline...");

    // 1. Create the pipeline, using the builder to configure the model
    let pipeline = TextGenerationPipelineBuilder::new(ModelOptions::Qwen3(Qwen3Size::Size0_6B))
        .temperature(0.7)
        .build()?;
    println!("Pipeline built successfully.");

    // 2. Define messages and max length
    let mut messages = vec![
        Message::system("You are a helpful assistant."),
        Message::user("Explain the concept of Large Language Models in simple terms."),
    ];
    let max_length = 500; // Maximum number of tokens to generate

    // Get the last user message in the messages vector
    let prompt = messages
        .iter()
        .rev()
        .find(|message| message.role() == "user")
        .unwrap()
        .content();

    println!("Generating text for prompt: '{}'", prompt);

    // 3. Generate text
    let generated_text = pipeline.message_completion(messages.clone(), max_length)?;

    println!("\n--- Generated Text ---");
    println!("{}", generated_text);
    println!("--- End of Text ---\n");

    // Add the models response to the messages
    messages.push(Message::assistant(generated_text.as_str()));

    messages.push(Message::user(
        "Explain the fibonacci sequence in simple terms.",
    ));

    let generated_text = pipeline.message_completion(messages, max_length)?;

    println!("\n--- Generated Text 2 ---");
    println!("{}", generated_text);
    println!("--- End of Text 2 ---");

    Ok(())
}
