use anyhow::Result;
use transformers::pipelines::text_generation_pipeline::*;

fn main() -> Result<()> {
    // Start by creating the pipeline, using the builder to configure any generation parameters.
    // Parameters are optional, defaults are set to good values for each model.
    let pipeline = TextGenerationPipelineBuilder::qwen3(Qwen3Size::Size0_6B)
        .temperature(0.7)
        .top_k(40)
        .max_len(1024)
        .build()?;

    // Get a completion from a prompt.
    let completion = pipeline
        .prompt_completion("Explain the concept of Large Language Models in simple terms.")?;

    println!("\n--- Generated Text ---");
    println!("{}", completion);

    // Create and use messages for your completions to keep a conversation going.
    let mut messages = vec![
        Message::system("You are a helpful pirate assistant."),
        Message::user("What is the capital of France?"),
    ];

    let completion = pipeline.message_completion(&messages)?;

    println!("\n--- Generated Text 2 ---");
    println!("{}", completion);

    // To continue the conversation, add the response to the messages
    messages.push(Message::assistant(&completion));
    messages.push(Message::user("What are some fun things to do there?"));

    // Now ask a follow-up question.
    let completion = pipeline.message_completion(&messages)?;

    println!("\n--- Generated Text 3 (Follow-up) ---");
    println!("{}", completion);

    Ok(())
}
