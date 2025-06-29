use transformers::pipelines::text_generation_pipeline::*;

#[tokio::main]
async fn main() -> Result<()> {
    println!("Building pipeline...");

    // 1. Create the pipeline, using the builder to configure the model
    let mut pipeline = TextGenerationPipelineBuilder::gemma3(Gemma3Size::Size1B)
        .temperature(0.7)
        .max_len(8192)
        .build()?;

    println!("Pipeline built successfully.");

    // 2. Define messages
    let messages = vec![Message::user(
        "Explain the concept of large language models in simple terms.",
    )];

    println!(
        "Generating completion for prompt: '{}'\n",
        messages.last_user().unwrap()
    );

    // 3. Generate text
    let mut stream = pipeline.message_completion_stream(&messages)?;

    while let Some(tok) = stream.next().await {
        print!("{}", tok);
        std::io::stdout().flush().unwrap();
    }

    Ok(())
}
