use transformers::pipelines::text_generation_pipeline::*;

#[tool]
/// Returns the weather in a given city
fn get_weather(city: String) -> String {
    let string = format!("The weather in {} is: 31 degress, cloudy skies", city);
    string
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("Building pipeline...");

    // 1. Create the pipeline, using the builder to configure the model
    let mut pipeline = TextGenerationPipelineBuilder::qwen3(Qwen3Size::Size0_6B)
        .temperature(0.7)
        .max_len(500)
        .build()?;

    println!("Pipeline built successfully.");

    pipeline.register_tools(tools![get_weather])?;

    // 2. Define messages
    let messages = vec![
        Message::system("You are a helpful assistant."),
        Message::user("What's the weather like in Tokyo?"),
    ];

    println!(
        "Generating completion for prompt: '{}'\n",
        messages.last_user().unwrap()
    );

    // 3. Generate text
    let mut stream = pipeline.message_completion_stream_with_tools(&messages)?;

    while let Some(tok) = stream.next().await {
        print!("{}", tok);
        std::io::stdout().flush().unwrap();
    }

    Ok(())
}
