use anyhow::Result;
use transformers::pipelines::text_generation_pipeline::*;
use futures::StreamExt;

#[tokio::main]
async fn main() -> Result<()> {
    // Start by creating the pipeline, using the builder to configure any generation parameters.
    let pipeline = TextGenerationPipelineBuilder::qwen3(Qwen3Size::Size0_6B)
        .max_len(1024)
        .build()
        .await?;

    let mut stream = pipeline
        .completion_stream(
        "Explain the concept of Large Language Models in simple terms.",
    )
    .await?;
    futures::pin_mut!(stream);

    println!("\n--- Generated Text ---");
    while let Some(tok) = stream.next().await {
        print!("{}", tok?);
        std::io::stdout().flush().unwrap();
    }

    // Also supports messages obviously
    let messages = vec![
        Message::system("You are a helpful pirate assistant."),
        Message::user("What is the capital of France?"),
    ];

    let mut stream_two = pipeline.completion_stream(&messages).await?;
    futures::pin_mut!(stream_two);

    println!("\n--- Generated Text 2 ---");
    while let Some(tok) = stream_two.next().await {
        print!("{}", tok?);
        std::io::stdout().flush().unwrap();
    }

    Ok(())
}
