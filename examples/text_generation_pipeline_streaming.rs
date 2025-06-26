use anyhow::Result;
use futures;
use futures::StreamExt;
use std::io::Write;
use transformers::models::Qwen3Size;
use transformers::pipelines::text_generation_pipeline::*;
use transformers::Message;

fn main() -> Result<()> {
    futures::executor::block_on(async {
        println!("Building pipeline...");

        // 1. Create the pipeline, using the builder to configure the model
        let mut pipeline = TextGenerationPipelineBuilder::qwen3(Qwen3Size::Size0_6B)
            .temperature(0.7)
            .max_len(500)
            .build()?;
        println!("Pipeline built successfully.");

        // 2. Define messages and max length
        let messages = vec![
            Message::system("You are a helpful assistant."),
            Message::user("Explain the concept of Large Language Models in simple terms."),
        ];

        // Get the last user message in the messages vector
        let prompt = messages
            .iter()
            .rev()
            .find(|message| message.role() == "user")
            .unwrap()
            .content();

        println!("Generating text for prompt: '{}'", prompt);

        // 3. Generate text
        let mut stream = pipeline.message_completion_stream(&messages)?;

        println!("\n--- Generated Text ---");
        while let Some(tok) = stream.next().await {
            print!("{}", tok);
            std::io::stdout().flush().unwrap();
        }
        println!("\n--- End of Text ---\n");

        Ok(())
    })
}
