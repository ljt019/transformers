use anyhow::Result;
use transformers::pipelines::text_generation_pipeline::*;

#[tokio::main]
async fn main() -> Result<()> {
    // Build a pipeline with XML parsing capabilities
    let pipeline = TextGenerationPipelineBuilder::qwen3(Qwen3Size::Size0_6B)
        .max_len(1024)
        .build_xml(&["think", "tool_response"])?;

    // Stream completion - this will yield Event items
    let mut stream = pipeline.completion_stream(
        "Think through this problem: If a train travels 60 miles in 1.5 hours, what is its average speed?"
    )?;

    println!("\n--- Streaming Events ---");
    while let Some(event) = stream.next().await {
        match event.tag() {
            Some("think") => {
                print!("[THINKING] {}", event.get_content());
            }
            Some("tool_response") => {
                print!("[TOOL] {}", event.get_content());
            }
            _ => {
                print!("{}", event.get_content());
            }
        }
        std::io::stdout().flush().unwrap();
    }
    println!();

    Ok(())
}