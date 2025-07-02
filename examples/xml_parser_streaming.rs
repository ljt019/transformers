use anyhow::Result;
use transformers::pipelines::text_generation_pipeline::*;

#[tool]
/// Calculates the average speed given distance and time
fn calculate_average_speed(
    distance_in_miles: u64,
    time_in_minutes: u64,
) -> Result<String, ToolError> {
    Ok(format!(
        "Average speed: {} mph",
        distance_in_miles / time_in_minutes
    ))
}

#[tokio::main]
async fn main() -> Result<()> {
    // Build a pipeline with XML parsing capabilities
    let pipeline = TextGenerationPipelineBuilder::qwen3(Qwen3Size::Size0_6B)
        .max_len(1024)
        .build_xml(&["think", "tool_response"])?;

    pipeline.register_tools(tools![calculate_average_speed])?;

    // Stream completion - this will yield Event items
    let mut stream = pipeline.completion_stream_with_tools(
        "Think through this problem: If a train travels 60 miles in 1.5 hours, what is its average speed?"
    )?;

    println!("\n--- Streaming Events ---");

    // Track whether we are currently printing content inside a <think> tag
    let mut in_thinking = false;

    while let Some(event) = stream.next().await {
        match event.tag() {
            // Content inside <think>...</think>
            Some("think") => {
                if !in_thinking {
                    // First token under the THINKING header
                    println!("[THINKING]");
                    in_thinking = true;
                }
                print!("{}", event.get_content());
            }
            // Content emitted by a tool call
            Some("tool_response") => {
                // Close THINKING block if we were inside one
                if in_thinking {
                    println!();
                    in_thinking = false;
                }
                print!("[TOOL] {}", event.get_content());
            }
            // Regular model output
            _ => {
                // Close THINKING block if we were inside one
                if in_thinking {
                    println!();
                    in_thinking = false;
                }
                print!("{}", event.get_content());
            }
        }
        std::io::stdout().flush().unwrap();
    }

    // Ensure we end with a newline
    if in_thinking {
        println!();
    }
    println!();

    Ok(())
}
