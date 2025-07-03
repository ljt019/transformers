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

    while let Some(event) = stream.next().await {
        match event.tag() {
            Some("think") => match event.part() {
                TagParts::Start => println!("[THINKING]"),
                TagParts::Content => print!("{}", event.get_content()),
                TagParts::End => println!(),
            },
            Some("tool_response") => match event.part() {
                TagParts::Start => print!("[TOOL] "),
                TagParts::Content => print!("{}", event.get_content()),
                TagParts::End => println!(),
            },
            _ => if event.part() == TagParts::Content {
                print!("{}", event.get_content());
            },
        }
        std::io::stdout().flush().unwrap();
    }

    println!();

    Ok(())
}
