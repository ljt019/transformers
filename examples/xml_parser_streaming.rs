use anyhow::Result;
use transformers::pipelines::text_generation_pipeline::*;

#[tokio::main]
async fn main() -> Result<()> {
    // Create an XML parser with tags to watch for
    let mut xml_parser_builder = XmlParserBuilder::new();
    let think_tag = xml_parser_builder.register_tag("think");
    let tool_response_tag = xml_parser_builder.register_tag("tool_response");
    let xml_parser = xml_parser_builder.build();

    // Build a pipeline with XML parsing capabilities
    let pipeline = TextGenerationPipelineBuilder::qwen3(Qwen3Size::Size0_6B)
        .max_len(1024)
        .with_xml_parser(xml_parser)
        .build_xml()?;

    // Stream completion - this will yield Event items
    let mut stream = pipeline.completion_stream(
        "Think through this problem: If a train travels 60 miles in 1.5 hours, what is its average speed?"
    )?;

    println!("\n--- Streaming Events ---");
    while let Some(event) = stream.next().await {
        match event.tag() {
            Some(tag) if tag == &think_tag => {
                print!("[THINKING] {}", event.get_content());
            }
            Some(tag) if tag == &tool_response_tag => {
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