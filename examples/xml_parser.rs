use anyhow::Result;
use transformers::pipelines::text_generation_pipeline::*;

fn main() -> Result<()> {
    // Create an XML parser with tags to watch for
    let mut xml_parser_builder = XmlParserBuilder::new();
    let think_tag = xml_parser_builder.register_tag("think");
    let xml_parser = xml_parser_builder.build();

    // Build a pipeline with XML parsing capabilities
    let pipeline = TextGenerationPipelineBuilder::qwen3(Qwen3Size::Size0_6B)
        .max_len(1024)
        .with_xml_parser(xml_parser)
        .build_xml()?;

    // Generate completion - this will return Vec<Event>
    let events = pipeline.completion(
        "Solve this math problem step by step: What is 15% of 80?"
    )?;

    println!("\n--- Generated Events ---");
    for event in events {
        match event.tag() {
            Some(tag) if tag == &think_tag => {
                println!("[THINKING] {}", event.get_content());
            }
            _ => {
                println!("[OUTPUT] {}", event.get_content());
            }
        }
    }

    Ok(())
}
