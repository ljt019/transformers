use anyhow::Result;
use transformers::pipelines::text_generation_pipeline::*;

fn main() -> Result<()> {
    // Build a pipeline with XML parsing capabilities
    let pipeline = TextGenerationPipelineBuilder::qwen3(Qwen3Size::Size0_6B)
        .max_len(1024)
        .build_xml(&["think"])?;

    // Generate completion - this will return Vec<Event>
    let events = pipeline.completion("Solve this math problem step by step: What is 15% of 80?")?;

    println!("\n--- Generated Events ---");
    for event in events {
        match event.tag() {
            Some("think") => match event.part() {
                TagParts::Start => println!("[THINKING]"),
                TagParts::Content => println!("{}", event.get_content()),
                TagParts::End => println!(),
            },
            _ => {
                println!("[OUTPUT] {}", event.get_content());
            }
        }
    }

    Ok(())
}
