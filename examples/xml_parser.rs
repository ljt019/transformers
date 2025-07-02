use anyhow::Result;
use transformers::pipelines::text_generation_pipeline::*;

fn main() -> Result<()> {
    let pipeline = TextGenerationPipelineBuilder::qwen3(Qwen3Size::Size0_6B)
        .max_len(1024)
        .build()?;

    let xml_parser = XmlParserBuilder::new().register_tag("think").build();

    let completion = pipeline.

    println!("\n--- Generated Text ---");
    println!("{}", completion);

    Ok(())
}
