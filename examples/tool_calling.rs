use anyhow::Result;
use transformers::pipelines::text_generation_pipeline::*;

#[tool]
/// Get the weather for a given city.
fn get_weather(city: String) -> String {
    println!("Debug: Getting weather for city: {}", city);
    format!(
        "Weather for city: {} - 20 degrees Celsius, sunny, and clear skies.",
        city
    )
}

fn main() -> Result<()> {
    println!("Building pipeline...");

    let mut pipeline = TextGenerationPipelineBuilder::qwen3(Qwen3Size::Size0_6B)
        .temperature(0.7)
        .max_len(1000)
        .build()?;

    pipeline.register_tools(tools![get_weather])?;

    let generated_text = pipeline.prompt_completion_with_tools("Get the weather for Britain.")?;

    println!("\n--- Generated Text---");
    println!("{}", generated_text);
    println!("--- End of Text---");

    Ok(())
}
