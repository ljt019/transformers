use anyhow::Result;
use transformers::pipelines::text_generation_pipeline::ToolError;
use transformers::pipelines::text_generation_pipeline::*;

#[tool(on_error = ErrorStrategy::Fail, retries = 5)]
/// Get the weather for a given city.
fn get_humidity(city: String) -> Result<String, ToolError> {
    Ok(format!("The humidity is 1% in {}.", city))
}

/*
    defaults to 3 retries, and ReturnToModel error strategy
*/

#[tool]
/// Get the weather for a given city in degrees celsius.
fn get_temperature(city: String) -> Result<String, ToolError> {
    return Ok(format!(
        "The temperature is 20 degrees celsius in {}.",
        city
    ));
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("Building pipeline...");

    let mut pipeline = TextGenerationPipelineBuilder::qwen3(Qwen3Size::Size0_6B)
        .max_len(8192)
        .build()?;

    println!("Pipeline built successfully.");

    pipeline.register_tools(tools![get_temperature, get_humidity])?;

    let mut stream =
        pipeline.prompt_completion_stream_with_tools("What's the weather like in Tokyo?")?;

    while let Some(tok) = stream.next().await {
        print!("{}", tok);
        std::io::stdout().flush().unwrap();
    }

    Ok(())
}
