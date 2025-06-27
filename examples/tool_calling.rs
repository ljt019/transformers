use anyhow::Result;
use thiserror::Error;
use transformers::pipelines::text_generation_pipeline::*;

#[derive(Debug, Error)]
pub enum WeatherError {
    #[error("City '{city}' not found")]
    CityNotFound { city: String },
    #[error("Weather service temporarily unavailable")]
    ServiceUnavailable,
}

#[tool(on_error = ErrorStrategy::ReturnToModel, retries = 3)]
/// Get the weather for a given city.
fn get_weather(city: String) -> Result<String, WeatherError> {
    // Simulate some error conditions
    if city.to_lowercase() == "japan" {
        return Err(WeatherError::CityNotFound { city });
    }

    Ok(format!(
        "Weather for city: {} - 20 degrees Celsius, sunny, and clear skies.",
        city
    ))
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("Building pipeline...");

    let mut pipeline = TextGenerationPipelineBuilder::qwen3(Qwen3Size::Size1_7B)
        .temperature(0.6)
        .top_p(0.95)
        .top_k(20)
        .min_p(0.0)
        .max_len(2000)
        .build()?;

    pipeline.register_tools(tools![get_weather])?;

    let mut stream =
        pipeline.prompt_completion_stream_with_tools("What's the weather like in Japan?")?;

    while let Some(tok) = stream.next().await {
        print!("{}", tok);
        std::io::stdout().flush().unwrap();
    }

    Ok(())
}
