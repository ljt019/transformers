use anyhow::Result;
use transformers::models::quantized_qwen3::Qwen3Size;
use transformers::pipelines::text_generation_pipeline::*;

fn main() -> Result<()> {
    println!("Building pipeline 1...");
    // 1. Create the pipeline, using the builder to configure the model
    let mut pipeline_1 = TextGenerationPipelineBuilder::qwen3(Qwen3Size::Size4B)
        .temperature(0.7)
        .max_len(10)
        .build()?;
    println!("Pipeline 1 built successfully.");

    println!("Building pipeline 2...");
    let mut pipeline_2 = TextGenerationPipelineBuilder::qwen3(Qwen3Size::Size4B)
        .temperature(0.7)
        .max_len(10)
        .build()?;
    println!("Pipeline 2 built successfully.");

    println!("Building pipeline 3...");
    let mut pipeline_3 = TextGenerationPipelineBuilder::qwen3(Qwen3Size::Size4B)
        .temperature(0.7)
        .max_len(10)
        .build()?;
    println!("Pipeline 3 built successfully.");

    println!("Building pipeline 4...");
    let mut pipeline_4 = TextGenerationPipelineBuilder::qwen3(Qwen3Size::Size4B)
        .temperature(0.7)
        .max_len(10)
        .build()?;
    println!("Pipeline 4 built successfully.");

    println!("Building pipeline 5...");
    let mut pipeline_5 = TextGenerationPipelineBuilder::qwen3(Qwen3Size::Size4B)
        .temperature(0.7)
        .max_len(10)
        .build()?;
    println!("Pipeline 5 built successfully.");

    println!("Building pipeline 6...");
    let mut pipeline_6 = TextGenerationPipelineBuilder::qwen3(Qwen3Size::Size4B)
        .temperature(0.7)
        .max_len(10)
        .build()?;
    println!("Pipeline 6 built successfully.");

    println!("Building pipeline 7...");
    let mut pipeline_7 = TextGenerationPipelineBuilder::qwen3(Qwen3Size::Size4B)
        .temperature(0.7)
        .max_len(10)
        .build()?;
    println!("Pipeline 7 built successfully.");

    println!("Building pipeline 8...");
    let mut pipeline_8 = TextGenerationPipelineBuilder::qwen3(Qwen3Size::Size4B)
        .temperature(0.7)
        .max_len(10)
        .build()?;
    println!("Pipeline 8 built successfully.");

    println!("Building pipeline 9...");
    let mut pipeline_9 = TextGenerationPipelineBuilder::qwen3(Qwen3Size::Size4B)
        .temperature(0.7)
        .max_len(10)
        .build()?;
    println!("Pipeline 9 built successfully.");

    println!("Building pipeline 10...");
    let mut pipeline_10 = TextGenerationPipelineBuilder::qwen3(Qwen3Size::Size4B)
        .temperature(0.7)
        .max_len(10)
        .build()?;
    println!("Pipeline 10 built successfully.");

    /*

    Prompting all of them

    */

    let prompt = "Hello, world!";

    let response_1 = pipeline_1.prompt_completion(prompt)?;
    println!("Response 1: {}", response_1);

    let response_2 = pipeline_2.prompt_completion(prompt)?;
    println!("Response 2: {}", response_2);

    let response_3 = pipeline_3.prompt_completion(prompt)?;
    println!("Response 3: {}", response_3);

    let response_4 = pipeline_4.prompt_completion(prompt)?;
    println!("Response 4: {}", response_4);

    let response_5 = pipeline_5.prompt_completion(prompt)?;
    println!("Response 5: {}", response_5);

    let response_6 = pipeline_6.prompt_completion(prompt)?;
    println!("Response 6: {}", response_6);

    let response_7 = pipeline_7.prompt_completion(prompt)?;
    println!("Response 7: {}", response_7);

    let response_8 = pipeline_8.prompt_completion(prompt)?;
    println!("Response 8: {}", response_8);

    let response_9 = pipeline_9.prompt_completion(prompt)?;
    println!("Response 9: {}", response_9);

    let response_10 = pipeline_10.prompt_completion(prompt)?;
    println!("Response 10: {}", response_10);

    Ok(())
}
