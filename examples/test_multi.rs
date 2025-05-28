use anyhow::Result;
use transformers::pipelines::text_generation_pipeline::*;

fn main() -> Result<()> {
    println!("Building pipeline 1...");
    // 1. Create the pipeline, using the builder to configure the model
    let pipeline_1 = TextGenerationPipelineBuilder::new(ModelOptions::Qwen3(Qwen3Size::Size1_7B))
        .temperature(0.7)
        .build()?;
    println!("Pipeline 1 built successfully.");

    println!("Building pipeline 2...");
    let pipeline_2 = TextGenerationPipelineBuilder::new(ModelOptions::Qwen3(Qwen3Size::Size1_7B))
        .temperature(0.7)
        .build()?;
    println!("Pipeline 2 built successfully.");

    println!("Building pipeline 3...");
    let pipeline_3 = TextGenerationPipelineBuilder::new(ModelOptions::Qwen3(Qwen3Size::Size1_7B))
        .temperature(0.7)
        .build()?;
    println!("Pipeline 3 built successfully.");

    println!("Building pipeline 4...");
    let pipeline_4 = TextGenerationPipelineBuilder::new(ModelOptions::Qwen3(Qwen3Size::Size1_7B))
        .temperature(0.7)
        .build()?;
    println!("Pipeline 4 built successfully.");

    println!("Building pipeline 5...");
    let pipeline_5 = TextGenerationPipelineBuilder::new(ModelOptions::Qwen3(Qwen3Size::Size1_7B))
        .temperature(0.7)
        .build()?;
    println!("Pipeline 5 built successfully.");

    println!("Building pipeline 6...");
    let pipeline_6 = TextGenerationPipelineBuilder::new(ModelOptions::Qwen3(Qwen3Size::Size1_7B))
        .temperature(0.7)
        .build()?;
    println!("Pipeline 6 built successfully.");

    println!("Building pipeline 7...");
    let pipeline_7 = TextGenerationPipelineBuilder::new(ModelOptions::Qwen3(Qwen3Size::Size1_7B))
        .temperature(0.7)
        .build()?;
    println!("Pipeline 7 built successfully.");

    println!("Building pipeline 8...");
    let pipeline_8 = TextGenerationPipelineBuilder::new(ModelOptions::Qwen3(Qwen3Size::Size1_7B))
        .temperature(0.7)
        .build()?;
    println!("Pipeline 8 built successfully.");

    println!("Building pipeline 9...");
    let pipeline_9 = TextGenerationPipelineBuilder::new(ModelOptions::Qwen3(Qwen3Size::Size1_7B))
        .temperature(0.7)
        .build()?;
    println!("Pipeline 9 built successfully.");

    println!("Building pipeline 10...");
    let pipeline_10 = TextGenerationPipelineBuilder::new(ModelOptions::Qwen3(Qwen3Size::Size1_7B))
        .temperature(0.7)
        .build()?;
    println!("Pipeline 10 built successfully.");

    /*

    Prompting all of them

    */

    let prompt = "Hello, world!";

    let response_1 = pipeline_1.prompt_completion(prompt, 10)?;
    println!("Response 1: {}", response_1);

    let response_2 = pipeline_2.prompt_completion(prompt, 10)?;
    println!("Response 2: {}", response_2);

    let response_3 = pipeline_3.prompt_completion(prompt, 10)?;
    println!("Response 3: {}", response_3);

    let response_4 = pipeline_4.prompt_completion(prompt, 10)?;
    println!("Response 4: {}", response_4);

    let response_5 = pipeline_5.prompt_completion(prompt, 10)?;
    println!("Response 5: {}", response_5);

    let response_6 = pipeline_6.prompt_completion(prompt, 10)?;
    println!("Response 6: {}", response_6);

    let response_7 = pipeline_7.prompt_completion(prompt, 10)?;
    println!("Response 7: {}", response_7);

    let response_8 = pipeline_8.prompt_completion(prompt, 10)?;
    println!("Response 8: {}", response_8);

    let response_9 = pipeline_9.prompt_completion(prompt, 10)?;
    println!("Response 9: {}", response_9);

    let response_10 = pipeline_10.prompt_completion(prompt, 10000)?;
    println!("Response 10: {}", response_10);

    Ok(())
}
