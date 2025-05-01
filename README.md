# rustformers

> ⚠️ **Work in Progress** ⚠️
>
> This project is currently under active development. The API may change, and features are still being added.
> Currently, only the `gemma-3-1b-it` model is supported.

My goal with this crate is to provide a simple and idiomatic Rust interface for using popular local Large Language Models (LLMs), leveraging the [Candle](https://github.com/huggingface/candle) framework. While Candle enables working with LLMs in Rust, this crate aims to offer an API inspired by [Python's transformers](https://huggingface.co/docs/transformers) library, but tailored for Rust developers.

Currently, `rustformers` supports a basic text generation pipeline.

## Getting Started

Add `rustformers` to your `Cargo.toml`:

```toml
[dependencies]
rustformers = { git = "https://github.com/your-username/rustformers.git" } # Replace with actual path/version when published
anyhow = "1.0"
```

## Example Usage

Here's a basic example demonstrating how to use the text generation pipeline with the Gemma 3.1b model:

```rust
use anyhow::Result;
use rustformers::pipelines::text_generation_pipeline::*;

fn main() -> Result<()> {
    println!("Building pipeline...");

    // 1. Create the builder
    let builder = TextGenerationPipelineBuilder::new();

    // 2. Set the model choice
    // Currently supports ModelOptions::Gemma3_1b
    let builder = builder.set_model_choice(ModelOptions::Gemma3_1b);

    // 3. Build the pipeline
    // This might take time as it downloads and loads the model weights
    let pipeline = builder.build()?;
    println!("Pipeline built successfully.");

    // 4. Define prompt and max length
    let prompt = "Explain the concept of Large Language Models in simple terms.";
    let max_length = 500; // Maximum number of tokens to generate

    println!("Generating text for prompt: '{}'", prompt);

    // 5. Generate text
    let generated_text = pipeline.generate_text(prompt, max_length)?;

    // 6. Print the result
    println!("
--- Generated Text ---");
    println!("{}", generated_text);
    println!("--- End of Text ---
");

    Ok(())
}
```

This example initializes the pipeline, sets the desired model (Gemma 3.1b in this case), builds it (which involves loading the model weights), and then uses it to generate text based on a prompt.

## Future Plans

* Support more models.
* Implement other pipeline types (e.g., text classification, summarization).
* Improve error handling and configuration options.