# rustformers v0.0.2

> ⚠️ **Work in Progress** ⚠️
>
> This crate is under active development. APIs may change as features are still being added.
> Current supported models:
>
> - Gemma3: sizes 1B, 4B, 12B, 27B
> - Phi4: size 14B

Rustformers provides a simple, idiomatic Rust interface for running local large language models (LLMs) via the [Candle](https://github.com/huggingface/candle) framework. It offers an API inspired by Python's [transformers](https://huggingface.co/docs/transformers), tailored for Rust developers.

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
rustformers = "0.0.2"
```

or

```cmd
cargo add rustformers
```

## Usage

```rust
use anyhow::Result;
use rustformers::pipelines::text_generation_pipeline::{
    TextGenerationPipelineBuilder, ModelOptions, Gemma3Size, Phi4Size,
};

fn main() -> Result<()> {
    // 1. Choose a model family and size
    let model_choice = ModelOptions::Gemma3(Gemma3Size::Size1B);
    // alternatively:
    // let model_choice = ModelOptions::Phi4(Phi4Size::Size14B);

    // 2. Build the pipeline with optional parameters
    let pipeline = TextGenerationPipelineBuilder::new(model_choice)
        .temperature(0.7)
        .repeat_penalty(1.1)
        .use_flash_attn(true) // only used by some models, probably will handle automatically soon
        .build()?;

    // 3. Generate text
    let prompt = "What is the meaning of life?";
    let generated = pipeline.generate_text(prompt, 100)?;
    println!("{}", generated);

    Ok(())
}
```

## Supported Models

- **Gemma3**: `1B`, `4B`, `12B`, `27B`
- **Phi4**: `14B`

## Future Plans

- Add more model families and sizes
- Support additional pipelines (summarization, classification)
- Improve performance and error handling
