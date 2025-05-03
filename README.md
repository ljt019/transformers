# transformers v0.0.3

> This crate is under active development. APIs may change as features are still being added.
> Current supported models:
>
> - Qwen3: sizes 0.6B, 1.7B, 4B, 8B, 14B, 32B
> - Gemma3: sizes 1B, 4B, 12B, 27B
> - Phi4: size 14B

Transformers provides a simple, idiomatic Rust interface for running local large language models (LLMs) via the [Candle](https://github.com/huggingface/candle) framework. It offers an API inspired by Python's [transformers](https://huggingface.co/docs/transformers), tailored for Rust developers.

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
transformers = "0.0.2"
```

or

```cmd
cargo add transformers
```

## Usage

```rust
use anyhow::Result;
use transformers::pipelines::text_generation_pipeline::{
    TextGenerationPipelineBuilder, ModelOptions, Qwen3Size,
};

fn main() -> Result<()> {
    // 1. Choose a model family and size
    let model_choice = ModelOptions::Qwen3(Qwen3Size::Size1_7B);

    // 2. Build the pipeline with optional parameters
    let pipeline = TextGenerationPipelineBuilder::new(model_choice)
        .temperature(0.7)
        .repeat_penalty(1.1)
        .build()?;

    // 3. Generate text
    let prompt = "What is the meaning of life?";
    let max_tokens = 100;

    let generated = pipeline.generate_text(prompt, max_tokens)?;
    println!("{}", generated);

    Ok(())
}
```

## Supported Models

- **Qwen3**: `0.6B`, `1.7B`, `4B`, `8B`, `14B`, `32B`
- **Gemma3**: `1B`, `4B`, `12B`, `27B`
- **Phi4**: `14B`

## Future Plans

- Add more model families and sizes
- Support additional pipelines (summarization, classification)
- Improve performance and error handling

## Credits

A special thanks to [Diaconu Radu-Mihai](https://github.com/radudiaconu0/) for transferring the `transformers` crate name on crates.io.
