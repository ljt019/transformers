# transformers v0.0.2

> This crate is under active development. APIs may change as features are still being added.
> Current supported models:
>
> - Gemma3: sizes 1B, 4B, 12B, 27B
> - Phi4: size 14B

Transformers provides a simple, idiomatic Rust interface for running local large language models (LLMs) via the [Candle](https://github.com/huggingface/candle) framework. It offers an API inspired by Python's [transformers](https://huggingface.co/docs/transformers), tailored for Rust developers.

## Installation

```cmd
cargo add transformers
```

## Usage

```rust
use transformers::pipelines::text_generation_pipeline::{
    TextGenerationPipelineBuilder, ModelOptions, Gemma3Size,
};

fn main() {
    // 1. Choose a model family and size
    let model_choice = ModelOptions::Gemma3(Gemma3Size::Size4B);

    // 2. Build the pipeline with optional parameters
    let pipeline = TextGenerationPipelineBuilder::new(model_choice)
        .temperature(0.7)
        .repeat_penalty(1.1)
        .build().unwrap();

    // 3. Generate text
    let prompt = "Explain the concept of Large Language Models in simple terms.";
    let max_tokens = 100;

    let generated = pipeline.generate_text(prompt, max_tokens).unwrap();
    println!("{}", generated);
}
```

## Supported Models

- **Gemma3**: `1B`, `4B`, `12B`, `27B`
- **Phi4**: `14B`

## Future Plans

- Add more model families and sizes
- Support additional pipelines (summarization, classification)
- Improve performance and error handling

## Credits

A special thanks to [Diaconu Radu-Mihai](https://github.com/radudiaconu0/) for transferring the `transformers` crate name on crates.io