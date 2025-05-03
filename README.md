# transformers v0.0.3

> This crate is under active development. APIs may change as features are still being added.
>
> **Supported Pipelines:**
>
> - Text Generation: Gemma3 (1B, 4B, 12B, 27B), Phi4 (14B)
> - Fill-Mask: ModernBERT (Base, Large)

Transformers provides a simple, idiomatic Rust interface for running local large language models (LLMs) via the [Candle](https://github.com/huggingface/candle) framework. It offers an API inspired by Python's [transformers](https://huggingface.co/docs/transformers), tailored for Rust developers.

## Installation

```cmd
cargo add transformers
```

## Usage

### Text Generation

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

### Fill Mask (ModernBERT)

```rust
use transformers::pipelines::fill_mask_pipeline::{
    FillMaskPipelineBuilder, ModernBertSize,
};

fn main() -> anyhow::Result<()> {
    // 1. Choose a model size (Base or Large)
    let size = ModernBertSize::Base;

    // 2. Build the pipeline
    let pipeline = FillMaskPipelineBuilder::new(size).build()?;

    // 3. Fill the mask
    let prompt = "The capital of France is [MASK].";
    let filled_text = pipeline.fill_mask(prompt)?;

    println!("{}", filled_text); // Should print: The capital of France is Paris.
    Ok(())
}
```

## Supported Models & Pipelines

**Text Generation**:

- Gemma3: `1B`, `4B`, `12B`, `27B`
- Phi4: `14B`

**Fill-Mask**:

- ModernBERT: `Base`, `Large` (using `answerdotai/ModernBERT-base` or `-large`)

## Future Plans

- Add more model families and sizes
- Support additional pipelines (summarization, classification)
- Improve performance and error handling

## Credits

A special thanks to [Diaconu Radu-Mihai](https://github.com/radudiaconu0/) for transferring the `transformers` crate name on crates.io