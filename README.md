# transformers v0.0.5

> This crate is under active development. APIs may change as features are still being added.
>
> **Supported Pipelines so far:**
>
> - Text Generation:
>   - Qwen3 (0.6B, 1.7B, 4B, 8B, 14B, 32B) *no moe yet*
>   - Gemma3 (1B, 4B, 12B, 27B)
>   - Phi4 (14B)
>
> - Fill-Mask:
>   - ModernBERT (Base, Large)
>
> - Sentiment Analysis:
>   - ModernBERT Finetune (Base, Large)

Transformers provides a simple, idiomatic Rust interface for running local large language models (LLMs) via the [Candle](https://github.com/huggingface/candle) framework. It offers an API inspired by Python's [transformers](https://huggingface.co/docs/transformers), tailored for Rust developers.

## Installation

```cmd
cargo add transformers
```

## Usage

At this point in development the only real way to interact with the models is through the given pipelines, I plan to eventually allow you to work with the models directly.

Some examples of how to use pipelines:

### Text Generation

```rust
use transformers::pipelines::text_generation_pipeline::{
    TextGenerationPipelineBuilder, ModelOptions, Qwen3Size,
};

fn main() {
    // 1. Choose a model family and size
    let model_choice = ModelOptions::Qwen3(Qwen3Size::Size0_6B);

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

### Sentiment Analysis (ModernBERT Finetune)

```rust
use transformers::pipelines::sentiment_analysis_pipeline::{SentimentAnalysisPipelineBuilder, SentimentModernBertSize};
use anyhow::Result;

fn main() -> Result<()> {
    // 1. Choose a model size (Base or Large)
    let size = SentimentModernBertSize::Base;

    // 2. Build the pipeline
    let pipeline = SentimentAnalysisPipelineBuilder::new(size).build()?;

    // 3. Analyze sentiment
    let sentence = "I love using Rust for my projects!";
    let sentiment = pipeline.predict(sentence)?;

    println!("Text: {}", sentence);
    println!("Predicted Sentiment: {}", sentiment); // Should predict positive sentiment
    Ok(())
}
```

## Supported Models & Pipelines

**Text Generation**:

- Gemma3: `1B`, `4B`, `12B`, `27B`
- Phi4: `14B`
- Qwen3: `0.6B`, `1.7B`, `4B`, `8B`, `14B`, `32B`

**Fill-Mask**:

- ModernBERT: `Base`, `Large` (using `answerdotai/ModernBERT-base` or `-large`)

**Sentiment Analysis**:

- ModernBert Multilingual Sentiment Finetune: `Base`, `Large` (using `clapAI/modernBERT-base-multilingual-sentiment` or `-large`)

## Future Plans

- Add more model families and sizes
- Support additional pipelines (summarization, classification)
- Improve performance and error handling

## Credits

A special thanks to [Diaconu Radu-Mihai](https://github.com/radudiaconu0/) for transferring the `transformers` crate name on crates.io
