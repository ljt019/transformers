# transformers v0.0.6

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
>
> - Zero-Shot Classification:
>   - ModernBERT NLI Finetune (Base, Large)

Transformers provides a simple, idiomatic Rust interface for running local large language models (LLMs) via the [Candle](https://github.com/huggingface/candle) framework. It offers an API inspired by Python's [transformers](https://huggingface.co/docs/transformers), tailored for Rust developers.

## Installation

```cmd
cargo add transformers
```

## Usage

At this point in development the only real way to interact with the models is through the given pipelines, I plan to eventually provide you with a simple interface to work with the models directly if you would like.

Inference will be quite slow at the moment, this is mostly due to not using the CUDA feature when compiling candle. I will be working on integrating this smoothly in future updates for much faster inference.

Some examples of how to use the existing pipelines:

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

### Zero-Shot Classification (ModernBERT NLI Finetune)

```rust
use transformers::pipelines::zero_shot_classification_pipeline::{
    ZeroShotClassificationPipelineBuilder, ZeroShotModernBertSize,
};
use anyhow::Result;

fn main() -> Result<()> {
    // 1. Choose a model size (Base or Large)
    let size = ZeroShotModernBertSize::Base;

    // 2. Build the pipeline
    let pipeline = ZeroShotClassificationPipelineBuilder::new(size).build()?;

    // 3. Classify text using candidate labels
    let text = "The Federal Reserve raised interest rates.";
    let candidate_labels = &["economics", "politics", "sports", "technology"];
    
    let results = pipeline.predict(text, candidate_labels)?;
    
    println!("Text: {}", text);
    for (label, score) in results {
        println!("- {}: {:.4}", label, score);
    }
    // Example output:
    // - economics: 0.8721
    // - politics: 0.1134
    // - technology: 0.0098
    // - sports: 0.0047
    
    Ok(())
}
```

## Supported Models & Pipelines

**Text Generation**:

- Gemma3: `1B`, `4B`, `12B`, `27B`
- Phi4: `14B`
- Qwen3: `0.6B`, `1.7B`, `4B`, `8B`, `14B`, `32B`

**Fill-Mask**:

- ModernBERT: `Base`, `Large` (using `answerdotai/ModernBERT-base`)

**Sentiment Analysis**:

- ModernBert Multilingual Sentiment Finetune: `Base`, `Large` (using `clapAI/modernBERT-base-multilingual-sentiment`)

**Zero-Shot Classification**:

- ModernBERT Zero-Shot NLI Finetune: `Base`, `Large` (using `MoritzLaurer/ModernBERT-base-zeroshot-v2.0`)

## Future Plans

- Add more model families and sizes
- Support additional pipelines (summarization, classification)
- Improve performance and error handling

## Credits

A special thanks to [Diaconu Radu-Mihai](https://github.com/radudiaconu0/) for transferring the `transformers` crate name on crates.io
