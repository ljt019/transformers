# transformers v0.0.7

<!-- CI / Workflow Badges -->
[<img alt="crates.io" src="https://img.shields.io/crates/v/transformers.svg?style=for-the-badge&color=fc8d62&logo=rust" height="19">](https://crates.io/crates/transformers)
[<img alt="docs.rs" src="https://img.shields.io/badge/docs.rs-transformers-66c2a5?style=for-the-badge&labelColor=555555&logo=docs.rs" height="19">](https://docs.rs/transformers)
![Build](https://github.com/ljt019/transformers/actions/workflows/build_and_release.yaml/badge.svg?branch=main)

> [!warning]
> ***This crate is under active development. APIs may change as features are still being added, and things tweaked.***

Transformers provides a simple, intuitive interface for Rust developers who want to work with Large Language Models locally, powered by the [Candle](https://github.com/huggingface/candle) crate. It offers an API inspired by Python's [Transformers](https://huggingface.co/docs/transformers), tailored for Rust developers.

## Supported Models & Pipelines

**Text Generation**:

- [Gemma3](https://huggingface.co/google/gemma-3-27b-it)
  - 1B
  - 4B
  - 12B
  - 27B
- [Phi4](https://huggingface.co/microsoft/phi-4)
  - 14B
- [Qwen3](https://huggingface.co/Qwen/Qwen3-32B)
  - 0.6B
  - 1.7B
  - 4B
  - 8B
  - 14B
  - 32B

**Fill-Mask**:

- [ModernBERT](https://huggingface.co/answerdotai/ModernBERT-base)
  - Base
  - Large

**Sentiment Analysis**:

- [ModernBERT-multilingual-sentiment](https://huggingface.co/clapAI/modernBERT-base-multilingual-sentiment)
  - Base
  - Large

**Zero-Shot Classification**:

- [ModernBERT-zeroshot](https://huggingface.co/MoritzLaurer/ModernBERT-base-zeroshot-v2.0)
  - Base
  - Large

## Installation

```cmd
cargo add transformers
```

## Usage

At this point in development the only real way to interact with the models is through the given pipelines, I plan to eventually provide you with a simple interface to work with the models directly if you would like.

Inference will be quite slow at the moment, this is mostly due to not using the CUDA feature when compiling candle. I will be working on integrating this smoothly in future updates for much faster inference.

Some examples of how to use the existing pipelines:

### Text Generation

There are two ways to generate text: by providing a simple prompt string, or by providing a list of messages for chat-like interactions.

#### Using `prompt_completion`

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

    // 3. Generate text from a prompt
    let prompt = "Explain the concept of Large Language Models in simple terms.";
    let max_tokens = 100;

    let generated = pipeline.prompt_completion(prompt, max_tokens).unwrap();
    println!("{}", generated);
}
```

#### Using `message_completion` with the `Message` Type

For more conversational interactions, you can use the `message_completion` method, which takes a vector of `Message` structs.

The `Message` struct represents a single message in a chat and has a `role` (system, user, or assistant) and `content`. You can create messages using:

- `Message::system(content: &str)`: For system prompts.
- `Message::user(content: &str)`: For user prompts.
- `Message::assistant(content: &str)`: For model responses.

```rust
use transformers::pipelines::text_generation_pipeline::{
    TextGenerationPipelineBuilder, ModelOptions, Phi4Size,
};
use transformers::Message; // Import the Message type

fn main() -> anyhow::Result<()> {
    // 1. Choose a model family and size
    let model_choice = ModelOptions::Phi4(Phi4Size::Size14B); // Example with a different model

    // 2. Build the pipeline
    let pipeline = TextGenerationPipelineBuilder::new(model_choice)
        .temperature(0.8)
        .build()?;

    // 3. Create a sequence of messages
    let messages = vec![
        Message::system("You are a friendly and helpful AI assistant."),
        Message::user("What is the capital of France?"),
    ];
    let max_tokens = 50;

    // 4. Generate a completion based on the messages
    let generated_response = pipeline.message_completion(messages, max_tokens)?;
    println!("Assistant: {}", generated_response);

    // You can continue the conversation by adding the assistant's response
    // and a new user message to the messages vector.

    Ok(())
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

## Future Plans

- Add more model families and sizes
- Support additional pipelines (summarization, classification)
- Improve performance and error handling

## Credits

A special thanks to [Diaconu Radu-Mihai](https://github.com/radudiaconu0/) for transferring the `transformers` crate name on crates.io
