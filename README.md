# rustformers

My goal with this crate is to make a really basic and simple interface to use popular local LLMs. [Candle](https://github.com/huggingface/candle) is a Rust-based framework for machine learning, making it possible to work with LLMs in Rust, though it can be tedious to use. I hope to create an API that aligns more closely with [python's transformers](https://huggingface.co/docs/transformers), a popular Python library for working with transformer models, but in a Rust fashion.

I probably will start with implementing a basic text-pipeline with Qwen3, candle so far only supports the non-moe models so I will likely start with those.