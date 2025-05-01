# rustformers

My goal with this crate is to make a really basic and simple interface to use popular local LLMs. Candle makes working with LLMs in Rust do-able but it's quite tedious to use, I hope to make an API that falls more in line with python's transformers but in a Rust fashion.

I probably will start with implementing a basic text-pipeline with Qwen3, candle so far only supports the non-moe models so I will likely start with those.