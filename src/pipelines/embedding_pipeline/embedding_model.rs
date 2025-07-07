use tokenizers::Tokenizer;

/// Trait for embedding models used in the embedding pipeline.
pub trait EmbeddingModel {
    type Options: std::fmt::Debug + Clone;

    fn new(options: Self::Options, device: candle_core::Device) -> anyhow::Result<Self>
    where
        Self: Sized;

    fn embed(&self, tokenizer: &Tokenizer, text: &str) -> anyhow::Result<Vec<f32>>;

    fn get_tokenizer(options: Self::Options) -> anyhow::Result<Tokenizer>;

    fn device(&self) -> &candle_core::Device;
}
