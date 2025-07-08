use tokenizers::Tokenizer;

pub trait SentimentAnalysisModel {
    type Options: std::fmt::Debug + Clone;

    async fn new(options: Self::Options, device: candle_core::Device) -> anyhow::Result<Self>
    where
        Self: Sized;

    fn predict(&self, tokenizer: &Tokenizer, text: &str) -> anyhow::Result<String>;

    async fn get_tokenizer(options: Self::Options) -> anyhow::Result<Tokenizer>;

    fn device(&self) -> &candle_core::Device;
}
