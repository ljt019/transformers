use tokenizers::Tokenizer;

pub trait FillMaskModel {
    type Options: std::fmt::Debug + Clone;

    fn new(options: Self::Options) -> anyhow::Result<Self>
    where
        Self: Sized;

    fn predict(&self, tokenizer: &Tokenizer, text: &str) -> anyhow::Result<String>;

    fn get_tokenizer(options: Self::Options) -> anyhow::Result<Tokenizer>;
}
