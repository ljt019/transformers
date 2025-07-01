use tokenizers::Tokenizer;

pub trait ZeroShotClassificationModel {
    type Options: std::fmt::Debug + Clone;

    fn new(options: Self::Options) -> anyhow::Result<Self>
    where
        Self: Sized;

    fn predict(
        &self,
        tokenizer: &Tokenizer,
        text: &str,
        candidate_labels: &[&str],
    ) -> anyhow::Result<Vec<(String, f32)>>;

    fn get_tokenizer(options: Self::Options) -> anyhow::Result<Tokenizer>;
}
