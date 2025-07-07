use tokenizers::Tokenizer;
use candle_core::Device;

/// Trait for reranking models.
pub trait RerankModel {
    type Options: std::fmt::Debug + Clone;
    
    fn new(options: Self::Options, device: Device) -> anyhow::Result<Self>
    where
        Self: Sized;
    
    fn rerank(
        &self,
        tokenizer: &Tokenizer,
        query: &str,
        documents: &[&str],
    ) -> anyhow::Result<Vec<(usize, f32)>>;
    
    fn get_tokenizer(options: Self::Options) -> anyhow::Result<Tokenizer>;
    
    fn device(&self) -> &Device;
}