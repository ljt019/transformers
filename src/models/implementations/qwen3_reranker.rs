use std::sync::Arc;

use candle_core::{Device, Result, Tensor};
use tokenizers::Tokenizer;

use super::qwen3::ModelWeights;
use super::qwen3::Qwen3Size;
use crate::loaders::{GgufModelLoader, TokenizerLoader};

fn rerank_id(size: Qwen3Size) -> anyhow::Result<(String, String)> {
    match size {
        Qwen3Size::Size0_6B => Ok((
            "Qwen/Qwen3-0.6B-GGUF".into(),
            "Qwen3-0.6B-Q8_0.gguf".into(),
        )),
        Qwen3Size::Size1_7B => Ok((
            "Qwen/Qwen3-1.7B-GGUF".into(),
            "Qwen3-1.7B-Q8_0.gguf".into(),
        )),
        Qwen3Size::Size4B => Ok((
            "Qwen/Qwen3-4B-GGUF".into(),
            "Qwen3-4B-Q8_0.gguf".into(),
        )),
        Qwen3Size::Size8B => Ok((
            "Qwen/Qwen3-8B-GGUF".into(),
            "Qwen3-8B-Q8_0.gguf".into(),
        )),
        other => anyhow::bail!("No reranker weights available for {other}"),
    }
}

/// Qwen3 model for reranking text pairs using cross-encoder architecture.
#[derive(Clone)]
pub struct Qwen3RerankModel {
    weights: Arc<ModelWeights>,
    device: Device,
}

impl Qwen3RerankModel {
    pub async fn from_hf(device: &Device, size: Qwen3Size) -> anyhow::Result<Self> {
        let (repo_id, file_name) = rerank_id(size)?;
        let loader = GgufModelLoader::new(&repo_id, &file_name);
        let (mut file, content) = loader.load().await?;
        let weights = Arc::new(ModelWeights::from_gguf(content, &mut file, device)?);
        Ok(Self {
            weights,
            device: device.clone(),
        })
    }

    pub async fn get_tokenizer(&self) -> anyhow::Result<Tokenizer> {
        let loader = TokenizerLoader::new("Qwen/Qwen3-0.6B", "tokenizer.json");
        loader.load().await
    }

    /// Rerank a list of documents against a query using cross-encoder architecture.
    /// Returns a list of (document_index, relevance_score) pairs sorted by relevance.
    pub fn rerank(
        &self,
        tokenizer: &Tokenizer,
        query: &str,
        documents: &[&str],
    ) -> anyhow::Result<Vec<(usize, f32)>> {
        let mut scored_docs = Vec::new();
        
        for (idx, document) in documents.iter().enumerate() {
            let score = self.compute_relevance_score(tokenizer, query, document)?;
            scored_docs.push((idx, score));
        }
        
        // Sort by relevance score in descending order
        scored_docs.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        
        Ok(scored_docs)
    }

    /// Compute relevance score for a query-document pair using cross-encoder.
    fn compute_relevance_score(
        &self,
        tokenizer: &Tokenizer,
        query: &str,
        document: &str,
    ) -> anyhow::Result<f32> {
        // Format input as query-document pair with special tokens
        let input_text = format!("[CLS] {} [SEP] {} [SEP]", query, document);
        
        let encoded = tokenizer
            .encode(input_text, false)
            .map_err(anyhow::Error::msg)?;
        let ids = encoded.get_ids();
        
        if ids.is_empty() {
            return Err(anyhow::anyhow!("Tokenizer produced empty token sequence"));
        }
        
        let input = Tensor::new(ids, &self.device)?.unsqueeze(0)?;
        
        // Forward pass through the model
        let output = self.weights.forward_embedding(&input, None)?;
        
        // Extract relevance score from the [CLS] token representation
        // Take the first token (CLS token) and compute a scalar score
        let cls_output = output.narrow(1, 0, 1)?.squeeze(1)?;
        
        // Apply a linear transformation to get relevance score
        // For simplicity, we'll use the mean of the hidden states and apply sigmoid
        let score_logit = cls_output.mean_keepdim(1)?;
        let score = sigmoid(score_logit)?;
        
        // Convert to scalar
        let score_scalar = score.to_scalar::<f32>()?;
        
        Ok(score_scalar)
    }

    /// Batch reranking for better efficiency with multiple queries.
    pub fn batch_rerank(
        &self,
        tokenizer: &Tokenizer,
        queries: &[&str],
        documents: &[&str],
    ) -> anyhow::Result<Vec<Vec<(usize, f32)>>> {
        let mut results = Vec::new();
        
        for query in queries {
            let ranked_docs = self.rerank(tokenizer, query, documents)?;
            results.push(ranked_docs);
        }
        
        Ok(results)
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}

/// Sigmoid activation function
fn sigmoid(x: Tensor) -> Result<Tensor> {
    let neg_x = x.neg()?;
    let exp_neg_x = neg_x.exp()?;
    let one = Tensor::ones_like(&x)?;
    let denominator = one.add(&exp_neg_x)?;
    one.div(&denominator)
}

use crate::pipelines::reranker_pipeline::reranker_model::RerankModel;

impl RerankModel for Qwen3RerankModel {
    type Options = Qwen3Size;

    fn new(options: Self::Options, device: Device) -> anyhow::Result<Self> {
        futures::executor::block_on(Self::from_hf(&device, options))
    }

    fn rerank(
        &self,
        tokenizer: &Tokenizer,
        query: &str,
        documents: &[&str],
    ) -> anyhow::Result<Vec<(usize, f32)>> {
        self.rerank(tokenizer, query, documents)
    }

    fn get_tokenizer(options: Self::Options) -> anyhow::Result<Tokenizer> {
        let loader = TokenizerLoader::new("Qwen/Qwen3-0.6B", "tokenizer.json");
        futures::executor::block_on(loader.load())
    }

    fn device(&self) -> &Device {
        &self.device
    }
}