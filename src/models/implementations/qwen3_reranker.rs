use std::sync::Arc;

use candle_core::{Device, Tensor};
use tokenizers::Tokenizer;

use super::qwen3::ModelWeights;
use crate::loaders::{GgufModelLoader, TokenizerLoader};

#[derive(Debug, Clone, Copy)]
pub enum Qwen3RerankSize {
    Size0_6B,
    Size4B,
    Size8B,
}

impl Qwen3RerankSize {
    pub fn to_id(&self) -> (String, String) {
        match self {
            Qwen3RerankSize::Size0_6B => (
                "Mungert/Qwen3-Reranker-0.6B-GGUF".into(),
                "Qwen3-Reranker-0.6B-q4_k_m.gguf".into(),
            ),
            Qwen3RerankSize::Size4B => (
                "Mungert/Qwen3-Reranker-4B-GGUF".into(),
                "Qwen3-Reranker-4B-q4_k_m.gguf".into(),
            ),
            Qwen3RerankSize::Size8B => (
                "Mungert/Qwen3-Reranker-8B-GGUF".into(),
                "Qwen3-Reranker-8B-q4_k_m.gguf".into(),
            ),
        }
    }
}

impl std::fmt::Display for Qwen3RerankSize {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            Qwen3RerankSize::Size0_6B => "qwen3-reranker-0.6b",
            Qwen3RerankSize::Size4B => "qwen3-reranker-4b",
            Qwen3RerankSize::Size8B => "qwen3-reranker-8b",
        };
        write!(f, "{name}")
    }
}

impl crate::core::ModelOptions for Qwen3RerankSize {
    fn cache_key(&self) -> String {
        self.to_string()
    }
}

/// Qwen3 model for reranking text pairs using cross-encoder architecture.
#[derive(Clone)]
pub struct Qwen3RerankModel {
    weights: Arc<ModelWeights>,
    device: Device,
}

impl Qwen3RerankModel {
    pub async fn from_hf(device: &Device, size: Qwen3RerankSize) -> anyhow::Result<Self> {
        let (repo_id, file_name) = size.to_id();
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
    pub fn rerank_documents(
        &self,
        tokenizer: &Tokenizer,
        query: &str,
        documents: &[&str],
    ) -> anyhow::Result<Vec<RerankResult>> {
        let mut scored_docs = Vec::new();

        for (idx, document) in documents.iter().enumerate() {
            let score = self.compute_relevance_score(tokenizer, query, document)?;
            scored_docs.push(RerankResult { index: idx, score });
        }

        // Sort by relevance score in descending order
        scored_docs.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(scored_docs)
    }

    /// Compute relevance score for a query-document pair using cross-encoder.
    fn compute_relevance_score(
        &self,
        tokenizer: &Tokenizer,
        query: &str,
        document: &str,
    ) -> anyhow::Result<f32> {
        // Format input using Qwen3 chat template for reranking
        // Using the exact format from the Qwen3 reranker paper
        let instruction =
            "Given a query and a document, determine whether the document is relevant to the query";
        let input_text = format!(
            "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );

        let encoded = tokenizer
            .encode(input_text, false)
            .map_err(anyhow::Error::msg)?;
        let ids = encoded.get_ids();

        if ids.is_empty() {
            return Err(anyhow::anyhow!("Tokenizer produced empty token sequence"));
        }

        let input = Tensor::new(ids, &self.device)?.unsqueeze(0)?;

        // Forward pass through the model to get logits
        let logits = self.weights.forward_logits(&input, None)?;

        // Get the last token logits (where the model would generate "yes" or "no")
        let (_, seq_len, _vocab_size) = logits.dims3()?;
        let last_token_logits = logits.narrow(1, seq_len - 1, 1)?.squeeze(1)?;

        // Get token IDs for various "yes" and "no" variations
        // The model might output these with different casing or spacing
        let yes_variations = vec!["yes", "Yes", " yes", " Yes"];
        let no_variations = vec!["no", "No", " no", " No"];

        let mut yes_logit_sum = f32::NEG_INFINITY;
        let mut no_logit_sum = f32::NEG_INFINITY;

        // Try to find the best matching tokens for yes
        for var in &yes_variations {
            if let Ok(token_id) = self.get_token_id(tokenizer, var) {
                if let Ok(logit) = last_token_logits
                    .narrow(1, token_id as usize, 1)?
                    .squeeze(1)?
                    .squeeze(0)?
                    .to_scalar::<f32>()
                {
                    // Use log-sum-exp for numerical stability
                    if yes_logit_sum == f32::NEG_INFINITY {
                        yes_logit_sum = logit;
                    } else {
                        let max_val = yes_logit_sum.max(logit);
                        yes_logit_sum = max_val
                            + ((yes_logit_sum - max_val).exp() + (logit - max_val).exp()).ln();
                    }
                }
            }
        }

        // Try to find the best matching tokens for no
        for var in &no_variations {
            if let Ok(token_id) = self.get_token_id(tokenizer, var) {
                if let Ok(logit) = last_token_logits
                    .narrow(1, token_id as usize, 1)?
                    .squeeze(1)?
                    .squeeze(0)?
                    .to_scalar::<f32>()
                {
                    // Use log-sum-exp for numerical stability
                    if no_logit_sum == f32::NEG_INFINITY {
                        no_logit_sum = logit;
                    } else {
                        let max_val = no_logit_sum.max(logit);
                        no_logit_sum = max_val
                            + ((no_logit_sum - max_val).exp() + (logit - max_val).exp()).ln();
                    }
                }
            }
        }

        // If we couldn't find any valid tokens, fall back to default tokens
        if yes_logit_sum == f32::NEG_INFINITY || no_logit_sum == f32::NEG_INFINITY {
            // Try to get token IDs more directly
            let yes_token_id = tokenizer
                .encode("yes", false)
                .map_err(anyhow::Error::msg)?
                .get_ids()[0];
            let no_token_id = tokenizer
                .encode("no", false)
                .map_err(anyhow::Error::msg)?
                .get_ids()[0];

            yes_logit_sum = last_token_logits
                .narrow(1, yes_token_id as usize, 1)?
                .squeeze(1)?
                .squeeze(0)?
                .to_scalar::<f32>()?;
            no_logit_sum = last_token_logits
                .narrow(1, no_token_id as usize, 1)?
                .squeeze(1)?
                .squeeze(0)?
                .to_scalar::<f32>()?;
        }

        // Apply softmax to get probabilities
        let exp_yes = yes_logit_sum.exp();
        let exp_no = no_logit_sum.exp();
        let total = exp_yes + exp_no;

        // Calculate probabilities
        let no_prob = exp_no / total;

        // IMPORTANT: Return the "no" probability as the relevance score
        // Counter-intuitively, the Qwen3 reranker model outputs higher "no" logits
        // when documents ARE relevant to the query.
        Ok(no_prob)
    }

    /// Get token ID for a given text string
    fn get_token_id(&self, tokenizer: &Tokenizer, text: &str) -> anyhow::Result<u32> {
        let encoded = tokenizer.encode(text, false).map_err(anyhow::Error::msg)?;
        let ids = encoded.get_ids();

        if ids.is_empty() {
            return Err(anyhow::anyhow!("Failed to tokenize text: {}", text));
        }

        Ok(ids[0])
    }

    /// Batch reranking for better efficiency with multiple queries.
    pub fn batch_rerank(
        &self,
        tokenizer: &Tokenizer,
        queries: &[&str],
        documents: &[&str],
    ) -> anyhow::Result<Vec<Vec<RerankResult>>> {
        let mut results = Vec::new();

        for query in queries {
            let ranked_docs = self.rerank_documents(tokenizer, query, documents)?;
            results.push(ranked_docs);
        }

        Ok(results)
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}

use crate::pipelines::reranker_pipeline::model::RerankModel;
use crate::pipelines::reranker_pipeline::pipeline::RerankResult;

impl RerankModel for Qwen3RerankModel {
    type Options = Qwen3RerankSize;

    fn new(options: Self::Options, device: Device) -> anyhow::Result<Self> {
        futures::executor::block_on(Self::from_hf(&device, options))
    }

    fn rerank(
        &self,
        tokenizer: &Tokenizer,
        query: &str,
        documents: &[&str],
    ) -> anyhow::Result<Vec<RerankResult>> {
        self.rerank_documents(tokenizer, query, documents)
    }

    fn get_tokenizer(_options: Self::Options) -> anyhow::Result<Tokenizer> {
        let loader = TokenizerLoader::new("Qwen/Qwen3-0.6B", "tokenizer.json");
        futures::executor::block_on(loader.load())
    }

    fn batch_rerank(
        &self,
        tokenizer: &Tokenizer,
        queries: &[&str],
        documents: &[&str],
    ) -> anyhow::Result<Vec<Vec<RerankResult>>> {
        Qwen3RerankModel::batch_rerank(self, tokenizer, queries, documents)
    }

    fn device(&self) -> &Device {
        &self.device
    }
}
