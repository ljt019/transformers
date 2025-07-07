use super::reranker_model::RerankModel;
use tokenizers::Tokenizer;

/// Result of reranking a single document.
#[derive(Debug, Clone, Copy)]
pub struct RerankResult {
    /// Index of the document in the input list.
    pub index: usize,
    /// Relevance score as predicted by the model.
    pub score: f32,
}

pub struct RerankPipeline<M: RerankModel> {
    pub(crate) model: M,
    pub(crate) tokenizer: Tokenizer,
}

impl<M: RerankModel> RerankPipeline<M> {
    /// Rerank a list of documents against a query.
    /// Returns a list of (document_index, relevance_score) pairs sorted by relevance.
    pub async fn rerank(&self, query: &str, documents: &[&str]) -> anyhow::Result<Vec<RerankResult>> {
        self.model.rerank(&self.tokenizer, query, documents)
    }

    /// Batch reranking for multiple queries against the same set of documents.
    pub async fn batch_rerank(&self, queries: &[&str], documents: &[&str]) -> anyhow::Result<Vec<Vec<RerankResult>>> {
        self.model.batch_rerank(&self.tokenizer, queries, documents)
    }

    /// Get the top-k most relevant documents for a query.
    pub async fn rerank_top_k(&self, query: &str, documents: &[&str], k: usize) -> anyhow::Result<Vec<RerankResult>> {
        let mut ranked = self.rerank(query, documents).await?;
        ranked.truncate(k);
        Ok(ranked)
    }

    /// Get relevance scores for all documents without sorting.
    pub async fn get_scores(&self, query: &str, documents: &[&str]) -> anyhow::Result<Vec<f32>> {
        let ranked = self.model.rerank(&self.tokenizer, query, documents)?;
        let mut scores = vec![0.0f32; documents.len()];
        for r in ranked {
            scores[r.index] = r.score;
        }
        Ok(scores)
    }

    pub fn device(&self) -> &candle_core::Device {
        self.model.device()
    }
}