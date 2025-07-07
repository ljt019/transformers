use super::reranker_model::RerankModel;
use tokenizers::Tokenizer;

pub struct RerankPipeline<M: RerankModel> {
    pub(crate) model: M,
    pub(crate) tokenizer: Tokenizer,
}

impl<M: RerankModel> RerankPipeline<M> {
    /// Rerank a list of documents against a query.
    /// Returns a list of (document_index, relevance_score) pairs sorted by relevance.
    pub fn rerank(&self, query: &str, documents: &[&str]) -> anyhow::Result<Vec<(usize, f32)>> {
        self.model.rerank(&self.tokenizer, query, documents)
    }

    /// Batch reranking for multiple queries against the same set of documents.
    pub fn batch_rerank(&self, queries: &[&str], documents: &[&str]) -> anyhow::Result<Vec<Vec<(usize, f32)>>> {
        let mut results = Vec::new();
        
        for query in queries {
            let ranked_docs = self.rerank(query, documents)?;
            results.push(ranked_docs);
        }
        
        Ok(results)
    }

    /// Get the top-k most relevant documents for a query.
    pub fn rerank_top_k(&self, query: &str, documents: &[&str], k: usize) -> anyhow::Result<Vec<(usize, f32)>> {
        let mut ranked = self.rerank(query, documents)?;
        ranked.truncate(k);
        Ok(ranked)
    }

    /// Get relevance scores for all documents without sorting.
    pub fn get_scores(&self, query: &str, documents: &[&str]) -> anyhow::Result<Vec<f32>> {
        let ranked = self.model.rerank(&self.tokenizer, query, documents)?;
        let mut scores = vec![0.0f32; documents.len()];
        
        for (idx, score) in ranked {
            scores[idx] = score;
        }
        
        Ok(scores)
    }

    pub fn device(&self) -> &candle_core::Device {
        self.model.device()
    }
}