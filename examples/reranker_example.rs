use anyhow::Result;
use transformers::pipelines::reranker_pipeline::RerankPipelineBuilder;
use transformers::models::implementations::qwen3_reranker::Qwen3RerankSize;

#[tokio::main]
async fn main() -> Result<()> {
    // Create a reranker pipeline using Qwen3-Reranker-0.6B
    let pipeline = RerankPipelineBuilder::qwen3(Qwen3RerankSize::Size0_6B)
        .cpu()
        .build()
        .await?;

    // Example query and documents
    let query = "What is machine learning?";
    let documents = vec![
        "Machine learning is a subset of artificial intelligence that focuses on algorithms.",
        "Cooking recipes often require precise measurements and timing.",
        "Supervised learning uses labeled data to train models.",
        "The weather today is sunny and warm.",
        "Deep learning is a type of machine learning using neural networks.",
    ];

    // Rerank the documents
    let ranked_results = pipeline.rerank(query, &documents)?;

    println!("Query: {}", query);
    println!("\nRanked documents:");
    for (i, (doc_idx, score)) in ranked_results.iter().enumerate() {
        println!("{}. [Score: {:.4}] {}", i + 1, score, documents[*doc_idx]);
    }

    // Get top-3 results
    let top_3 = pipeline.rerank_top_k(query, &documents, 3)?;
    println!("\nTop 3 results:");
    for (i, (doc_idx, score)) in top_3.iter().enumerate() {
        println!("{}. [Score: {:.4}] {}", i + 1, score, documents[*doc_idx]);
    }

    Ok(())
}