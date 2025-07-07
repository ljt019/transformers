use anyhow::Result;
use transformers::models::implementations::qwen3_reranker::Qwen3RerankSize;
use transformers::pipelines::reranker_pipeline::RerankPipelineBuilder;

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
    let ranked_results = pipeline.rerank(query, &documents).await?;

    println!("Query: {}", query);
    println!("\nRanked documents:");
    for (i, result) in ranked_results.iter().enumerate() {
        println!("{}. [Score: {:.4}] {}", i + 1, result.score, documents[result.index]);
    }

    // Get top-3 results
    let top_3 = pipeline.rerank_top_k(query, &documents, 3).await?;
    println!("\nTop 3 results:");
    for (i, result) in top_3.iter().enumerate() {
        println!("{}. [Score: {:.4}] {}", i + 1, result.score, documents[result.index]);
    }

    Ok(())
}
