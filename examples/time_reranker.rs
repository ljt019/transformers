use anyhow::Result;
use transformers::models::implementations::Qwen3RerankSize;
use transformers::pipelines::reranker_pipeline::*;
use transformers::pipelines::utils::BasePipelineBuilder;
use transformers::pipelines::utils::DeviceSelectable;
use std::time::Instant;

#[tokio::main]
async fn main() -> Result<()> {
    println!("=== Timing Reranker Pipeline ===\n");

    // Time pipeline creation
    let start = Instant::now();
    let rerank_pipe = RerankPipelineBuilder::qwen3(Qwen3RerankSize::Size0_6B)
        .cpu()
        .build()
        .await?;
    println!("Pipeline creation: {:?}", start.elapsed());

    // Simple test documents
    let documents = vec![
        "Machine learning is a subset of artificial intelligence.",
        "Mathematics is the study of numbers and patterns.",
    ];
    let query = "How do neural networks work?";

    // Time first rerank (includes any lazy initialization)
    let start = Instant::now();
    let results = rerank_pipe
        .rerank(query, &documents.iter().map(|d| *d).collect::<Vec<_>>())
        .await?;
    println!("First rerank: {:?}", start.elapsed());
    println!("Results: {:?}", results);

    // Time second rerank (should be faster)
    let start = Instant::now();
    let results = rerank_pipe
        .rerank(query, &documents.iter().map(|d| *d).collect::<Vec<_>>())
        .await?;
    println!("Second rerank: {:?}", start.elapsed());

    // Time 10 reranks
    let start = Instant::now();
    for _ in 0..10 {
        let _ = rerank_pipe
            .rerank(query, &documents.iter().map(|d| *d).collect::<Vec<_>>())
            .await?;
    }
    let elapsed = start.elapsed();
    println!("10 reranks: {:?} (avg: {:?})", elapsed, elapsed / 10);

    Ok(())
}