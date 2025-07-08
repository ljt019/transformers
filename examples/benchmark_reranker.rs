use std::time::Instant;
use transformers::models::implementations::Qwen3RerankSize;
use transformers::pipelines::reranker_pipeline::*;
use transformers::pipelines::utils::BasePipelineBuilder;
use transformers::pipelines::utils::DeviceSelectable;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Test documents
    let documents = vec![
        "Machine learning is a subset of artificial intelligence.",
        "Mathematics is the study of numbers and patterns.", 
        "Physics is the fundamental science of the universe.",
        "Cooking is both an art and a science.",
    ];
    let query = "How do neural networks work?";

    println!("=== Reranker Performance Benchmark ===\n");

    // Benchmark 1: Pipeline approach
    println!("1. Testing pipeline approach...");
    let start = Instant::now();
    
    let rerank_pipe = RerankPipelineBuilder::qwen3(Qwen3RerankSize::Size0_6B)
        .cpu()
        .build()
        .await?;
    
    let pipeline_build_time = start.elapsed();
    println!("   Pipeline build time: {:?}", pipeline_build_time);

    // Warm up
    let _ = rerank_pipe
        .rerank(query, &documents.iter().map(|d| *d).collect::<Vec<_>>())
        .await?;

    let start = Instant::now();
    for _ in 0..5 {
        let _ = rerank_pipe
            .rerank(query, &documents.iter().map(|d| *d).collect::<Vec<_>>())
            .await?;
    }
    let pipeline_rerank_time = start.elapsed();
    println!("   Pipeline rerank time (5 iterations): {:?}", pipeline_rerank_time);
    println!("   Average per iteration: {:?}", pipeline_rerank_time / 5);

    // Benchmark 2: Direct model approach (sync in spawn_blocking)
    println!("\n2. Testing direct model approach (in spawn_blocking)...");
    let start = Instant::now();
    
    tokio::task::spawn_blocking(move || -> anyhow::Result<()> {
        use transformers::models::implementations::qwen3_reranker::{Qwen3RerankModel, Qwen3RerankSize};
        use transformers::pipelines::reranker_pipeline::model::RerankModel;
        use candle_core::Device;
        
        // Load model directly
        let start_load = Instant::now();
        let mut model = futures::executor::block_on(
            Qwen3RerankModel::from_hf(&Device::Cpu, Qwen3RerankSize::Size0_6B)
        )?;
        let tokenizer = futures::executor::block_on(model.get_tokenizer())?;
        let load_time = start_load.elapsed();
        println!("   Model load time: {:?}", load_time);
        
        // Warm up
        let _ = model.rerank_documents(&tokenizer, query, &documents.iter().map(|d| *d).collect::<Vec<_>>())?;
        
        // Benchmark
        let start_rerank = Instant::now();
        for _ in 0..5 {
            let _ = model.rerank_documents(&tokenizer, query, &documents.iter().map(|d| *d).collect::<Vec<_>>())?;
        }
        let rerank_time = start_rerank.elapsed();
        println!("   Direct rerank time (5 iterations): {:?}", rerank_time);
        println!("   Average per iteration: {:?}", rerank_time / 5);
        
        Ok(())
    }).await??;
    
    let direct_total_time = start.elapsed();
    println!("   Total direct time: {:?}", direct_total_time);

    Ok(())
}