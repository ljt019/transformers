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

    let start = Instant::now();
    let results = rerank_pipe
        .rerank(query, &documents.iter().map(|d| *d).collect::<Vec<_>>())
        .await?;
    let pipeline_rerank_time = start.elapsed();
    println!("   Pipeline rerank time: {:?}", pipeline_rerank_time);
    println!("   Total pipeline time: {:?}", pipeline_build_time + pipeline_rerank_time);

    // Print results
    println!("\n   Results:");
    for (i, res) in results.iter().enumerate() {
        println!("   {}. Doc {} - Score: {:.4}", i+1, res.index, res.score);
    }

    // Benchmark 2: Direct model approach (sync in spawn_blocking)
    println!("\n2. Testing direct model approach (in spawn_blocking)...");
    let start = Instant::now();
    
    let results2 = tokio::task::spawn_blocking(move || -> anyhow::Result<Vec<(usize, f32)>> {
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
        
        // Rerank
        let start_rerank = Instant::now();
        let results = model.rerank_documents(&tokenizer, query, &documents.iter().map(|d| *d).collect::<Vec<_>>())?;
        let rerank_time = start_rerank.elapsed();
        println!("   Direct rerank time: {:?}", rerank_time);
        
        Ok(results.into_iter().map(|r| (r.index, r.score)).collect())
    }).await??;
    
    let direct_total_time = start.elapsed();
    println!("   Total direct time: {:?}", direct_total_time);
    
    println!("\n   Results:");
    for (i, (idx, score)) in results2.iter().enumerate() {
        println!("   {}. Doc {} - Score: {:.4}", i+1, idx, score);
    }

    // Compare
    println!("\n=== Performance Comparison ===");
    println!("Pipeline approach: {:?}", pipeline_build_time + pipeline_rerank_time);
    println!("Direct approach: {:?}", direct_total_time);
    let speedup = direct_total_time.as_secs_f64() / (pipeline_build_time + pipeline_rerank_time).as_secs_f64();
    println!("Direct is {:.2}x faster", speedup);

    Ok(())
}