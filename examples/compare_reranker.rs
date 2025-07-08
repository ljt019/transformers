use anyhow::Result;
use std::time::Instant;

#[tokio::main]
async fn main() -> Result<()> {
    let documents = vec![
        "Machine learning is a subset of artificial intelligence.",
        "Mathematics is the study of numbers and patterns.",
    ];
    let query = "How do neural networks work?";

    println!("=== Testing Direct Model Usage (like minimal example) ===");
    let start = Instant::now();
    
    // Direct usage like minimal example
    use candle_core::Device;
    use transformers::models::implementations::qwen3_reranker::{Qwen3RerankModel, Qwen3RerankSize};
    
    let mut model = Qwen3RerankModel::from_hf(&Device::Cpu, Qwen3RerankSize::Size0_6B).await?;
    let tokenizer = model.get_tokenizer().await?;
    println!("Model loading time: {:?}", start.elapsed());
    
    let start = Instant::now();
    let results = model.rerank_documents(&tokenizer, query, &documents)?;
    println!("Direct rerank time: {:?}", start.elapsed());
    for res in &results {
        println!("  Doc {} - Score: {:.4}", res.index, res.score);
    }

    println!("\n=== Testing Pipeline Usage ===");
    let start = Instant::now();
    
    use transformers::pipelines::reranker_pipeline::*;
    use transformers::pipelines::utils::{BasePipelineBuilder, DeviceSelectable};
    
    let rerank_pipe = RerankPipelineBuilder::qwen3(Qwen3RerankSize::Size0_6B)
        .cpu()
        .build()
        .await?;
    println!("Pipeline build time: {:?}", start.elapsed());
    
    let start = Instant::now();
    let results2 = rerank_pipe.rerank(query, &documents).await?;
    println!("Pipeline rerank time: {:?}", start.elapsed());
    for res in &results2 {
        println!("  Doc {} - Score: {:.4}", res.index, res.score);
    }

    Ok(())
}