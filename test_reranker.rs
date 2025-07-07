use transformers::models::implementations::qwen3_reranker::Qwen3RerankSize;
use transformers::pipelines::reranker_pipeline::RerankPipelineBuilder;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Test that all sizes are available
    let sizes = vec![
        Qwen3RerankSize::Size0_6B,
        Qwen3RerankSize::Size4B,
        Qwen3RerankSize::Size8B,
    ];
    
    for size in sizes {
        println!("Testing size: {}", size);
        
        // Test the model configuration
        let (repo_id, file_name) = size.to_id();
        println!("  Repository: {}", repo_id);
        println!("  File name: {}", file_name);
        
        // Test cache key generation
        let cache_key = size.cache_key();
        println!("  Cache key: {}", cache_key);
        
        // Test pipeline builder creation (don't actually build to avoid downloading)
        let builder = RerankPipelineBuilder::qwen3(size);
        println!("  Pipeline builder created successfully");
        
        println!();
    }
    
    println!("All tests passed! ✓");
    Ok(())
}