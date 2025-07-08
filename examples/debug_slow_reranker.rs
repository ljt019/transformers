use std::time::Instant;
use transformers::models::implementations::Qwen3RerankSize;
use transformers::pipelines::reranker_pipeline::*;
use transformers::pipelines::utils::BasePipelineBuilder;
use transformers::pipelines::utils::DeviceSelectable;
use std::path::Path;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    println!("=== Debugging Slow Reranker ===\n");

    // Check if corpus exists
    let corpus_path = Path::new("examples").join("reranker").join("corpus");
    println!("Corpus path exists: {}", corpus_path.exists());
    
    // Load documents
    let start = Instant::now();
    let mut documents = Vec::new();
    let mut filenames = Vec::new();
    
    if corpus_path.is_dir() {
        for entry in std::fs::read_dir(&corpus_path)? {
            let entry = entry?;
            if entry.file_type()?.is_file() {
                let content = std::fs::read_to_string(entry.path())?;
                if !content.trim().is_empty() {
                    println!("Loaded: {} ({} bytes)", entry.file_name().to_string_lossy(), content.len());
                    documents.push(content);
                    filenames.push(entry.file_name().to_string_lossy().to_string());
                }
            }
        }
    }
    println!("Document loading time: {:?}", start.elapsed());
    println!("Total documents: {}", documents.len());

    // Build pipeline with detailed timing
    println!("\nBuilding pipeline...");
    let start_total = Instant::now();
    
    let rerank_pipe = RerankPipelineBuilder::qwen3(Qwen3RerankSize::Size0_6B)
        .cpu()
        .build()
        .await?;
    
    let pipeline_time = start_total.elapsed();
    println!("Pipeline build time: {:?}", pipeline_time);

    // Test with single document first
    let query = "How do neural networks and deep learning work?";
    println!("\nTesting with single document...");
    let single_doc = vec![documents[0].as_str()];
    
    let start = Instant::now();
    let result = rerank_pipe.rerank(query, &single_doc).await?;
    let single_time = start.elapsed();
    println!("Single document rerank time: {:?}", single_time);
    println!("Score: {:.4}", result[0].score);

    // Now test with all documents
    println!("\nTesting with all {} documents...", documents.len());
    let doc_refs: Vec<&str> = documents.iter().map(|d| d.as_str()).collect();
    
    let start = Instant::now();
    let results = rerank_pipe.rerank(query, &doc_refs).await?;
    let all_time = start.elapsed();
    println!("All documents rerank time: {:?}", all_time);
    println!("Time per document: {:?}", all_time / documents.len() as u32);

    // Print results
    println!("\nResults:");
    for (i, res) in results.iter().enumerate() {
        println!("{}. {} - Score: {:.4}", i+1, filenames[res.index], res.score);
    }

    // Now let's profile what happens inside the model
    println!("\n=== Profiling Model Internals ===");
    
    // Create a minimal version inline to compare
    let start = Instant::now();
    use candle_core::{Device, Tensor};
    use transformers::loaders::{GgufModelLoader, TokenizerLoader};
    
    let loader = GgufModelLoader::new(
        "Mungert/Qwen3-Reranker-0.6B-GGUF",
        "Qwen3-Reranker-0.6B-q4_k_m.gguf"
    );
    let tok_loader = TokenizerLoader::new("Qwen/Qwen3-0.6B", "tokenizer.json");
    
    println!("Created loaders: {:?}", start.elapsed());
    
    let start = Instant::now();
    let (_file, _content) = loader.load().await?;
    println!("Loaded GGUF file: {:?}", start.elapsed());
    
    let start = Instant::now();
    let tokenizer = tok_loader.load().await?;
    println!("Loaded tokenizer: {:?}", start.elapsed());

    Ok(())
}