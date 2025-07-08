use anyhow::Result;
use transformers::models::implementations::Qwen3RerankSize;
use transformers::pipelines::reranker_pipeline::*;
use transformers::pipelines::utils::BasePipelineBuilder;
use transformers::pipelines::utils::DeviceSelectable;

use std::fs;
use std::path::Path;

#[tokio::main]
async fn main() -> Result<()> {
    println!("Building pipeline...");

    /* ---------- build the pipeline ---------- */
    let rerank_pipe = RerankPipelineBuilder::qwen3(Qwen3RerankSize::Size4B)
        .cpu()
        .build()
        .await?;

    println!("Pipeline built\n");

    println!("Loading corpus...");

    /* ---------- load corpus (directory of text files) ---------- */
    let corpus_path = Path::new("examples").join("reranker").join("corpus");
    let mut documents: Vec<String> = Vec::new();
    let mut filenames: Vec<String> = Vec::new();

    if corpus_path.is_dir() {
        for entry in fs::read_dir(&corpus_path)? {
            let entry = entry?;
            if entry.file_type()?.is_file() {
                let content = fs::read_to_string(entry.path())?;
                if !content.trim().is_empty() {
                    documents.push(content);
                    filenames.push(entry.file_name().to_string_lossy().to_string());
                }
            }
        }
    }

    println!("📚 Loaded {} documents from corpus\n", documents.len());

    /* ---------- incoming query ---------- */
    let query = "How do neural networks and deep learning work?";
    println!("🔍 Query: \"{}\"", query);

    // Helper function to truncate documents for display
    fn truncate_doc(doc: &str, max_chars: usize) -> String {
        if doc.len() <= max_chars {
            doc.to_string()
        } else {
            format!("{}...", &doc[..max_chars])
        }
    }

    let reranked = rerank_pipe
        .rerank(
            query,
            &documents.iter().map(|d| d.as_str()).collect::<Vec<&str>>(),
        )
        .await?;

    println!("\n=== Final ranking from Qwen3-Reranker ===");
    for (rank, res) in reranked.iter().enumerate() {
        println!(
            "{:>2}. [score {:.4}] {} | {}",
            rank + 1,
            res.score,
            filenames[res.index],
            truncate_doc(&documents[res.index], 80)
        );
    }

    Ok(())
}
