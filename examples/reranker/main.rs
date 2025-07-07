use anyhow::Result;
use transformers::models::implementations::{Qwen3EmbeddingSize, Qwen3RerankSize};
use transformers::pipelines::embedding_pipeline::*;
use transformers::pipelines::reranker_pipeline::*;

use std::fs;
use std::path::Path;

#[tokio::main]
async fn main() -> Result<()> {
    println!("Building pipelines...");

    /* ---------- build the two pipelines ---------- */
    let embed_pipe = EmbeddingPipelineBuilder::qwen3(Qwen3EmbeddingSize::Size0_6B)
        .cpu() // or .cuda(0) if you have GPU
        .build()
        .await?;

    let rerank_pipe = RerankPipelineBuilder::qwen3(Qwen3RerankSize::Size0_6B)
        .cpu()
        .build()
        .await?;

    println!("Pipelines built\n");

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

    // Fallback to a tiny hard-coded corpus if the directory is missing or empty
    if documents.is_empty() {
        documents = vec![
            "Machine learning is a subset of artificial intelligence that focuses on algorithms."
                .to_string(),
            "Cooking recipes often require precise measurements and timing.".to_string(),
            "Supervised learning uses labeled data to train models.".to_string(),
            "The weather today is sunny and warm.".to_string(),
            "Deep learning is a type of machine learning using neural networks.".to_string(),
        ];
        filenames = vec![
            "demo_doc_1".to_string(),
            "demo_doc_2".to_string(),
            "demo_doc_3".to_string(),
            "demo_doc_4".to_string(),
            "demo_doc_5".to_string(),
        ];
        eprintln!("[info] corpus directory not found or empty – using built-in demo corpus");
    }

    println!("📚 Loaded {} documents from corpus\n", documents.len());

    println!("Embedding corpus...");

    /* ---------- embed the corpus once up front ---------- */
    let mut doc_embeddings = Vec::new();
    for doc in &documents {
        doc_embeddings.push(embed_pipe.embed(doc).await.expect("embedding failed"));
    }

    println!("Corpus embedded!");

    /* ---------- incoming query ---------- */
    let query = "How do neural networks and deep learning work?";
    println!("🔍 Query: \"{}\"", query);

    let query_emb = embed_pipe.embed(query).await?;

    // Helper function to truncate documents for display
    fn truncate_doc(doc: &str, max_chars: usize) -> String {
        if doc.len() <= max_chars {
            doc.to_string()
        } else {
            format!("{}...", &doc[..max_chars])
        }
    }

    /* ---------- stage-1: ANN recall ---------- */
    let mut scored: Vec<(usize, f32)> = doc_embeddings
        .iter()
        .enumerate()
        .map(|(i, emb)| (i, EmbeddingPipeline::<Qwen3EmbeddingModel>::cosine_similarity(&query_emb, emb)))
        .collect();

    // higher cosine = more similar
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // keep the top-k (here k = 3)
    let k = 3;
    let candidate_idxs: Vec<usize> = scored.iter().take(k).map(|(i, _)| *i).collect();
    let candidate_docs: Vec<&str> = candidate_idxs
        .iter()
        .map(|&i| documents[i].as_str())
        .collect();

    println!("=== Top-{k} after embedding recall ===");
    for (rank, (idx, score)) in scored.iter().take(k).enumerate() {
        println!(
            "{:>2}. [sim {:.4}] {} | {}",
            rank + 1,
            score,
            filenames[*idx],
            truncate_doc(&documents[*idx], 80)
        );
    }

    /* ---------- stage-2: rerank those candidates ---------- */
    let reranked = rerank_pipe.rerank(query, &candidate_docs).await?;

    println!("\n=== Final ranking from Qwen3-Reranker ===");
    for (rank, res) in reranked.iter().enumerate() {
        println!(
            "{:>2}. [score {:.4}] {} | {}",
            rank + 1,
            res.score,
            filenames[candidate_idxs[res.index]],
            truncate_doc(candidate_docs[res.index], 80)
        );
    }

    Ok(())
}
