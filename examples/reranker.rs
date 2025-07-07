use anyhow::Result;
use transformers::models::implementations::qwen3::Qwen3Size;
use transformers::pipelines::embedding_pipeline::*;
use transformers::pipelines::reranker_pipeline::*;

fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|v| v * v).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|v| v * v).sum::<f32>().sqrt();
    dot / (norm_a * norm_b)
}

#[tokio::main]
async fn main() -> Result<()> {
    /* ---------- build the two pipelines ---------- */
    let embed_pipe = EmbeddingPipelineBuilder::qwen3(Qwen3Size::Size0_6B)
        .cpu() // or .cuda(0) if you have GPU
        .build()
        .await?;

    let rerank_pipe = RerankPipelineBuilder::qwen3(Qwen3RerankSize::Size0_6B)
        .cpu()
        .build()
        .await?;

    /* ---------- tiny demo corpus ---------- */
    let documents = vec![
        "Machine learning is a subset of artificial intelligence that focuses on algorithms.",
        "Cooking recipes often require precise measurements and timing.",
        "Supervised learning uses labeled data to train models.",
        "The weather today is sunny and warm.",
        "Deep learning is a type of machine learning using neural networks.",
    ];

    /* ---------- embed the corpus once up front ---------- */
    let doc_embeddings: Vec<Vec<f32>> = documents
        .iter()
        .map(|doc| embed_pipe.embed(doc).expect("embedding failed"))
        .collect();

    /* ---------- incoming query ---------- */
    let query = "What is machine learning?";
    let query_emb = embed_pipe.embed(query)?;

    /* ---------- stage-1: ANN recall ---------- */
    let mut scored: Vec<(usize, f32)> = doc_embeddings
        .iter()
        .enumerate()
        .map(|(i, emb)| (i, cosine_sim(&query_emb, emb)))
        .collect();

    // higher cosine = more similar
    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    // keep the top-k (here k = 3)
    let k = 3;
    let candidate_idxs: Vec<usize> = scored.iter().take(k).map(|(i, _)| *i).collect();
    let candidate_docs: Vec<&str> = candidate_idxs.iter().map(|&i| documents[i]).collect();

    println!("=== Top-{k} after embedding recall ===");
    for (rank, (idx, score)) in scored.iter().take(k).enumerate() {
        println!("{:>2}. [sim {:.4}] {}", rank + 1, score, documents[*idx]);
    }

    /* ---------- stage-2: rerank those candidates ---------- */
    let reranked = rerank_pipe.rerank(query, &candidate_docs).await?;

    println!("\n=== Final ranking from Qwen3-Reranker ===");
    for (rank, res) in reranked.iter().enumerate() {
        println!(
            "{:>2}. [score {:.4}] {}",
            rank + 1,
            res.score,
            candidate_docs[res.index]
        );
    }

    Ok(())
}
