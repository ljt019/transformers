use anyhow::Result;
use transformers::pipelines::embedding_pipeline::*;

#[tokio::main]
async fn main() -> Result<()> {
    println!("Building pipeline...");

    let pipeline = EmbeddingPipelineBuilder::qwen3(Qwen3EmbeddingSize::Size0_6B)
        .build()
        .await?;

    println!("Pipeline built successfully.");

    let emb_hello_world = pipeline.embed("hello world").await?;

    let emb_typical_first_program =
        pipeline.embed("Typical first program programmers learn to write").await?;

    let emb_random_string = pipeline.embed("I like firetrucks").await?;

    // Figure out which embedding is closest to emb_hello_world
    let embeddings = vec![
        ("typical first program", &emb_typical_first_program),
        ("random string", &emb_random_string),
    ];

    let scores: Vec<Vec<f32>> = vec![emb_typical_first_program.clone(), emb_random_string.clone()]
        .into_iter()
        .collect::<Vec<Vec<f32>>>();
    let top = EmbeddingPipeline::<Qwen3EmbeddingModel>::top_k(&emb_hello_world, &scores, 1);
    let closest_to_hello_world = top.first().map(|(i, _)| embeddings[*i].0);

    println!("\n=== Embedding Similarity Results ===");
    println!("Query: \"hello world\"");
    println!("Closest match: {:?}", closest_to_hello_world);

    Ok(())
}
