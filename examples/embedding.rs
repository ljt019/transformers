use transformers::pipelines::embedding_pipeline::*;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let pipeline = EmbeddingPipelineBuilder::qwen3(Qwen3EmbeddingSize::Size0_6B)
        .build()
        .await?;

    let emb_hello_world = pipeline.embed("hello world")?;

    let emb_typical_first_program =
        pipeline.embed("Typical first program programmers learn to write")?;

    let emb_random_string = pipeline.embed("I like firetrucks")?;

    // Figure out which embedding is closest to emb_hello_world
    let embeddings = vec![
        ("typical first program", &emb_typical_first_program),
        ("random string", &emb_random_string),
    ];

    let closest_to_hello_world = embeddings.iter().min_by_key(|(_, emb)| {
        // Calculate cosine distance (1 - cosine similarity)
        let dot_product: f32 = emb_hello_world
            .iter()
            .zip(emb.iter())
            .map(|(a, b)| a * b)
            .sum();
        let norm_a: f32 = emb_hello_world.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        let cosine_sim = dot_product / (norm_a * norm_b);
        ((1.0 - cosine_sim) * 1000.0) as u32 // Convert to integer for comparison
    });

    println!("Closest to hello world: {:?}", closest_to_hello_world);

    Ok(())
}
