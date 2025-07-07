use transformers::pipelines::embedding_pipeline::*;

#[tokio::test]
async fn basic_embedding() -> anyhow::Result<()> {
    let pipeline = EmbeddingPipelineBuilder::new(Qwen3EmbeddingSize::Size0_6B)
        .build()
        .await?;
    let emb = pipeline.embed("hello world")?;
    assert!(!emb.is_empty());
    Ok(())
}

#[tokio::test]
async fn select_cpu_device() -> anyhow::Result<()> {
    let pipeline = EmbeddingPipelineBuilder::new(Qwen3EmbeddingSize::Size0_6B)
        .cpu()
        .build()
        .await?;
    assert!(pipeline.device().is_cpu());
    Ok(())
}
