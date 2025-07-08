use transformers::pipelines::embedding_pipeline::*;
use transformers::pipelines::utils::DeviceSelectable;

#[tokio::test]
async fn basic_embedding() -> anyhow::Result<()> {
    let pipeline = EmbeddingPipelineBuilder::qwen3(Qwen3EmbeddingSize::Size0_6B)
        .build()
        .await?;
    let emb = pipeline.embed("hello world").await?;
    assert!(!emb.is_empty());

    let doc_embs = pipeline.embed_batch(&["hello there", "goodbye"]).await?;
    let top = EmbeddingPipeline::<Qwen3EmbeddingModel>::top_k(&emb, &doc_embs, 1);
    assert_eq!(top.len(), 1);
    Ok(())
}

#[tokio::test]
async fn select_cpu_device() -> anyhow::Result<()> {
    let pipeline = EmbeddingPipelineBuilder::qwen3(Qwen3EmbeddingSize::Size0_6B)
        .cpu()
        .build()
        .await?;
    assert!(pipeline.device().is_cpu());
    Ok(())
}
