use transformers::pipelines::embedding_pipeline::*;
use transformers::pipelines::utils::BasePipelineBuilder;

#[tokio::test]
async fn batch_embedding() -> anyhow::Result<()> {
    let pipeline = EmbeddingPipelineBuilder::qwen3(Qwen3EmbeddingSize::Size0_6B)
        .build()
        .await?;
    let inputs = ["hello", "world"];
    let embs = pipeline.embed_batch(&inputs).await?;
    assert_eq!(embs.len(), inputs.len());
    for emb in embs {
        assert!(!emb.is_empty());
    }
    Ok(())
}
