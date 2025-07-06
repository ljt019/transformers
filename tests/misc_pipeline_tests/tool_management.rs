use transformers::pipelines::text_generation::*;

#[tool]
fn echo(msg: String) -> String {
    msg
}

#[tokio::test]
async fn unregister_and_clear() -> anyhow::Result<()> {
    let pipeline = TextGenerationPipelineBuilder::qwen3(Qwen3Size::Size0_6B)
        .seed(0)
        .max_len(20)
        .build()
        .await?;

    pipeline.register_tools(tools![echo]).await?;
    assert_eq!(pipeline.registered_tools().await.len(), 1);

    pipeline.unregister_tool("echo").await?;
    assert!(pipeline.registered_tools().await.is_empty());

    pipeline.register_tools(tools![echo]).await?;
    assert_eq!(pipeline.registered_tools().await.len(), 1);

    pipeline.clear_tools().await?;
    assert!(pipeline.registered_tools().await.is_empty());
    Ok(())
}
