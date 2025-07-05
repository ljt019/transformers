use transformers::pipelines::text_generation_pipeline::*;

#[tool]
fn echo(msg: String) -> String {
    msg
}

#[test]
fn unregister_and_clear() -> anyhow::Result<()> {
    let pipeline = TextGenerationPipelineBuilder::qwen3(Qwen3Size::Size0_6B)
        .seed(0)
        .max_len(20)
        .build()?;

    pipeline.register_tools(tools![echo])?;
    assert_eq!(pipeline.registered_tools().len(), 1);

    pipeline.unregister_tool("echo")?;
    assert!(pipeline.registered_tools().is_empty());

    pipeline.register_tool(echo)?;
    assert_eq!(pipeline.registered_tools().len(), 1);

    pipeline.clear_tools()?;
    assert!(pipeline.registered_tools().is_empty());
    Ok(())
}
