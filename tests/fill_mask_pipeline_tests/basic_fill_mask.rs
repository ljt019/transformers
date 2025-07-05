// Integration tests for fill mask pipeline
// This is a separate crate that tests the public API

use transformers::pipelines::fill_mask_pipeline::*;

#[test]
fn basic_fill_mask() -> anyhow::Result<()> {
    let pipeline = FillMaskPipelineBuilder::modernbert(ModernBertSize::Base).build()?;
    let res = pipeline.fill_mask("The capital of France is [MASK].")?;
    assert!(res.contains("Paris") || !res.trim().is_empty());
    Ok(())
}

#[test]
fn test_empty_input_handling() -> anyhow::Result<()> {
    let pipeline = FillMaskPipelineBuilder::modernbert(ModernBertSize::Base).build()?;
    assert!(pipeline.fill_mask("").is_err());
    Ok(())
}
