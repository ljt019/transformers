// Integration tests for zero shot classification pipeline
// This is a separate crate that tests the public API

use transformers::pipelines::zero_shot_classification_pipeline::*;

#[test]
fn basic_zero_shot_classification() -> anyhow::Result<()> {
    let pipeline =
        ZeroShotClassificationPipelineBuilder::modernbert(ModernBertSize::Base).build()?;
    let labels = ["politics", "sports"];
    let res = pipeline.predict("The election results were surprising", &labels)?;
    assert_eq!(res.len(), 2);
    Ok(())
}
