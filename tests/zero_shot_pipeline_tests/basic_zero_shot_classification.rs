// Integration tests for zero shot classification pipeline
// This is a separate crate that tests the public API

use transformers::pipelines::zero_shot_classification_pipeline::*;
use candle_core::DeviceLocation;

#[test]
fn basic_zero_shot_classification() -> anyhow::Result<()> {
    let pipeline =
        ZeroShotClassificationPipelineBuilder::modernbert(ModernBertSize::Base).build()?;
    let labels = ["politics", "sports"];
    let res = pipeline.predict("The election results were surprising", &labels)?;
    assert_eq!(res.len(), 2);
    Ok(())
}

#[test]
fn select_cuda_device() -> anyhow::Result<()> {
    let pipeline = ZeroShotClassificationPipelineBuilder::modernbert(ModernBertSize::Base)
        .cuda_device(0)
        .build()?;
    match pipeline.device().location() {
        DeviceLocation::Cuda { gpu_id } => assert_eq!(gpu_id, 0),
        _ => {}
    }
    Ok(())
}

#[test]
fn select_cpu_device() -> anyhow::Result<()> {
    let pipeline = ZeroShotClassificationPipelineBuilder::modernbert(ModernBertSize::Base)
        .cpu()
        .build()?;
    assert!(pipeline.device().is_cpu());
    Ok(())
}
