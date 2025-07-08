// Integration tests for zero shot classification pipeline
// This is a separate crate that tests the public API

use candle_core::DeviceLocation;
use transformers::pipelines::utils::BasePipelineBuilder;
use transformers::pipelines::utils::DeviceSelectable;
use transformers::pipelines::zero_shot_classification_pipeline::*;

#[tokio::test]
async fn basic_zero_shot_classification() -> anyhow::Result<()> {
    let pipeline = ZeroShotClassificationPipelineBuilder::modernbert(ModernBertSize::Base)
        .build()
        .await?;
    let labels = ["politics", "sports"];
    let res = pipeline.classify("The election results were surprising", &labels)?;
    assert_eq!(res.len(), 2);
    Ok(())
}

#[tokio::test]
async fn select_cuda_device() -> anyhow::Result<()> {
    let pipeline = ZeroShotClassificationPipelineBuilder::modernbert(ModernBertSize::Base)
        .cuda_device(0)
        .build()
        .await?;
    match pipeline.device().location() {
        DeviceLocation::Cuda { gpu_id } => assert_eq!(gpu_id, 0),
        _ => {}
    }
    Ok(())
}

#[tokio::test]
async fn select_cpu_device() -> anyhow::Result<()> {
    let pipeline = ZeroShotClassificationPipelineBuilder::modernbert(ModernBertSize::Base)
        .cpu()
        .build()
        .await?;
    assert!(pipeline.device().is_cpu());
    Ok(())
}
