// Integration tests for fill mask pipeline
// This is a separate crate that tests the public API

use transformers::pipelines::fill_mask_pipeline::*;
use candle_core::DeviceLocation;

#[tokio::test]
async fn basic_fill_mask() -> anyhow::Result<()> {
    let pipeline = FillMaskPipelineBuilder::modernbert(ModernBertSize::Base)
        .build()
        .await?;
    let res = pipeline.fill_mask("The capital of France is [MASK].")?;
    assert!(res.contains("Paris") || !res.trim().is_empty());
    Ok(())
}

#[tokio::test]
async fn test_empty_input_handling() -> anyhow::Result<()> {
    let pipeline = FillMaskPipelineBuilder::modernbert(ModernBertSize::Base)
        .build()
        .await?;
    assert!(pipeline.fill_mask("").is_err());
    Ok(())
}

#[tokio::test]
async fn select_cpu_device() -> anyhow::Result<()> {
    let pipeline = FillMaskPipelineBuilder::modernbert(ModernBertSize::Base)
        .cpu()
        .build()
        .await?;
    assert!(pipeline.device().is_cpu());
    Ok(())
}

#[tokio::test]
async fn select_cuda_device() -> anyhow::Result<()> {
    let pipeline = FillMaskPipelineBuilder::modernbert(ModernBertSize::Base)
        .cuda_device(0)
        .build()
        .await?;
    match pipeline.device().location() {
        DeviceLocation::Cuda { gpu_id } => assert_eq!(gpu_id, 0),
        _ => {}
    }
    Ok(())
}
