use super::loaders::GgufModelLoader;
use candle_core::Device;

use super::load_device;
use super::GenerationParams;

#[derive(Clone)]
pub struct ModelConfig {
    pub device: Device,
    pub model_loader: GgufModelLoader,
    pub params: GenerationParams,
}

impl ModelConfig {
    pub fn new(params: GenerationParams, model_loader: GgufModelLoader) -> anyhow::Result<Self> {
        let device = load_device()?;
        Ok(Self {
            device,
            model_loader,
            params,
        })
    }
}
