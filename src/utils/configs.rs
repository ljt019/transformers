use candle_core::Device;

use super::load_device;
use crate::models::raw::generation::GenerationParams;

#[derive(Clone)]
pub struct ModelConfig {
    pub device: Device,
    pub params: GenerationParams,
}

impl ModelConfig {
    pub fn new(params: GenerationParams) -> anyhow::Result<Self> {
        let device = load_device()?;
        Ok(Self { device, params })
    }
}
