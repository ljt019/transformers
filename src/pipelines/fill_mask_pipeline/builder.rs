use super::model::FillMaskModel;
use super::pipeline::FillMaskPipeline;
use crate::core::{global_cache, ModelOptions};
use crate::pipelines::utils::{build_cache_key, DeviceRequest};

pub struct FillMaskPipelineBuilder<M: FillMaskModel> {
    options: M::Options,
    device_request: DeviceRequest,
}

impl<M: FillMaskModel> FillMaskPipelineBuilder<M> {
    pub fn new(options: M::Options) -> Self {
        Self {
            options,
            device_request: DeviceRequest::Default,
        }
    }

    pub fn cpu(mut self) -> Self {
        self.device_request = DeviceRequest::Cpu;
        self
    }

    pub fn cuda_device(mut self, index: usize) -> Self {
        self.device_request = DeviceRequest::Cuda(index);
        self
    }

    pub fn device(mut self, device: candle_core::Device) -> Self {
        self.device_request = DeviceRequest::Explicit(device);
        self
    }

    pub async fn build(self) -> anyhow::Result<FillMaskPipeline<M>>
    where
        M: Clone + Send + Sync + 'static,
        M::Options: ModelOptions + Clone,
    {
        let device = self.device_request.resolve()?;
        let key = build_cache_key(&self.options, &device);
        let model = global_cache()
            .get_or_create(&key, || M::new(self.options.clone(), device.clone()))
            .await?;
        let tokenizer = M::get_tokenizer(self.options)?;
        Ok(FillMaskPipeline { model, tokenizer })
    }
}

impl FillMaskPipelineBuilder<crate::models::implementations::modernbert::FillMaskModernBertModel> {
    pub fn modernbert(size: crate::models::ModernBertSize) -> Self {
        Self::new(size)
    }
}

