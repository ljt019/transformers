use super::model::ZeroShotClassificationModel;
use super::pipeline::ZeroShotClassificationPipeline;
use crate::core::{global_cache, ModelOptions};
use crate::pipelines::utils::{build_cache_key, DeviceRequest, DeviceSelectable};

pub struct ZeroShotClassificationPipelineBuilder<M: ZeroShotClassificationModel> {
    options: M::Options,
    device_request: DeviceRequest,
}

impl<M: ZeroShotClassificationModel> ZeroShotClassificationPipelineBuilder<M> {
    pub fn new(options: M::Options) -> Self {
        Self {
            options,
            device_request: DeviceRequest::Default,
        }
    }

    pub async fn build(self) -> anyhow::Result<ZeroShotClassificationPipeline<M>>
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
        Ok(ZeroShotClassificationPipeline { model, tokenizer })
    }
}

impl<M: ZeroShotClassificationModel> DeviceSelectable for ZeroShotClassificationPipelineBuilder<M> {
    fn device_request_mut(&mut self) -> &mut DeviceRequest {
        &mut self.device_request
    }
}

impl
    ZeroShotClassificationPipelineBuilder<
        crate::models::implementations::modernbert::ZeroShotModernBertModel,
    >
{
    pub fn modernbert(size: crate::models::ModernBertSize) -> Self {
        Self::new(size)
    }
}

