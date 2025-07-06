use super::sentiment_analysis_model::SentimentAnalysisModel;
use super::sentiment_analysis_pipeline::SentimentAnalysisPipeline;
use crate::core::{global_cache, ModelOptions};

pub struct SentimentAnalysisPipelineBuilder<M: SentimentAnalysisModel> {
    options: M::Options,
    device: Option<candle_core::Device>,
}

impl<M: SentimentAnalysisModel> SentimentAnalysisPipelineBuilder<M> {
    pub fn new(options: M::Options) -> Self {
        Self {
            options,
            device: None,
        }
    }

    pub fn cpu(mut self) -> Self {
        self.device = Some(candle_core::Device::Cpu);
        self
    }

    pub fn cuda_device(mut self, index: usize) -> Self {
        let dev =
            candle_core::Device::new_cuda_with_stream(index).unwrap_or(candle_core::Device::Cpu);
        self.device = Some(dev);
        self
    }

    pub fn device(mut self, device: candle_core::Device) -> Self {
        self.device = Some(device);
        self
    }

    pub async fn build(self) -> anyhow::Result<SentimentAnalysisPipeline<M>>
    where
        M: Clone + Send + Sync + 'static,
        M::Options: ModelOptions + Clone,
    {
        let device = match self.device {
            Some(d) => d,
            None => crate::pipelines::utils::load_device()?,
        };
        let key = format!("{}-{:?}", self.options.cache_key(), device.location());
        let model = global_cache()
            .get_or_create(&key, || M::new(self.options.clone(), device.clone()))
            .await?;
        let tokenizer = M::get_tokenizer(self.options)?;
        Ok(SentimentAnalysisPipeline { model, tokenizer })
    }
}

impl
    SentimentAnalysisPipelineBuilder<
        crate::models::implementations::modernbert::SentimentModernBertModel,
    >
{
    pub fn modernbert(size: crate::models::ModernBertSize) -> Self {
        Self::new(size)
    }
}
