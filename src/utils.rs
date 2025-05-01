use candle_core::quantized::gguf_file;
use candle_core::{CudaDevice, Device};
use candle_transformers::models::quantized_gemma3::ModelWeights;
use hf_hub;
use std::fs::File;
use std::io::BufReader;
use tokenizers::Tokenizer;

#[derive(Debug, Clone)]
pub struct HfConfig {
    pub tokenizer_repo: String,
    pub tokenizer_filename: String,
    pub model_repo: String,
    pub model_filename: String,
}

impl Default for HfConfig {
    fn default() -> Self {
        Self {
            tokenizer_repo: "google/gemma-3-1b-it".into(),
            tokenizer_filename: "tokenizer.json".into(),
            model_repo: "unsloth/gemma-3-1b-it-GGUF".into(),
            model_filename: "gemma-3-1b-it-Q4_K_M.gguf".into(),
        }
    }
}

impl HfConfig {
    pub fn new(
        tokenizer_repo: &str,
        tokenizer_filename: &str,
        model_repo: &str,
        model_filename: &str,
    ) -> Self {
        Self {
            tokenizer_repo: tokenizer_repo.into(),
            tokenizer_filename: tokenizer_filename.into(),
            model_repo: model_repo.into(),
            model_filename: model_filename.into(),
        }
    }
}

/// Loads a device to be used for the model. Uses CUDA by default, falling back to CPU if CUDA is not available.
pub fn load_device() -> anyhow::Result<Device> {
    match CudaDevice::new_with_stream(0) {
        Ok(cuda_device) => Ok(Device::Cuda(cuda_device)),
        Err(err) => {
            println!("CUDA not available, using CPU: {}", err);
            Ok(Device::Cpu)
        }
    }
}

/// Loads a model onto a given device, downloading it from Hugging Face if necessary.
pub fn load_model(device: &Device, hf_config: &HfConfig) -> anyhow::Result<ModelWeights> {
    let model_path = {
        let api = hf_hub::api::sync::Api::new()?;
        let repo = hf_config.model_repo.clone();
        let api = api.model(repo.to_string());
        api.get(hf_config.model_filename.as_str())?
    };

    let mut reader = BufReader::new(File::open(model_path)?);
    let content = gguf_file::Content::read(&mut reader)?;
    let model = ModelWeights::from_gguf(content, &mut reader, device)?;
    Ok(model)
}

/// Loads a tokenizer from Hugging Face, downloading it if necessary.
pub fn load_tokenizer(hf_config: &HfConfig) -> anyhow::Result<Tokenizer> {
    let tokenizer_path = {
        let api = hf_hub::api::sync::Api::new()?;
        let repo = hf_config.tokenizer_repo.clone();

        let api = api.model(repo.to_string());

        api.get(hf_config.tokenizer_filename.as_str())?
    };

    let tokenizer = tokenizers::Tokenizer::from_file(tokenizer_path).map_err(anyhow::Error::msg)?;

    Ok(tokenizer)
}
