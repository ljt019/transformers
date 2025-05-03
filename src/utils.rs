use crate::pipelines::text_generation_pipeline::LargeLanguageModel;
use candle_core::quantized::gguf_file;
use candle_core::{CudaDevice, Device, Tensor};
use candle_transformers::generation::{LogitsProcessor, Sampling};
use candle_transformers::models::quantized_gemma3;
use candle_transformers::models::quantized_phi3;
use candle_transformers::models::quantized_qwen3;
use hf_hub;
use std::cell::RefCell;
use std::fs::File;
use tokenizers::Tokenizer;

#[derive(Debug, Clone)]
pub struct HfConfig {
    pub tokenizer_repo: String,
    pub tokenizer_filename: String,
    pub model_repo: String,
    pub model_filename: String,
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

/// Generation parameters for language models.
#[derive(Debug, Clone)]
pub struct GenerationParams {
    pub temperature: f64,
    pub repeat_penalty: f32,
    pub repeat_last_n: usize,
    pub seed: u64,
    pub use_flash_attn: bool,
}

impl GenerationParams {
    pub fn new(
        temperature: f64,
        repeat_penalty: f32,
        repeat_last_n: usize,
        seed: u64,
        use_flash_attn: bool,
    ) -> Self {
        Self {
            temperature,
            repeat_penalty,
            repeat_last_n,
            seed,
            use_flash_attn,
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

/// Loads Gemma3 GGUF model weights.
pub fn load_gemma3_model_weights(
    device: &Device,
    hf_config: &HfConfig,
) -> anyhow::Result<quantized_gemma3::ModelWeights> {
    let model_path = {
        let api = hf_hub::api::sync::Api::new()?;
        let repo = hf_config.model_repo.clone();
        let api = api.model(repo.to_string());
        api.get(hf_config.model_filename.as_str())?
    };
    let mut file = File::open(&model_path)?;
    let content = gguf_file::Content::read(&mut file).map_err(|e| e.with_path(model_path))?;
    quantized_gemma3::ModelWeights::from_gguf(content, &mut file, device)
        .map_err(anyhow::Error::from)
}

/// Loads Phi3 GGUF model weights (used for Phi-4).
pub fn load_phi3_model_weights(
    device: &Device,
    hf_config: &HfConfig,
    use_flash_attn: bool,
) -> anyhow::Result<quantized_phi3::ModelWeights> {
    let model_path = {
        let api = hf_hub::api::sync::Api::new()?;
        let repo = hf_config.model_repo.clone();
        let api = api.model(repo.to_string());
        api.get(hf_config.model_filename.as_str())?
    };
    let mut file = File::open(&model_path)?;
    let content = gguf_file::Content::read(&mut file).map_err(|e| e.with_path(model_path))?;
    quantized_phi3::ModelWeights::from_gguf(use_flash_attn, content, &mut file, device)
        .map_err(anyhow::Error::from)
}

/// Loads Qwen3 GGUF model weights (used for qwen3_quantized)
pub fn load_qwen3_model_weights(
    device: &Device,
    hf_config: &HfConfig,
) -> anyhow::Result<quantized_qwen3::ModelWeights> {
    let model_path = {
        let api = hf_hub::api::sync::Api::new()?;
        let repo = hf_config.model_repo.clone();
        let api = api.model(repo.to_string());
        api.get(hf_config.model_filename.as_str())?
    };

    let mut file = File::open(&model_path)?;
    let content = gguf_file::Content::read(&mut file).map_err(|e| e.with_path(model_path))?;
    quantized_qwen3::ModelWeights::from_gguf(content, &mut file, device)
        .map_err(anyhow::Error::from)
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

/// Initializes a LogitsProcessor based on sampling parameters.
pub fn initialize_logits_processor(params: &GenerationParams, seed: u64) -> LogitsProcessor {
    let sampling = if params.temperature <= 0. {
        Sampling::ArgMax
    } else {
        Sampling::All {
            temperature: params.temperature,
        }
    };
    LogitsProcessor::from_sampling(seed, sampling)
}

/// Trait for quantized models that support the forward pass required for generation.
pub trait QuantizedModelWeights {
    fn forward(&mut self, xs: &Tensor, start_pos: usize) -> candle_core::Result<Tensor>;
}

impl QuantizedModelWeights for quantized_gemma3::ModelWeights {
    fn forward(&mut self, xs: &Tensor, start_pos: usize) -> candle_core::Result<Tensor> {
        self.forward(xs, start_pos)
    }
}

impl QuantizedModelWeights for quantized_phi3::ModelWeights {
    fn forward(&mut self, xs: &Tensor, start_pos: usize) -> candle_core::Result<Tensor> {
        self.forward(xs, start_pos)
    }
}

/// Helper to initialize quantized model weights with logging.
pub fn init_quantized<M, F>(model_name: &str, load_fn: F) -> anyhow::Result<RefCell<M>>
where
    F: FnOnce() -> anyhow::Result<M>,
{
    println!("Loading {} model (this might take a while)...", model_name);
    let weights = load_fn()?;
    println!("{} model loaded successfully.", model_name);
    Ok(RefCell::new(weights))
}

/// Generic function to generate text using a quantized model.
pub fn generate_quantized_text<M: QuantizedModelWeights>(
    model_weights: &mut M,
    device: &Device,
    tokenizer: &Tokenizer,
    prompt_tokens: &[u32],
    eos_token_str: &str,
    params: &GenerationParams,
    max_length: usize,
) -> anyhow::Result<String> {
    let prompt_len = prompt_tokens.len();
    let mut logits_processor = initialize_logits_processor(params, params.seed);
    let mut all_generated_tokens: Vec<u32> = Vec::with_capacity(max_length);

    let input = Tensor::new(prompt_tokens, device)?.unsqueeze(0)?;
    let logits = model_weights.forward(&input, 0)?;
    let logits = logits.squeeze(0)?;
    let mut next_token = logits_processor.sample(&logits)?;
    all_generated_tokens.push(next_token);

    let eos_token_id = *tokenizer
        .get_vocab(true)
        .get(eos_token_str)
        .ok_or_else(|| anyhow::anyhow!("EOS token '{}' not found in vocabulary", eos_token_str))?;

    for index in 0..max_length {
        if next_token == eos_token_id {
            break;
        }

        let context_size = prompt_len + index;
        let input = Tensor::new(&[next_token], device)?.unsqueeze(0)?;
        let logits = model_weights.forward(&input, context_size)?;
        let logits = logits.squeeze(0)?;

        let start_at = all_generated_tokens
            .len()
            .saturating_sub(params.repeat_last_n);
        let penalty_context = &all_generated_tokens[start_at..];

        let logits = if params.repeat_penalty <= 1. || penalty_context.is_empty() {
            logits
        } else {
            candle_transformers::utils::apply_repeat_penalty(
                &logits,
                params.repeat_penalty,
                penalty_context,
            )?
        };

        next_token = logits_processor.sample(&logits)?;

        if next_token == eos_token_id {
            break;
        }
        all_generated_tokens.push(next_token);
    }

    let generated_text = tokenizer
        .decode(&all_generated_tokens, true)
        .map_err(|e| anyhow::anyhow!("Failed to decode generated tokens: {}", e))?;

    Ok(generated_text)
}

#[allow(dead_code)]
/// Generic container for quantized models, holding configuration and model-specific details.
#[derive(Clone)] // Clone might be tricky if M is not Clone, but RefCell<M> is. Let's see.
pub struct QuantizedModel<M: QuantizedModelWeights> {
    pub name: &'static str,
    pub weights: RefCell<M>,
    pub format_prompt: fn(&str) -> String,
    pub eos_token: &'static str,
    pub config: ModelConfig,
}

// Implement the core LLM trait for the generic quantized model container.
impl<M: QuantizedModelWeights> LargeLanguageModel for QuantizedModel<M> {
    fn prompt_model(
        &self,
        tokenizer: &Tokenizer,
        prompt: &str,
        max_length: usize,
    ) -> anyhow::Result<String> {
        let formatted_prompt = (self.format_prompt)(prompt);
        let tokens = tokenizer
            .encode(formatted_prompt, true) // Use the formatted prompt
            .map_err(anyhow::Error::msg)?;
        let prompt_tokens = tokens.get_ids();

        generate_quantized_text(
            &mut *self.weights.borrow_mut(),
            &self.config.device,
            tokenizer,
            prompt_tokens,
            self.eos_token, // Use the EOS token from the struct
            &self.config.params,
            max_length,
        )
    }
}

#[derive(Clone)]
pub struct ModelConfig {
    pub device: Device,
    pub hf_config: HfConfig,
    pub params: GenerationParams,
}

impl ModelConfig {
    pub fn new(params: GenerationParams, hf_config: HfConfig) -> anyhow::Result<Self> {
        let device = load_device()?;
        Ok(Self {
            device,
            hf_config,
            params,
        })
    }
}
