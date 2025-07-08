//! Model and tokenizer loading utilities for Hugging Face Hub integration.
//!
//! This module provides loaders for downloading and loading various AI model components
//! from Hugging Face Hub, including:
//! - Model weight files (GGUF format)
//! - Tokenizers (JSON format)  
//! - Generation configuration files
//!
//! ## Main Types
//!
//! - [`HfLoader`] - Generic Hugging Face file loader with retry logic
//! - [`TokenizerLoader`] - Loads tokenizers from Hugging Face repositories
//! - [`GenerationConfigLoader`] - Loads generation configuration files
//! - [`GgufModelLoader`] - Loads GGUF format model weight files
//!
//! ## Usage Example
//!
//! ```rust,no_run
//! use transformers::loaders::{TokenizerLoader, GgufModelLoader};
//!
//! // Load a tokenizer
//! let tokenizer_loader = TokenizerLoader::new("microsoft/DialoGPT-small", "tokenizer.json");
//! let tokenizer = tokenizer_loader.load().await?;
//!
//! // Load model weights
//! let model_loader = GgufModelLoader::new("microsoft/DialoGPT-small", "model.gguf");
//! let (file, content) = model_loader.load().await?;
//! # Ok::<(), anyhow::Error>(())
//! ```
//!
//! All loaders include built-in retry logic to handle temporary network issues
//! and Hugging Face Hub lock acquisition failures.

use crate::core::GenerationConfig;
use serde::Deserialize;
use hf_hub::api::tokio::Api as HfApi;
use std::path::PathBuf;
use tokenizers::Tokenizer;

#[derive(Debug, Clone)]
pub struct HfLoader {
    pub repo: String,
    pub filename: String,
}

impl HfLoader {
    pub fn new(repo: &str, filename: &str) -> Self {
        Self {
            repo: repo.into(),
            filename: filename.into(),
        }
    }

    pub async fn load(&self) -> anyhow::Result<PathBuf> {
        let hf_api = hf_hub::api::tokio::ApiBuilder::new()
            .with_chunk_size(None)
            .build()?;
        let hf_repo = self.repo.clone();
        let hf_api = hf_api.model(hf_repo);

        // Retry logic for lock acquisition failures
        let max_retries = 3;
        let mut last_error = None;

        for attempt in 0..max_retries {
            match hf_api.get(self.filename.as_str()).await {
                Ok(path) => return Ok(path),
                Err(e) => {
                    let error_msg = e.to_string();
                    if error_msg.contains("Lock acquisition failed") && attempt < max_retries - 1 {
                        // Wait before retrying, with exponential backoff
                        let wait_time = std::time::Duration::from_millis(100 * (1 << attempt));
                        tokio::time::sleep(wait_time).await;
                        last_error = Some(e);
                        continue;
                    }
                    return Err(e.into());
                }
            }
        }

        // If we exhausted all retries, return the last error
        Err(last_error.unwrap().into())
    }
}

#[derive(Clone)]
pub struct TokenizerLoader {
    pub tokenizer_file_loader: HfLoader,
}

impl TokenizerLoader {
    pub fn new(repo: &str, filename: &str) -> Self {
        let tokenizer_file_loader = HfLoader::new(repo, filename);

        Self {
            tokenizer_file_loader,
        }
    }

    pub async fn load(&self) -> anyhow::Result<Tokenizer> {
        let tokenizer_file_path = self.tokenizer_file_loader.load().await?;

        let tokenizer =
            tokenizers::Tokenizer::from_file(tokenizer_file_path).map_err(anyhow::Error::msg)?;

        Ok(tokenizer)
    }
}


pub struct GenerationConfigLoader {
    pub generation_config_file_loader: HfLoader,
}

#[derive(Deserialize)]
struct RawGenerationConfig {
    temperature: Option<f64>,
    top_p: Option<f64>,
    top_k: Option<u64>,
    min_p: Option<f64>,
    #[serde(alias = "repetition_penalty", alias = "repeat_penalty")]
    repeat_penalty: Option<f32>,
    repeat_last_n: Option<usize>,
    #[serde(alias = "eos_token_id", alias = "eos_token_ids")]
    eos_token_ids: Option<serde_json::Value>,
}

impl GenerationConfigLoader {
    pub fn new(repo: &str, filename: &str) -> Self {
        let generation_config_file_loader = HfLoader::new(repo, filename);

        Self {
            generation_config_file_loader,
        }
    }

    pub async fn load(&self) -> anyhow::Result<GenerationConfig> {
        let generation_config_file_path = self
            .generation_config_file_loader
            .load()
            .await?;

        let generation_config_content =
            std::fs::read_to_string(generation_config_file_path)?;

        let raw: RawGenerationConfig = serde_json::from_str(&generation_config_content)?;

        let eos_token_ids = match raw.eos_token_ids {
            Some(serde_json::Value::Number(n)) => vec![n
                .as_u64()
                .ok_or_else(|| anyhow::anyhow!("Invalid EOS token ID"))?],
            Some(serde_json::Value::Array(arr)) => arr
                .into_iter()
                .map(|v| v
                    .as_u64()
                    .ok_or_else(|| anyhow::anyhow!("Invalid EOS token ID in array")))
                .collect::<Result<Vec<_>, _>>()?,
            _ => Vec::new(),
        };

        Ok(GenerationConfig {
            temperature: raw.temperature,
            top_p: raw.top_p,
            top_k: raw.top_k,
            min_p: raw.min_p,
            repeat_penalty: raw.repeat_penalty,
            repeat_last_n: raw.repeat_last_n,
            eos_token_ids,
        })
    }
}

#[derive(Clone)]
pub struct GgufModelLoader {
    pub model_file_loader: HfLoader,
}

impl GgufModelLoader {
    pub fn new(model_repo: &str, model_filename: &str) -> Self {
        let model_file_loader = HfLoader::new(model_repo, model_filename);

        Self { model_file_loader }
    }

    pub async fn load(
        &self,
    ) -> anyhow::Result<(std::fs::File, candle_core::quantized::gguf_file::Content)> {
        let model_file_path = self.model_file_loader.load().await?;

        let mut file = std::fs::File::open(&model_file_path)?;
        let file_content = candle_core::quantized::gguf_file::Content::read(&mut file)
            .map_err(|e| e.with_path(model_file_path))?;

        Ok((file, file_content))
    }
}
