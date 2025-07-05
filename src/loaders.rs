use hf_hub::api::sync::Api as HfApi;
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

    pub fn load(&self) -> anyhow::Result<PathBuf> {
        let file_path = {
            let hf_api = HfApi::new()?;
            let hf_repo = self.repo.clone();

            let hf_api = hf_api.model(hf_repo);

            hf_api.get(self.filename.as_str())?
        };

        return Ok(file_path);
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

    pub fn load(&self) -> anyhow::Result<Tokenizer> {
        let tokenizer_file_path = self.tokenizer_file_loader.load()?;

        let tokenizer =
            tokenizers::Tokenizer::from_file(tokenizer_file_path).map_err(anyhow::Error::msg)?;

        Ok(tokenizer)
    }
}

#[derive(Clone)]
pub struct GenerationConfig {
    pub temperature: Option<f64>,
    pub top_p: Option<f64>,
    pub top_k: Option<u64>,
    pub min_p: Option<f64>,
    pub repeat_penalty: Option<f32>,
    pub repeat_last_n: Option<usize>,
    pub eos_token_ids: Vec<u64>,
}

pub struct GenerationConfigLoader {
    pub generation_config_file_loader: HfLoader,
}

impl GenerationConfigLoader {
    pub fn new(repo: &str, filename: &str) -> Self {
        let generation_config_file_loader = HfLoader::new(repo, filename);

        Self {
            generation_config_file_loader,
        }
    }

    pub fn load(&self) -> anyhow::Result<GenerationConfig> {
        let generation_config_file_path = self.generation_config_file_loader.load()?;

        let generation_config_content = std::fs::read_to_string(generation_config_file_path)?;

        let config_json: serde_json::Value = serde_json::from_str(&generation_config_content)?;

        // All fields are optional to handle inconsistent JSON files
        let temperature = config_json.get("temperature").and_then(|v| v.as_f64());
        let top_p = config_json.get("top_p").and_then(|v| v.as_f64());
        let top_k = config_json.get("top_k").and_then(|v| v.as_u64());
        let min_p = config_json.get("min_p").and_then(|v| v.as_f64());
        let repeat_penalty = config_json
            .get("repetition_penalty")
            .or_else(|| config_json.get("repeat_penalty"))
            .and_then(|v| v.as_f64())
            .map(|v| v as f32);
        let repeat_last_n = config_json
            .get("repeat_last_n")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize);

        // Handle both single EOS token ID and array of EOS token IDs
        let eos_token_ids = match config_json.get("eos_token_id") {
            Some(serde_json::Value::Number(n)) => vec![n.as_u64().expect("Invalid EOS token ID")],
            Some(serde_json::Value::Array(arr)) => arr
                .iter()
                .map(|v| v.as_u64().expect("Invalid EOS token ID in array"))
                .collect(),
            _ => {
                // Try alternative field names
                match config_json.get("eos_token_ids") {
                    Some(serde_json::Value::Array(arr)) => arr
                        .iter()
                        .map(|v| v.as_u64().expect("Invalid EOS token ID in array"))
                        .collect(),
                    _ => vec![], // Empty vec instead of panic
                }
            }
        };

        Ok(GenerationConfig {
            temperature,
            top_p,
            top_k,
            min_p,
            repeat_penalty,
            repeat_last_n,
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

    pub fn load(
        &self,
    ) -> anyhow::Result<(std::fs::File, candle_core::quantized::gguf_file::Content)> {
        let model_file_path = self.model_file_loader.load()?;

        let mut file = std::fs::File::open(&model_file_path)?;
        let file_content = candle_core::quantized::gguf_file::Content::read(&mut file)
            .map_err(|e| e.with_path(model_file_path))?;

        return Ok((file, file_content));
    }
}
