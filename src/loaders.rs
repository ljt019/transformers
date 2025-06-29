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
    pub temperature: f64,
    pub top_p: f64,
    pub top_k: u64,
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

        let temperature = config_json["temperature"].as_f64().unwrap_or(0.7);
        let top_p = config_json["top_p"].as_f64().unwrap_or(1.0);
        let top_k = config_json["top_k"].as_u64().unwrap_or(40);

        // Handle both single EOS token ID and array of EOS token IDs
        let eos_token_ids = match &config_json["eos_token_id"] {
            serde_json::Value::Number(n) => vec![n.as_u64().expect("Invalid EOS token ID")],
            serde_json::Value::Array(arr) => arr
                .iter()
                .map(|v| v.as_u64().expect("Invalid EOS token ID in array"))
                .collect(),
            _ => panic!("CAN'T FIND EOS IN TOKENIZER CONFIG"),
        };

        Ok(GenerationConfig {
            temperature,
            top_p,
            top_k,
            eos_token_ids,
        })
    }
}

pub struct TokenizerConfig {
    pub chat_template: String,
    pub special_tokens: Vec<String>,
}

pub struct TokenizerConfigLoader {
    pub tokenizer_config_file_loader: HfLoader,
}

impl TokenizerConfigLoader {
    pub fn new(repo: &str, filename: &str) -> Self {
        let tokenizer_config_file_loader = HfLoader::new(repo, filename);

        Self {
            tokenizer_config_file_loader,
        }
    }

    pub fn load(&self) -> anyhow::Result<TokenizerConfig> {
        let tokenizer_config_file_path = self.tokenizer_config_file_loader.load()?;

        let file = std::fs::File::open(tokenizer_config_file_path)?;
        let config_json: serde_json::Value = serde_json::from_reader(file)?;

        let chat_template = config_json["chat_template"].as_str().unwrap_or("");

        let special_tokens = config_json["additional_special_tokens"]
            .as_array()
            .expect("Couldn't find special tokens")
            .iter()
            .map(|token| {
                token
                    .as_str()
                    .expect("Couldn't find special tokens")
                    .to_string()
            })
            .collect::<Vec<String>>();

        Ok(TokenizerConfig {
            chat_template: chat_template.to_string(),
            special_tokens,
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
