use hf_hub::api::sync::Api as HfApi;
use std::path::PathBuf;
use tokenizers::Tokenizer;

use crate::models::raw::models::quantized_gemma3;
use crate::models::raw::models::quantized_phi3;
use crate::models::raw::models::quantized_qwen3;

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
pub enum LoadedGgufModelWeights {
    Phi3(quantized_phi3::ModelWeights),
    Qwen3(quantized_qwen3::ModelWeights),
    Gemma3(quantized_gemma3::ModelWeights),
}

#[derive(Clone)]
pub struct GgufModelLoader {
    pub device: candle_core::Device,
    pub model_file_loader: HfLoader,
}

impl GgufModelLoader {
    pub fn new(device: candle_core::Device, model_repo: &str, model_filename: &str) -> Self {
        let model_file_loader = HfLoader::new(model_repo, model_filename);

        Self {
            device,
            model_file_loader,
        }
    }

    pub fn load(&self) -> anyhow::Result<candle_core::quantized::gguf_file::Content> {
        let model_file_path = self.model_file_loader.load()?;

        let mut file = std::fs::File::open(&model_file_path)?;
        let file_content = candle_core::quantized::gguf_file::Content::read(&mut file)
            .map_err(|e| e.with_path(model_file_path))?;

        return Ok((file, file_content));
    }
}
