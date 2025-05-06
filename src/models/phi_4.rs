use crate::models::raw::models::quantized_phi3;
use crate::utils::configs::ModelConfig;
use crate::utils::loaders::{GgufModelLoader, LoadedGgufModelWeights, TokenizerLoader};
use std::cell::RefCell;

use crate::pipelines::TextGenerationModel;

pub enum Phi4Size {
    Size14B,
}

// Use the generic QuantizedModel with Phi3 weights (for Phi-4) specific types and configuration
pub type QuantizedPhi4Model = crate::utils::QuantizedModel<quantized_phi3::ModelWeights>;

impl QuantizedPhi4Model {
    pub fn new(config: ModelConfig) -> anyhow::Result<Self> {
        // Load weights and tokenizer using the new loader
        let loaded_weights = config.model_loader.load_weights()?;

        // Extract the Phi3 weights from the enum
        let phi3_weights = match loaded_weights {
            LoadedGgufModelWeights::Phi3(weights) => Ok(weights),
            _ => Err(anyhow::anyhow!(
                "Loaded unexpected model type for Phi4, expected Phi3 weights"
            )),
        }?;

        // Wrap the weights in RefCell
        let weights = RefCell::new(phi3_weights);

        // Define the prompt formatter for Phi (identity function)
        let format_prompt = |prompt: &str| -> String { prompt.to_string() };

        Ok(Self {
            name: "Phi4",
            weights,
            format_prompt,
            eos_token: "<|endoftext|>", // EOS token for Phi
            config,
        })
    }
}

impl TextGenerationModel for QuantizedPhi4Model {
    fn load_tokenizer(&self) -> anyhow::Result<tokenizers::Tokenizer> {
        let tokenizer_loader = TokenizerLoader::new("google/gemma-3-1b-it", "tokenizer.json");

        let tokenizer = tokenizer_loader.load_tokenizer()?;

        Ok(tokenizer)
    }

    fn load_model_weights(&self, _size: Phi4Size) -> anyhow::Result<LoadedGgufModelWeights> {
        let (repo, file_name) = ("microsoft/phi-4-gguf", "phi-4-q4.gguf");

        let gguf_loader = GgufModelLoader::new(self.config.device, repo, file_name);

        let (gguf_file, gguf_content) = gguf_loader.load()?;

        let weights = quantized_phi3::ModelWeights::from_gguf(
            gguf_content,
            &mut gguf_file,
            &self.config.device,
        )?;

        Ok(weights)
    }

    fn prompt(&self, prompt: &str, max_len: usize) -> anyhow::Result<String> {
        todo!()
    }
}
