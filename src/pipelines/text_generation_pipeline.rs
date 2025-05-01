use std::path::PathBuf;

use crate::gemma_3_quantized::ModelConfig;
use crate::gemma_3_quantized::ModelParams;
use crate::gemma_3_quantized::QuantizedGemma3Model;

/*
let pipeline = TextGeneratorPipeline::new('gemma-3-1b-it-quantized');

prompt = "Hello, how are you?";

let generated_text = pipeline.generate_text(prompt, max_length);
*/

use crate::utils::HfConfig;

pub enum ModelOptions {
    Gemma3_1b,
}

pub struct TextGenerationPipelineBuilder {
    model_choice: Option<ModelOptions>,
}

impl TextGenerationPipelineBuilder {
    pub fn new() -> Self {
        Self { model_choice: None }
    }

    pub fn set_model_choice(mut self, model_choice: ModelOptions) -> Self {
        self.model_choice = Some(model_choice);
        self
    }

    pub fn build(self) -> anyhow::Result<TextGenerationPipeline> {
        if !self.model_choice.is_some() {
            return Err(anyhow::anyhow!("Model choice is not set"));
        }

        let model_choice = self.model_choice.unwrap();

        let model_config = match model_choice {
            ModelOptions::Gemma3_1b => ModelConfig::new(
                ModelParams::default(),
                HfConfig::new(
                    "google/gemma-3-1b-it",
                    "tokenizer.json",
                    "unsloth/gemma-3-1b-it-GGUF",
                    "gemma-3-1b-it-Q4_K_M.gguf",
                ),
            ),
        }?;

        TextGenerationPipeline::new(model_config)
    }
}

pub struct TextGenerationPipeline {
    model: Box<dyn LargeLanguageModel>,
    tokenizer: tokenizers::Tokenizer,
}

impl TextGenerationPipeline {
    pub fn new(model_config: ModelConfig) -> anyhow::Result<Self> {
        let gemma_model = QuantizedGemma3Model::new(model_config.clone())?;

        let tokenizer = crate::utils::load_tokenizer(&model_config.hf_config)?;

        Ok(TextGenerationPipeline {
            model: Box::new(gemma_model),
            tokenizer,
        })
    }

    pub fn generate_text(&self, prompt: &str, max_length: usize) -> anyhow::Result<String> {
        self.model.prompt_model(&self.tokenizer, prompt, max_length)
    }
}

pub trait LargeLanguageModel {
    fn prompt_model(
        &self,
        tokenizer: &tokenizers::Tokenizer,
        prompt: &str,
        max_length: usize,
    ) -> anyhow::Result<String>;
}
