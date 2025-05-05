use crate::models::raw::models::quantized_gemma3;

use crate::utils::{self, load_gemma3_model_weights, ModelConfig};

// Use the generic QuantizedModel with Gemma3 specific types and configuration
pub type QuantizedGemma3Model = crate::utils::QuantizedModel<quantized_gemma3::ModelWeights>;

impl QuantizedGemma3Model {
    pub fn new(config: ModelConfig) -> anyhow::Result<Self> {
        // Loader closure for Gemma3
        let loader = || load_gemma3_model_weights(&config.device, &config.hf_config);

        // Use the generic initializer
        let weights = utils::init_quantized("Gemma3", loader)?;

        // Define the prompt formatter for Gemma3
        let format_prompt = |prompt: &str| -> String {
            format!("<start_of_turn>user\n{prompt}<end_of_turn>\n<start_of_turn>model\n")
        };

        Ok(Self {
            name: "Gemma3",
            weights,
            format_prompt,
            eos_token: "<end_of_turn>", // EOS token for Gemma3
            config,
        })
    }
}
