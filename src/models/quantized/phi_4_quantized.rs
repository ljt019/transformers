use candle_transformers::models::quantized_phi3;

use crate::utils::{
    self, // Import the utils module directly
    load_phi3_model_weights,
    ModelConfig,
};

// Use the generic QuantizedModel with Phi3 weights (for Phi-4) specific types and configuration
pub type QuantizedPhi4Model = crate::utils::QuantizedModel<quantized_phi3::ModelWeights>;

impl QuantizedPhi4Model {
    pub fn new(config: ModelConfig) -> anyhow::Result<Self> {
        // Loader closure for Phi3/Phi4, including flash_attn parameter
        let loader = || {
            load_phi3_model_weights(
                &config.device,
                &config.hf_config,
                config.params.use_flash_attn,
            )
        };

        // Use the generic initializer
        let weights = utils::init_quantized("Phi4", loader)?;

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
