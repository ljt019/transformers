use crate::models::raw::models::quantized_qwen3;

use crate::utils::{self, load_qwen3_model_weights, ModelConfig, QuantizedModelWeights};

// Use the generic QuantizedModel with Qwen3 specific types and configuration
pub type QuantizedQwen3Model = crate::utils::QuantizedModel<quantized_qwen3::ModelWeights>;

// Implement the QuantizedModelWeights trait for Qwen3 model weights
impl QuantizedModelWeights for quantized_qwen3::ModelWeights {
    fn forward(
        &mut self,
        xs: &candle_core::Tensor,
        start_pos: usize,
    ) -> candle_core::Result<candle_core::Tensor> {
        self.forward(xs, start_pos)
    }
}

impl QuantizedQwen3Model {
    pub fn new(config: ModelConfig) -> anyhow::Result<Self> {
        // Loader closure for Qwen3
        let loader = || load_qwen3_model_weights(&config.device, &config.hf_config);

        // Use the generic initializer
        let weights = utils::init_quantized(loader)?;

        // Define the prompt formatter for Qwen3
        let format_prompt = |prompt: &str| -> String {
            format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n")
        };

        Ok(Self {
            name: "Qwen3",
            weights,
            format_prompt,
            eos_token: "<|endoftext|>",
            config,
        })
    }
}
