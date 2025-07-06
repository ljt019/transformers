pub mod generation;

pub mod components {
    pub mod layers;
    pub mod attention;
    pub mod quantization;

    pub use layers::RmsNorm;
    pub use quantization::{QMatMul, VarBuilder};
    pub use attention::repeat_kv;
}

pub mod implementations {
    pub mod modernbert;
    pub mod gemma3;
    pub mod qwen3;
}

pub use components::{repeat_kv, QMatMul, RmsNorm, VarBuilder};
pub use implementations::{gemma3 as quantized_gemma3, modernbert, qwen3 as quantized_qwen3};
