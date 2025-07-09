//! Common layer implementations shared across multiple models.
//!
//! This module provides unified implementations of common neural network layers
//! used across different model architectures like Qwen3, Gemma3, etc.

use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::VarBuilder;
use crate::models::components::{QMatMul, RmsNorm};
use crate::models::components::attention::repeat_kv;
use crate::models::components::attention::KvCache;
use std::sync::Arc;

/// Parameters for RoPE (Rotary Position Embedding) configuration
pub trait RoPEParams: Clone + Send + Sync {
    /// Base frequency for RoPE calculations  
    fn theta(&self) -> f64;
    /// Whether to apply frequency scaling
    fn use_scaled_rope(&self) -> bool {
        false
    }
    /// Additional configuration specific to the model
    fn scaling_factor(&self) -> f64 {
        1.0
    }
}

/// Default RoPE parameters used by most models
#[derive(Clone)]
pub struct DefaultRoPEParams {
    pub theta: f64,
}

impl RoPEParams for DefaultRoPEParams {
    fn theta(&self) -> f64 {
        self.theta
    }
}

/// Qwen3-specific RoPE parameters
#[derive(Clone)]
pub struct Qwen3RoPEParams;

impl RoPEParams for Qwen3RoPEParams {
    fn theta(&self) -> f64 {
        1_000_000.0
    }
}

/// Gemma3-specific RoPE parameters  
#[derive(Clone)]
pub struct Gemma3RoPEParams {
    pub rope_frequency: f32,
}

impl RoPEParams for Gemma3RoPEParams {
    fn theta(&self) -> f64 {
        self.rope_frequency as f64
    }
}

/// Shared RoPE implementation with model-specific customization via params
pub struct RoPE<P: RoPEParams = DefaultRoPEParams> {
    cos: Tensor,
    sin: Tensor,
    params: P,
}

impl<P: RoPEParams> RoPE<P> {
    /// Create a new RoPE instance with the given parameters
    pub fn new(
        dtype: DType,
        head_dim: usize,
        max_seq_len: usize,
        params: P,
        device: &Device,
    ) -> Result<Self> {
        let theta = params.theta();
        let scaling_factor = params.scaling_factor();
        
        let inv_freq = (0..head_dim)
            .step_by(2)
            .map(|i| 1.0 / (theta.powf((i as f64) / (head_dim as f64)) * scaling_factor))
            .collect::<Vec<_>>();

        let t = (0..max_seq_len).map(|i| i as f64).collect::<Vec<_>>();
        
        let mut freqs = Vec::with_capacity(max_seq_len * head_dim / 2);
        for &ti in &t {
            for &freq in &inv_freq {
                freqs.push(ti * freq);
            }
        }

        let freqs = Tensor::from_vec(freqs, (max_seq_len, head_dim / 2), device)?;
        let cos = freqs.cos()?.to_dtype(dtype)?;
        let sin = freqs.sin()?.to_dtype(dtype)?;

        Ok(Self { cos, sin, params })
    }

    /// Apply RoPE to query and key tensors
    pub fn apply(&self, q: &Tensor, k: &Tensor, position_offset: usize) -> Result<(Tensor, Tensor)> {
        let (_batch_size, _num_heads, seq_len, head_dim) = q.dims4()?;
        let half_dim = head_dim / 2;

        let cos = self.cos.narrow(0, position_offset, seq_len)?;
        let sin = self.sin.narrow(0, position_offset, seq_len)?;

        // Split q and k into halves for rotation
        let q1 = q.narrow(D::Minus1, 0, half_dim)?;
        let q2 = q.narrow(D::Minus1, half_dim, half_dim)?;
        let k1 = k.narrow(D::Minus1, 0, half_dim)?;
        let k2 = k.narrow(D::Minus1, half_dim, half_dim)?;

        // Apply rotation
        let q_rot = Tensor::cat(&[
            (q1.broadcast_mul(&cos)? - q2.broadcast_mul(&sin)?)?,
            (q1.broadcast_mul(&sin)? + q2.broadcast_mul(&cos)?)?,
        ], D::Minus1)?;

        let k_rot = Tensor::cat(&[
            (k1.broadcast_mul(&cos)? - k2.broadcast_mul(&sin)?)?,
            (k1.broadcast_mul(&sin)? + k2.broadcast_mul(&cos)?)?,
        ], D::Minus1)?;

        Ok((q_rot, k_rot))
    }
}

/// Shared FeedForward network implementation
pub struct FeedForward {
    pub gate_proj: QMatMul,
    pub up_proj: QMatMul,
    pub down_proj: QMatMul,
}

/// Weight naming configuration for models
pub struct WeightNaming {
    pub mlp_gate: &'static str,
    pub mlp_up: &'static str,
    pub mlp_down: &'static str,
    pub attn_q: &'static str,
    pub attn_k: &'static str,
    pub attn_v: &'static str,
    pub attn_o: &'static str,
    pub attn_q_norm: &'static str,
    pub attn_k_norm: &'static str,
}

impl Default for WeightNaming {
    fn default() -> Self {
        Self {
            mlp_gate: "mlp.gate_proj.weight",
            mlp_up: "mlp.up_proj.weight",
            mlp_down: "mlp.down_proj.weight",
            attn_q: "attn.q_proj.weight",
            attn_k: "attn.k_proj.weight",
            attn_v: "attn.v_proj.weight",
            attn_o: "attn.o_proj.weight",
            attn_q_norm: "attn.q_norm.weight",
            attn_k_norm: "attn.k_norm.weight",
        }
    }
}

impl WeightNaming {
    pub fn qwen3() -> Self {
        Self {
            mlp_gate: "ffn_gate.weight",
            mlp_up: "ffn_up.weight",
            mlp_down: "ffn_down.weight",
            attn_q: "attn_q.weight",
            attn_k: "attn_k.weight",
            attn_v: "attn_v.weight",
            attn_o: "attn_output.weight",
            attn_q_norm: "attn_q_norm.weight",
            attn_k_norm: "attn_k_norm.weight",
        }
    }
}

impl FeedForward {
    /// Load FeedForward weights from GGUF format
    pub fn load<R: std::io::Read + std::io::Seek>(
        content: &candle_core::quantized::gguf_file::Content,
        reader: &mut R,
        layer_prefix: &str,
        device: &Device,
    ) -> Result<Self> {
        let naming = WeightNaming::default();
        Self::load_with_naming(content, reader, layer_prefix, device, &naming)
    }

    /// Load FeedForward weights with custom naming
    pub fn load_with_naming<R: std::io::Read + std::io::Seek>(
        content: &candle_core::quantized::gguf_file::Content,
        reader: &mut R,
        layer_prefix: &str,
        device: &Device,
        naming: &WeightNaming,
    ) -> Result<Self> {
        let gate_proj = QMatMul::from_weights(content.tensor(
            reader,
            &format!("{layer_prefix}.{}", naming.mlp_gate),
            device,
        )?)?;
        let up_proj = QMatMul::from_weights(content.tensor(
            reader,
            &format!("{layer_prefix}.{}", naming.mlp_up),
            device,
        )?)?;
        let down_proj = QMatMul::from_weights(content.tensor(
            reader,
            &format!("{layer_prefix}.{}", naming.mlp_down),
            device,
        )?)?;

        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }
}

impl Module for FeedForward {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(x)?.silu()?;
        let up = self.up_proj.forward(x)?;
        let hidden = (gate * up)?;
        self.down_proj.forward(&hidden)
    }
}

/// Configuration for attention layer
pub struct AttentionConfig {
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub sliding_window_size: Option<usize>,
    pub rms_eps: f64,
}

/// Shared Attention implementation with optional query/key normalization
pub struct Attention<P: RoPEParams = DefaultRoPEParams> {
    pub q_proj: QMatMul,
    pub k_proj: QMatMul,
    pub v_proj: QMatMul,
    pub o_proj: QMatMul,
    pub q_norm: Option<RmsNorm>,
    pub k_norm: Option<RmsNorm>,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub head_dim: usize,
    pub sliding_window_size: Option<usize>,
    pub rope: Arc<RoPE<P>>,
}

impl<P: RoPEParams> Attention<P> {
    /// Load attention weights from GGUF format
    pub fn load<R: std::io::Read + std::io::Seek>(
        content: &candle_core::quantized::gguf_file::Content,
        reader: &mut R,
        layer_prefix: &str,
        config: AttentionConfig,
        rope: Arc<RoPE<P>>,
        device: &Device,
        use_qk_norm: bool,
    ) -> Result<Self> {
        let naming = WeightNaming::default();
        Self::load_with_naming(content, reader, layer_prefix, config, rope, device, use_qk_norm, &naming)
    }

    /// Load attention weights with custom naming
    pub fn load_with_naming<R: std::io::Read + std::io::Seek>(
        content: &candle_core::quantized::gguf_file::Content,
        reader: &mut R,
        layer_prefix: &str,
        config: AttentionConfig,
        rope: Arc<RoPE<P>>,
        device: &Device,
        use_qk_norm: bool,
        naming: &WeightNaming,
    ) -> Result<Self> {
        let q_proj = QMatMul::from_weights(content.tensor(
            reader,
            &format!("{layer_prefix}.{}", naming.attn_q),
            device,
        )?)?;
        let k_proj = QMatMul::from_weights(content.tensor(
            reader,
            &format!("{layer_prefix}.{}", naming.attn_k),
            device,
        )?)?;
        let v_proj = QMatMul::from_weights(content.tensor(
            reader,
            &format!("{layer_prefix}.{}", naming.attn_v),
            device,
        )?)?;
        let o_proj = QMatMul::from_weights(content.tensor(
            reader,
            &format!("{layer_prefix}.{}", naming.attn_o),
            device,
        )?)?;

        let (q_norm, k_norm) = if use_qk_norm {
            let q_norm = RmsNorm::from_qtensor(
                content.tensor(
                    reader,
                    &format!("{layer_prefix}.{}", naming.attn_q_norm),
                    device,
                )?,
                config.rms_eps,
            )?;
            let k_norm = RmsNorm::from_qtensor(
                content.tensor(
                    reader,
                    &format!("{layer_prefix}.{}", naming.attn_k_norm),
                    device,
                )?,
                config.rms_eps,
            )?;
            (Some(q_norm), Some(k_norm))
        } else {
            (None, None)
        };

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads: config.num_heads,
            num_kv_heads: config.num_kv_heads,
            head_dim: config.head_dim,
            sliding_window_size: config.sliding_window_size,
            rope,
        })
    }

    /// Forward pass through attention layer
    pub fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        position_offset: usize,
        kv_cache: &mut KvCache,
    ) -> Result<Tensor> {
        let (batch_size, seq_len, _hidden_size) = hidden_states.dims3()?;

        // Project to Q, K, V
        let q = self.q_proj.forward(hidden_states)?;
        let k = self.k_proj.forward(hidden_states)?;
        let v = self.v_proj.forward(hidden_states)?;

        // Reshape for attention
        let q = q.reshape((batch_size, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k.reshape((batch_size, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v.reshape((batch_size, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Apply query/key normalization if configured
        let (q, k) = if let (Some(q_norm), Some(k_norm)) = (&self.q_norm, &self.k_norm) {
            (q_norm.forward(&q)?, k_norm.forward(&k)?)
        } else {
            (q, k)
        };

        // Apply RoPE
        let (q, k) = self.rope.apply(&q, &k, position_offset)?;

        // Update KV cache
        let (k, v) = kv_cache.update(k, v)?;

        // Repeat K,V if needed for GQA
        let (k, v) = if self.num_kv_heads < self.num_heads {
            let n_rep = self.num_heads / self.num_kv_heads;
            (repeat_kv(k, n_rep)?, repeat_kv(v, n_rep)?)
        } else {
            (k, v)
        };

        // Compute attention scores
        let scale = (self.head_dim as f64).sqrt();
        let scores = (q.matmul(&k.transpose(D::Minus2, D::Minus1)?)? / scale)?;

        // Apply attention mask if provided
        let scores = if let Some(mask) = attention_mask {
            scores.broadcast_add(mask)?
        } else {
            scores
        };

        // Softmax and apply to values
        let attn_weights = candle_nn::ops::softmax_last_dim(&scores)?;
        let attn_output = attn_weights.matmul(&v)?;

        // Reshape and project output
        let attn_output = attn_output
            .transpose(1, 2)?
            .reshape((batch_size, seq_len, self.num_heads * self.head_dim))?;

        self.o_proj.forward(&attn_output)
    }
}