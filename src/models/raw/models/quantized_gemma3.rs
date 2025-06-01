//! Gemma 3 model implementation with quantization support.
//!
//! Gemma 3 is a family of multimodal language models developed by Google.
//! This implementation provides quantization for reduced memory usage and faster inference.
//!
//! Key characteristics:
//! - Group-Query Attention (GQA) with specialized key-value heads
//! - RMSNorm for layer normalization
//! - Specialized attention patterns with separate normalization for Q/K/V
//! - Feed-forward network with SwiGLU activation
//! - Support for 2/3/4/8-bit quantization
//!
//! References:
//! - [Gemma 3 Models](https://blog.google/technology/developers/gemma-3/)
//!

use candle_core::quantized::{gguf_file, QTensor};
use candle_core::{DType, Device, IndexOp, Result, Tensor};
use candle_nn::{kv_cache::KvCache, Embedding, Module};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use super::super::quantized_nn::RmsNorm;
use crate::models::ModelWeightForward;

pub const MAX_SEQ_LEN: usize = 131072; // Gemma 3 supports 128K context window
pub const DEFAULT_SLIDING_WINDOW_TYPE: usize = 6;
pub const DEFAULT_ROPE_FREQUENCY: f32 = 1_000_000.;
pub const DEFAULT_ROPE_FREQUENCY_SLIDING: f32 = 10_000.;
pub const DEFAULT_ROPE_FREQUENCY_SCALE_FACTOR: f32 = 1.;

/// Mask cache key for efficient caching
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
struct MaskCacheKey {
    seq_len: usize,
    index_pos: usize,
    sliding_window_size: Option<usize>,
}

/// Cached mask entry
#[derive(Debug, Clone)]
struct CachedMask {
    mask: Tensor,
    dtype: DType,
}

impl CachedMask {
    fn get_mask(&self, target_dtype: DType, b_sz: usize) -> Result<Tensor> {
        let mask = if self.dtype != target_dtype {
            self.mask.to_dtype(target_dtype)?
        } else {
            self.mask.clone()
        };

        // Expand to batch size if needed
        if mask.dims()[0] != b_sz {
            // Check tensor dimensions to handle different mask shapes robustly
            let mask_dims = mask.dims();
            match mask_dims.len() {
                4 => {
                    // Standard 4D mask: (batch, heads, seq_len, total_len)
                    mask.expand((b_sz, mask_dims[1], mask_dims[2], mask_dims[3]))
                }
                3 => {
                    // 3D mask: (batch, seq_len, total_len) - add head dimension
                    let expanded = mask.expand((b_sz, mask_dims[1], mask_dims[2]))?;
                    expanded.unsqueeze(1) // Add head dimension
                }
                2 => {
                    // 2D mask: (seq_len, total_len) - add batch and head dimensions
                    let expanded = mask.expand((b_sz, mask_dims[0], mask_dims[1]))?;
                    expanded.unsqueeze(1) // Add head dimension
                }
                _ => candle_core::bail!(
                    "Unsupported mask tensor shape: {:?}. Expected 2D, 3D, or 4D tensor.",
                    mask_dims
                ),
            }
        } else {
            Ok(mask)
        }
    }
}

#[derive(Debug, Clone)]
struct QMatMul {
    inner: candle_core::quantized::QMatMul,
    span: tracing::Span,
}

impl QMatMul {
    fn from_qtensor(qtensor: QTensor) -> Result<Self> {
        let inner = candle_core::quantized::QMatMul::from_qtensor(qtensor)?;
        let span = tracing::span!(tracing::Level::TRACE, "qmatmul");
        Ok(Self { inner, span })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        self.inner.forward(xs)
    }
}

#[derive(Debug, Clone)]
struct Mlp {
    feed_forward_gate: QMatMul, // ffn_gate in GGUF
    feed_forward_up: QMatMul,   // ffn_up in GGUF
    feed_forward_down: QMatMul, // ffn_down in GGUF
}

impl Module for Mlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = self.feed_forward_gate.forward(xs)?;
        let up = self.feed_forward_up.forward(xs)?;
        let silu = candle_nn::ops::silu(&gate)?;
        let gated = (silu * up)?;
        self.feed_forward_down.forward(&gated)
    }
}

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(head_dim: usize, rope_frequency: f32, device: &Device) -> Result<Self> {
        let theta: Vec<_> = (0..head_dim)
            .step_by(2)
            .map(|i| 1f32 / rope_frequency.powf(i as f32 / head_dim as f32))
            .collect();
        let theta = Tensor::new(theta.as_slice(), device)?;
        let idx_theta = Tensor::arange(0f32, MAX_SEQ_LEN as f32, device)?
            .reshape((MAX_SEQ_LEN, 1))?
            .matmul(&theta.reshape((1, theta.elem_count()))?)?;
        let cos = idx_theta.cos()?;
        let sin = idx_theta.sin()?;
        Ok(Self { sin, cos })
    }

    fn apply_rotary_emb_qkv(
        &self,
        q: &Tensor,
        k: &Tensor,
        index_pos: usize,
    ) -> Result<(Tensor, Tensor)> {
        let (_b_sz, _h, seq_len, _n_embd) = q.dims4()?;
        let cos = self.cos.narrow(0, index_pos, seq_len)?;
        let sin = self.sin.narrow(0, index_pos, seq_len)?;
        let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

// New efficient architecture - shared weights with per-pipeline KV caches

/// Layer weights without KV cache (shareable)
#[derive(Debug, Clone)]
pub struct LayerWeightsNoCache {
    // Attention components
    pub attention_wq: QMatMul,
    pub attention_wk: QMatMul,
    pub attention_wv: QMatMul,
    pub attention_wo: QMatMul,

    // Specialized normalization for Q and K
    pub attention_q_norm: RmsNorm,
    pub attention_k_norm: RmsNorm,

    // Layer normalization
    pub attention_norm: RmsNorm,      // Applied before attention
    pub post_attention_norm: RmsNorm, // Applied after attention
    pub ffn_norm: RmsNorm,            // Applied before feedforward
    pub post_ffn_norm: RmsNorm,       // Applied after feedforward

    // Feed-forward network
    pub mlp: Mlp,

    // Attention parameters
    pub n_head: usize,    // Number of query heads
    pub n_kv_head: usize, // Number of key-value heads
    pub head_dim: usize,  // Dimension of each head
    pub q_dim: usize,     // Total dimension for queries

    pub sliding_window_size: Option<usize>,
    pub rotary_embedding: RotaryEmbedding,
    pub neg_inf: Tensor,

    // Tracing
    pub span_attn: tracing::Span,
    pub span_mlp: tracing::Span,
}

impl LayerWeightsNoCache {
    /// Create causal mask using appropriate strategy based on sequence size
    fn create_mask(
        &self,
        seq_len: usize,
        index_pos: usize,
        sliding_window_size: Option<usize>,
        device: &Device,
    ) -> Result<Tensor> {
        // Choose strategy based on sequence size
        if seq_len <= 64 || seq_len + index_pos <= 128 {
            self.create_mask_direct(seq_len, index_pos, sliding_window_size, device)
        } else {
            self.create_mask_efficient(seq_len, index_pos, sliding_window_size, device)
        }
    }

    /// Create memory-efficient causal mask using smaller tensors and broadcasting
    fn create_mask_efficient(
        &self,
        seq_len: usize,
        index_pos: usize,
        sliding_window_size: Option<usize>,
        device: &Device,
    ) -> Result<Tensor> {
        // Create a smaller (seq_len, seq_len) causal mask first using broadcasting
        let causal_mask = if seq_len > 1 {
            // Create row indices (i): [0, 1, 2, ...] expanded to (seq_len, 1)
            let row_indices = Tensor::arange(0f32, seq_len as f32, device)?.unsqueeze(1)?;
            // Create column indices (j): [0, 1, 2, ...] expanded to (1, seq_len)
            let col_indices = Tensor::arange(0f32, seq_len as f32, device)?.unsqueeze(0)?;

            // Causal condition: j > i (upper triangular)
            let causal_condition = col_indices.broadcast_gt(&row_indices)?;

            // Convert boolean to mask: true -> NEG_INF, false -> 0
            let neg_inf_tensor = Tensor::full(f32::NEG_INFINITY, (seq_len, seq_len), device)?;
            let zeros_tensor = Tensor::zeros((seq_len, seq_len), DType::F32, device)?;
            causal_condition.where_cond(&neg_inf_tensor, &zeros_tensor)?
        } else {
            Tensor::zeros((seq_len, seq_len), DType::F32, device)?
        };

        // If no cache context, just expand the square mask
        if index_pos == 0 {
            let mut final_mask = causal_mask;

            // Apply sliding window if needed
            if let Some(window_size) = sliding_window_size {
                // Use broadcasting for sliding window mask: i > j + window_size
                let row_indices = Tensor::arange(0f32, seq_len as f32, device)?.unsqueeze(1)?;
                let col_indices = Tensor::arange(0f32, seq_len as f32, device)?.unsqueeze(0)?;
                let window_size_tensor = Tensor::new(&[[window_size as f32]], device)?;

                // Sliding window condition: i > j + window_size
                let sliding_condition =
                    row_indices.broadcast_gt(&col_indices.broadcast_add(&window_size_tensor)?)?;

                let window_neg_inf = Tensor::full(f32::NEG_INFINITY, (seq_len, seq_len), device)?;
                let window_zeros = Tensor::zeros((seq_len, seq_len), DType::F32, device)?;
                let window_mask = sliding_condition.where_cond(&window_neg_inf, &window_zeros)?;
                final_mask = final_mask.maximum(&window_mask)?;
            }

            return final_mask.unsqueeze(0)?.unsqueeze(0); // Add batch and head dims
        }

        // For cached context, create the full mask efficiently
        // Left part: all zeros (can attend to cached context)
        let left_part = Tensor::zeros((seq_len, index_pos), DType::F32, device)?;

        // Right part: causal mask for new tokens
        let right_part = causal_mask;

        // Concatenate along the last dimension
        let mut final_mask = Tensor::cat(&[&left_part, &right_part], 1)?;

        // Apply sliding window if needed
        if let Some(window_size) = sliding_window_size {
            // Use broadcasting for sliding window mask with cached context
            let total_len = seq_len + index_pos;
            let row_indices = Tensor::arange(0f32, seq_len as f32, device)?.unsqueeze(1)?;
            let col_indices = Tensor::arange(0f32, total_len as f32, device)?.unsqueeze(0)?;
            let index_pos_tensor = if index_pos == 0 {
                Tensor::zeros((1, 1), DType::F32, device)?
            } else {
                Tensor::new(&[[index_pos as f32]], device)?
            };
            let window_size_tensor = Tensor::new(&[[window_size as f32]], device)?;

            // Sliding window condition: (i + index_pos) > (j + window_size)
            let current_pos = row_indices.broadcast_add(&index_pos_tensor)?;
            let sliding_condition =
                current_pos.broadcast_gt(&col_indices.broadcast_add(&window_size_tensor)?)?;

            let sliding_total_neg_inf =
                Tensor::full(f32::NEG_INFINITY, (seq_len, total_len), device)?;
            let sliding_total_zeros = Tensor::zeros((seq_len, total_len), DType::F32, device)?;
            let window_mask =
                sliding_condition.where_cond(&sliding_total_neg_inf, &sliding_total_zeros)?;
            final_mask = final_mask.maximum(&window_mask)?;
        }

        final_mask.unsqueeze(0)?.unsqueeze(0) // Add batch and head dims
    }

    /// Fallback direct method for small sequences using broadcasting
    fn create_mask_direct(
        &self,
        seq_len: usize,
        index_pos: usize,
        sliding_window_size: Option<usize>,
        device: &Device,
    ) -> Result<Tensor> {
        let total_len = seq_len + index_pos;

        // Create index tensors for broadcasting
        let row_indices = Tensor::arange(0f32, seq_len as f32, device)?.unsqueeze(1)?;
        let col_indices = Tensor::arange(0f32, total_len as f32, device)?.unsqueeze(0)?;
        // Create index_pos_tensor as a (1,1) tensor for broadcasting
        let index_pos_tensor = if index_pos == 0 {
            Tensor::zeros((1, 1), DType::F32, device)?
        } else {
            Tensor::new(&[[index_pos as f32]], device)?
        };
        // Start with zeros
        let mut mask = Tensor::zeros((seq_len, total_len), DType::F32, device)?;

        // Causal mask: j > (i + index_pos)
        let current_pos = row_indices.clone().broadcast_add(&index_pos_tensor)?;
        let causal_condition = col_indices.broadcast_gt(&current_pos)?;
        let neg_inf_tensor = Tensor::full(f32::NEG_INFINITY, (seq_len, total_len), device)?;
        mask = causal_condition.where_cond(&neg_inf_tensor, &mask)?;

        // Sliding window mask if enabled
        if let Some(window_size) = sliding_window_size {
            let window_size_tensor = Tensor::new(&[[window_size as f32]], device)?;
            // Sliding window condition: (i + index_pos) > (j + window_size)
            let sliding_condition =
                current_pos.broadcast_gt(&col_indices.broadcast_add(&window_size_tensor)?)?;
            let sliding_neg_inf = Tensor::full(f32::NEG_INFINITY, (seq_len, total_len), device)?;
            let sliding_zeros = Tensor::zeros((seq_len, total_len), DType::F32, device)?;
            let sliding_mask = sliding_condition.where_cond(&sliding_neg_inf, &sliding_zeros)?;
            mask = mask.maximum(&sliding_mask)?;
        }

        mask.unsqueeze(0)?.unsqueeze(0) // Add batch and head dims
    }

    /// Get or create mask with caching
    fn get_cached_mask(
        seq_len: usize,
        index_pos: usize,
        sliding_window_size: Option<usize>,
        mask_cache: &mut HashMap<MaskCacheKey, CachedMask>,
        layer: &LayerWeightsNoCache,
        device: &Device,
        target_dtype: DType,
        b_sz: usize,
    ) -> Result<Tensor> {
        // Clear mask cache when starting a new sequence to prevent unbounded memory growth
        if index_pos == 0 {
            mask_cache.clear();
        }

        let cache_key = MaskCacheKey {
            seq_len,
            index_pos,
            sliding_window_size,
        };

        // Check if mask is in cache
        if let Some(cached) = mask_cache.get(&cache_key) {
            return cached.get_mask(target_dtype, b_sz);
        }

        // Create new mask
        let mask = layer.create_mask(seq_len, index_pos, sliding_window_size, device)?;

        // Cache the mask (store as F32 to avoid dtype conversions in cache)
        let cached_mask = CachedMask {
            mask: mask.clone(),
            dtype: DType::F32,
        };
        mask_cache.insert(cache_key, cached_mask);

        // Return mask in correct dtype and batch size
        let mask = if DType::F32 != target_dtype {
            mask.to_dtype(target_dtype)?
        } else {
            mask
        };

        if mask.dims()[0] != b_sz {
            mask.expand((b_sz, mask.dim(1)?, mask.dim(2)?, mask.dim(3)?))
        } else {
            Ok(mask)
        }
    }

    fn forward_attn(
        &self,
        x: &Tensor,
        mask: Option<&Tensor>,
        index_pos: usize,
        kv_cache: &mut KvCache,
    ) -> Result<Tensor> {
        let _enter = self.span_attn.enter();
        let (b_sz, seq_len, _) = x.dims3()?;

        let q = self.attention_wq.forward(x)?;
        let k = self.attention_wk.forward(x)?;
        let v = self.attention_wv.forward(x)?;

        let q = q.reshape((b_sz, seq_len, self.n_head, self.head_dim))?;
        let k = k.reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?;
        let v = v.reshape((b_sz, seq_len, self.n_kv_head, self.head_dim))?;

        // Apply normalization to Q and K
        let q = q.transpose(1, 2)?.flatten(0, 2)?;
        let k = k.transpose(1, 2)?.flatten(0, 2)?;
        let q = self.attention_q_norm.forward(&q)?;
        let k = self.attention_k_norm.forward(&k)?;
        let q = q.reshape((b_sz, self.n_head, seq_len, self.head_dim))?;
        let k = k.reshape((b_sz, self.n_kv_head, seq_len, self.head_dim))?;
        let v = v.transpose(1, 2)?;

        // Apply rotary embedding
        let (q, k) = self
            .rotary_embedding
            .apply_rotary_emb_qkv(&q, &k, index_pos)?;

        // Update KV cache
        if index_pos == 0 {
            kv_cache.reset();
        }
        let k_cont = k.contiguous()?;
        let v_cont = v.contiguous()?;
        let (k, v) = kv_cache.append(&k_cont, &v_cont)?;

        // Expand KV to match query heads if needed (for GQA)
        let k = if self.n_kv_head != self.n_head {
            let repeat_count = self.n_head / self.n_kv_head;
            super::super::utils::repeat_kv(k, repeat_count)?
        } else {
            k
        };
        let v = if self.n_kv_head != self.n_head {
            let repeat_count = self.n_head / self.n_kv_head;
            super::super::utils::repeat_kv(v, repeat_count)?
        } else {
            v
        };

        // Attention computation
        let scale = (self.head_dim as f64).powf(-0.5);
        let scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        let scores = match mask {
            Some(mask) => scores.broadcast_add(mask)?,
            None => scores,
        };
        let attn = candle_nn::ops::softmax_last_dim(&scores)?;
        let out = attn.matmul(&v)?;

        let out = out.transpose(1, 2)?.reshape((b_sz, seq_len, self.q_dim))?;
        self.attention_wo.forward(&out)
    }
}

/// Shared model weights (immutable, shareable across pipelines)
#[derive(Debug, Clone)]
pub struct Weights {
    pub tok_embeddings: Embedding,
    pub embedding_length: usize,
    pub layers: Vec<LayerWeightsNoCache>,
    pub norm: RmsNorm,
    pub output: QMatMul,
    pub device: Device,
    pub span: tracing::Span,
    pub span_output: tracing::Span,
}

impl Weights {
    pub fn from_gguf<R: std::io::Seek + std::io::Read>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        let md_get = |s: &str| match ct.metadata.get(s) {
            None => candle_core::bail!("cannot find {s} in metadata"),
            Some(v) => Ok(v),
        };

        let head_count = md_get("gemma3.attention.head_count")?.to_u32()? as usize;
        let head_count_kv = md_get("gemma3.attention.head_count_kv")?.to_u32()? as usize;
        let block_count = md_get("gemma3.block_count")?.to_u32()? as usize;
        let embedding_length = md_get("gemma3.embedding_length")?.to_u32()? as usize;
        let key_length = md_get("gemma3.attention.key_length")?.to_u32()? as usize;
        let _value_length = md_get("gemma3.attention.value_length")?.to_u32()? as usize;
        let rms_norm_eps = md_get("gemma3.attention.layer_norm_rms_epsilon")?.to_f32()? as f64;
        let sliding_window_size = md_get("gemma3.attention.sliding_window")?.to_u32()? as usize;

        let sliding_window_type = md_get("gemma3.attention.sliding_window_type")
            .and_then(|m| Ok(m.to_u32()? as usize))
            .unwrap_or(DEFAULT_SLIDING_WINDOW_TYPE);

        let rope_freq_base = md_get("gemma3.rope.freq_base")
            .and_then(|m| m.to_f32())
            .unwrap_or(DEFAULT_ROPE_FREQUENCY);

        let rope_freq_base_sliding = md_get("gemma3.rope.local_freq_base")
            .and_then(|m| m.to_f32())
            .unwrap_or(DEFAULT_ROPE_FREQUENCY_SLIDING);

        // Compute the dimensions for queries, keys, and values
        let q_dim = head_count * key_length;

        let neg_inf = Tensor::new(f32::NEG_INFINITY, device)?;

        // Load token embeddings and output projection
        let tok_embeddings = ct.tensor(reader, "token_embd.weight", device)?;
        let tok_embeddings = tok_embeddings.dequantize(device)?;
        let norm = RmsNorm::from_qtensor(
            ct.tensor(reader, "output_norm.weight", device)?,
            rms_norm_eps,
        )?;
        let output = match ct.tensor(reader, "output.weight", device) {
            Ok(tensor) => tensor,
            Err(_) => ct.tensor(reader, "token_embd.weight", device)?, // Use tied weights if output.weight doesn't exist
        };

        let mut layers = Vec::with_capacity(block_count);
        for layer_idx in 0..block_count {
            let prefix = format!("blk.{layer_idx}");

            let attention_wq = ct.tensor(reader, &format!("{prefix}.attn_q.weight"), device)?;
            let attention_wk = ct.tensor(reader, &format!("{prefix}.attn_k.weight"), device)?;
            let attention_wv = ct.tensor(reader, &format!("{prefix}.attn_v.weight"), device)?;
            let attention_wo =
                ct.tensor(reader, &format!("{prefix}.attn_output.weight"), device)?;

            let attention_q_norm = RmsNorm::from_qtensor(
                ct.tensor(reader, &format!("{prefix}.attn_q_norm.weight"), device)?,
                rms_norm_eps,
            )?;

            let attention_k_norm = RmsNorm::from_qtensor(
                ct.tensor(reader, &format!("{prefix}.attn_k_norm.weight"), device)?,
                rms_norm_eps,
            )?;

            let attention_norm = RmsNorm::from_qtensor(
                ct.tensor(reader, &format!("{prefix}.attn_norm.weight"), device)?,
                rms_norm_eps,
            )?;

            let post_attention_norm = RmsNorm::from_qtensor(
                ct.tensor(
                    reader,
                    &format!("{prefix}.post_attention_norm.weight"),
                    device,
                )?,
                rms_norm_eps,
            )?;

            let ffn_norm = RmsNorm::from_qtensor(
                ct.tensor(reader, &format!("{prefix}.ffn_norm.weight"), device)?,
                rms_norm_eps,
            )?;

            let post_ffn_norm = RmsNorm::from_qtensor(
                ct.tensor(reader, &format!("{prefix}.post_ffw_norm.weight"), device)?,
                rms_norm_eps,
            )?;

            let feed_forward_gate =
                ct.tensor(reader, &format!("{prefix}.ffn_gate.weight"), device)?;
            let feed_forward_up = ct.tensor(reader, &format!("{prefix}.ffn_up.weight"), device)?;
            let feed_forward_down =
                ct.tensor(reader, &format!("{prefix}.ffn_down.weight"), device)?;

            let mlp = Mlp {
                feed_forward_gate: QMatMul::from_qtensor(feed_forward_gate)?,
                feed_forward_up: QMatMul::from_qtensor(feed_forward_up)?,
                feed_forward_down: QMatMul::from_qtensor(feed_forward_down)?,
            };

            // Sliding window pattern hardcoded to 6 because it's not explicitly defined
            let is_sliding = (layer_idx + 1) % sliding_window_type > 0;
            let sliding_window_size = is_sliding.then_some(sliding_window_size);
            let layer_rope_frequency = if is_sliding {
                rope_freq_base_sliding
            } else {
                rope_freq_base
            };

            let rotary_embedding = RotaryEmbedding::new(key_length, layer_rope_frequency, device)?;

            // Tracing spans
            let span_attn = tracing::span!(tracing::Level::TRACE, "attn");
            let span_mlp = tracing::span!(tracing::Level::TRACE, "attn-mlp");

            layers.push(LayerWeightsNoCache {
                attention_wq: QMatMul::from_qtensor(attention_wq)?,
                attention_wk: QMatMul::from_qtensor(attention_wk)?,
                attention_wv: QMatMul::from_qtensor(attention_wv)?,
                attention_wo: QMatMul::from_qtensor(attention_wo)?,
                attention_q_norm,
                attention_k_norm,
                attention_norm,
                post_attention_norm,
                ffn_norm,
                post_ffn_norm,
                mlp,
                n_head: head_count,
                n_kv_head: head_count_kv,
                head_dim: key_length,
                q_dim,
                sliding_window_size,
                rotary_embedding,
                neg_inf: neg_inf.clone(),
                span_attn,
                span_mlp,
            })
        }

        let span = tracing::span!(tracing::Level::TRACE, "model");
        let span_output = tracing::span!(tracing::Level::TRACE, "output");

        Ok(Self {
            tok_embeddings: Embedding::new(tok_embeddings, embedding_length),
            embedding_length,
            layers,
            norm,
            output: QMatMul::from_qtensor(output)?,
            device: device.clone(),
            span,
            span_output,
        })
    }
}

/// Per-pipeline state that holds individual KV caches
pub struct PipelineState {
    pub weights: Arc<Weights>,
    pub kv_caches: Vec<KvCache>,
    pub mask_cache: HashMap<MaskCacheKey, CachedMask>,
}

impl PipelineState {
    pub fn new(shared_weights: Arc<Weights>) -> Self {
        let num_layers = shared_weights.layers.len();
        let initial_cache_size = 64; // Small initial size, grows dynamically
        let kv_caches: Vec<KvCache> = (0..num_layers)
            .map(|_| KvCache::new(2, initial_cache_size))
            .collect();

        Self {
            weights: shared_weights,
            kv_caches,
            mask_cache: HashMap::new(),
        }
    }

    pub fn new_with_cache_size(shared_weights: Arc<Weights>, initial_cache_size: usize) -> Self {
        let num_layers = shared_weights.layers.len();
        let kv_caches: Vec<KvCache> = (0..num_layers)
            .map(|_| KvCache::new(2, initial_cache_size))
            .collect();

        Self {
            weights: shared_weights,
            kv_caches,
            mask_cache: HashMap::new(),
        }
    }

    /// Pre-compute masks for all unique sliding window configurations in this model
    /// This avoids redundant mask computation during the forward pass
    fn precompute_masks_for_forward(
        &mut self,
        seq_len: usize,
        index_pos: usize,
        device: &Device,
        dtype: DType,
        b_sz: usize,
    ) -> Result<Vec<(Option<usize>, Tensor)>> {
        // Collect all unique sliding window configurations
        let mut unique_configs = HashSet::new();
        for layer in &self.weights.layers {
            unique_configs.insert(layer.sliding_window_size);
        }

        // Compute mask for each unique configuration
        let mut masks = Vec::new();
        for config in unique_configs {
            // Find a layer that matches this sliding window configuration
            let template_layer = self
                .weights
                .layers
                .iter()
                .find(|layer| layer.sliding_window_size == config)
                .unwrap_or(&self.weights.layers[0]); // Fallback to first layer if no match

            let mask = LayerWeightsNoCache::get_cached_mask(
                seq_len,
                index_pos,
                config,
                &mut self.mask_cache,
                template_layer,
                device,
                dtype,
                b_sz,
            )?;
            masks.push((config, mask));
        }

        Ok(masks)
    }

    pub fn forward(&mut self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        let (b_sz, seq_len) = x.dims2()?;

        // Clear KV caches when processing conversation history (index_pos = 0)
        // This ensures clean state when the user provides full conversation context
        if index_pos == 0 {
            // Clear all KV caches to ensure clean state
            for kv_cache in &mut self.kv_caches {
                kv_cache.reset();
            }
            // Clear mask cache as well
            self.mask_cache.clear();
        }

        // Pre-compute masks for all unique sliding window configurations to avoid redundant work
        let masks = if seq_len == 1 {
            Vec::new() // No masks needed for single token
        } else {
            // Use F32 dtype for masks to match attention scores, not input token dtype
            self.precompute_masks_for_forward(seq_len, index_pos, x.device(), DType::F32, b_sz)?
        };

        let _enter = self.weights.span.enter();

        let mut layer_in = self.weights.tok_embeddings.forward(x)?;
        layer_in = (layer_in * (self.weights.embedding_length as f64).sqrt())?;

        for (layer_idx, layer) in self.weights.layers.iter().enumerate() {
            let attention_mask = if seq_len == 1 {
                None
            } else {
                // Find the pre-computed mask for this layer's configuration
                masks
                    .iter()
                    .find(|(config, _)| *config == layer.sliding_window_size)
                    .map(|(_, mask)| mask.clone())
            };

            // Attention block
            let residual = &layer_in;
            let x = layer.attention_norm.forward(&layer_in)?;
            let x = layer.forward_attn(
                &x,
                attention_mask.as_ref(),
                index_pos,
                &mut self.kv_caches[layer_idx],
            )?;
            let x = layer.post_attention_norm.forward(&x)?;
            let x = (x + residual)?;

            // Feed-forward block
            let _enter = layer.span_mlp.enter();
            let residual = &x;
            let x = layer.ffn_norm.forward(&x)?;
            let x = layer.mlp.forward(&x)?;
            let x = layer.post_ffn_norm.forward(&x)?;
            let x = (x + residual)?;
            drop(_enter);

            layer_in = x;
        }

        let _enter = self.weights.span_output.enter();

        let x = layer_in.i((.., seq_len - 1, ..))?;
        let x = self.weights.norm.forward(&x)?;
        let output = self.weights.output.forward(&x)?;

        Ok(output)
    }
}

impl ModelWeightForward for PipelineState {
    fn forward(&mut self, x: &Tensor, index_pos: usize) -> Result<Tensor> {
        self.forward(x, index_pos)
    }
}
