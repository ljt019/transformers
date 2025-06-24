//! High-performance Qwen3 implementation with quantization support.
//!
//! This implementation provides efficient inference for Qwen3 models with:
//! - Quantized weights for reduced memory usage
//! - KV caching for autoregressive generation
//! - Multi-context support for concurrent conversations
//! - GPU acceleration via Candle framework
//!
//! # Quick Start
//! ```rust
//! use std::fs::File;
//! let mut file = File::open("model.gguf")?;
//! let model = Qwen3Model::from_gguf(&mut file, &Device::Cpu)?;
//! let mut ctx = model.new_context();
//! let output = ctx.generate(&input_tokens)?;
//! ```

use crate::models::RmsNorm;
use candle_core::quantized::{gguf_file, QMatMul};
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{kv_cache::KvCache, Activation, Embedding, Module};
use std::io::{Read, Seek};
use std::sync::Arc;
use tokenizers::Tokenizer;

// Constants
const DEFAULT_CACHE_SIZE: usize = 64;
const KV_CACHE_DIMS: usize = 2;

/// Repeats a key or value tensor for grouped query attention
/// The input tensor should have a shape `(batch, num_kv_heads, seq_len, head_dim)`,
pub fn repeat_kv(xs: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        Ok(xs)
    } else {
        let (b_sz, n_kv_head, seq_len, head_dim) = xs.dims4()?;
        // Using cat is faster than a broadcast as it avoids going through a potentially
        // strided copy.
        // https://github.com/huggingface/candle/pull/2043
        Tensor::cat(&vec![&xs; n_rep], 2)?.reshape((b_sz, n_kv_head * n_rep, seq_len, head_dim))
    }
}

/// Rotary Position Embedding (RoPE) implementation.
#[derive(Debug, Clone)]
struct RoPE {
    cos: Tensor,
    sin: Tensor,
}

impl RoPE {
    fn new(
        dtype: DType,
        head_dim: usize,
        max_seq_len: usize,
        theta: f64,
        device: &Device,
    ) -> Result<Self> {
        let inv_freq: Vec<f32> = (0..head_dim)
            .step_by(2)
            .map(|i| (1.0 / theta.powf(i as f64 / head_dim as f64)) as f32)
            .collect();

        // Compute the length first to avoid borrowing after move.
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), device)?.to_dtype(dtype)?;
        let positions = Tensor::arange(0u32, max_seq_len as u32, device)?
            .to_dtype(dtype)?
            .reshape((max_seq_len, 1))?;
        let angles = positions.matmul(&inv_freq)?;

        Ok(Self {
            cos: angles.cos()?,
            sin: angles.sin()?,
        })
    }

    /// Apply rotary embeddings to query and key tensors.
    /// Shape: (batch, num_heads, seq_len, head_dim)
    fn apply(&self, q: &Tensor, k: &Tensor, position_offset: usize) -> Result<(Tensor, Tensor)> {
        let seq_len = q.dim(2)?;
        let cos = self
            .cos
            .narrow(0, position_offset, seq_len)?
            .to_dtype(q.dtype())?;
        let sin = self
            .sin
            .narrow(0, position_offset, seq_len)?
            .to_dtype(q.dtype())?;

        let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

/// Feed-forward network with SwiGLU activation.
#[derive(Debug, Clone)]
struct FeedForward {
    gate_proj: QMatMul,
    up_proj: QMatMul,
    down_proj: QMatMul,
}

impl FeedForward {
    fn load<R: Read + Seek>(
        content: &gguf_file::Content,
        reader: &mut R,
        layer_prefix: &str,
        device: &Device,
    ) -> Result<Self> {
        let gate_proj = QMatMul::from_qtensor(content.tensor(
            reader,
            &format!("{layer_prefix}.ffn_gate.weight"),
            device,
        )?)?;
        let up_proj = QMatMul::from_qtensor(content.tensor(
            reader,
            &format!("{layer_prefix}.ffn_up.weight"),
            device,
        )?)?;
        let down_proj = QMatMul::from_qtensor(content.tensor(
            reader,
            &format!("{layer_prefix}.ffn_down.weight"),
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
        let gate = self.gate_proj.forward(x)?.apply(&Activation::Silu)?;
        let up = self.up_proj.forward(x)?;
        let hidden = (gate * up)?;
        self.down_proj.forward(&hidden)
    }
}

/// Multi-head attention with Grouped Query Attention support.
#[derive(Debug)]
struct Attention {
    q_proj: QMatMul,
    k_proj: QMatMul,
    v_proj: QMatMul,
    o_proj: QMatMul,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rope: Arc<RoPE>,
}

impl Attention {
    fn load<R: Read + Seek>(
        content: &gguf_file::Content,
        reader: &mut R,
        layer_prefix: &str,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rope: Arc<RoPE>,
        rms_eps: f64,
        device: &Device,
    ) -> Result<Self> {
        let q_proj = QMatMul::from_qtensor(content.tensor(
            reader,
            &format!("{layer_prefix}.attn_q.weight"),
            device,
        )?)?;
        let k_proj = QMatMul::from_qtensor(content.tensor(
            reader,
            &format!("{layer_prefix}.attn_k.weight"),
            device,
        )?)?;
        let v_proj = QMatMul::from_qtensor(content.tensor(
            reader,
            &format!("{layer_prefix}.attn_v.weight"),
            device,
        )?)?;
        let o_proj = QMatMul::from_qtensor(content.tensor(
            reader,
            &format!("{layer_prefix}.attn_output.weight"),
            device,
        )?)?;

        let q_norm = RmsNorm::from_qtensor(
            content.tensor(
                reader,
                &format!("{layer_prefix}.attn_q_norm.weight"),
                device,
            )?,
            rms_eps,
        )?;
        let k_norm = RmsNorm::from_qtensor(
            content.tensor(
                reader,
                &format!("{layer_prefix}.attn_k_norm.weight"),
                device,
            )?,
            rms_eps,
        )?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            head_dim,
            rope,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        position_offset: usize,
        kv_cache: &mut KvCache,
    ) -> Result<Tensor> {
        let (batch, seq_len, _) = hidden_states.dims3()?;

        // Project to Q, K, V
        let queries = self
            .q_proj
            .forward(hidden_states)?
            .reshape((batch, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?; // (batch, num_heads, seq_len, head_dim)

        let keys = self
            .k_proj
            .forward(hidden_states)?
            .reshape((batch, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let values = self
            .v_proj
            .forward(hidden_states)?
            .reshape((batch, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Apply Q/K normalization
        let q_flat = queries.flatten(0, 2)?;
        let k_flat = keys.flatten(0, 2)?;
        let q_normed = self.q_norm.forward(&q_flat)?.reshape(queries.shape())?;
        let k_normed = self.k_norm.forward(&k_flat)?.reshape(keys.shape())?;

        // Apply RoPE
        let (queries, keys) = self.rope.apply(&q_normed, &k_normed, position_offset)?;

        // Update KV cache
        if position_offset == 0 {
            kv_cache.reset();
        }
        let (keys, values) = kv_cache.append(&keys.contiguous()?, &values.contiguous()?)?;

        // Expand KV for Grouped Query Attention
        let num_groups = self.num_heads / self.num_kv_heads;
        let keys = repeat_kv(keys, num_groups)?.contiguous()?;
        let values = repeat_kv(values, num_groups)?.contiguous()?;

        // Compute attention
        let scale = (self.head_dim as f64).sqrt().recip();
        let mut attention_scores = (queries.matmul(&keys.transpose(2, 3)?)? * scale)?;

        if let Some(mask) = attention_mask {
            attention_scores = attention_scores.broadcast_add(mask)?;
        }

        let attention_probs = candle_nn::ops::softmax_last_dim(&attention_scores)?;
        let context = attention_probs.matmul(&values)?;

        // Reshape and project output
        let output =
            context
                .transpose(1, 2)?
                .reshape((batch, seq_len, self.num_heads * self.head_dim))?;

        self.o_proj.forward(&output)
    }
}

/// Single transformer layer with pre-normalization.
#[derive(Debug)]
struct TransformerLayer {
    attention: Attention,
    feed_forward: FeedForward,
    attention_norm: RmsNorm,
    ffn_norm: RmsNorm,
}

impl TransformerLayer {
    fn load<R: Read + Seek>(
        content: &gguf_file::Content,
        reader: &mut R,
        layer_idx: usize,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rope: Arc<RoPE>,
        rms_eps: f64,
        device: &Device,
    ) -> Result<Self> {
        let prefix = format!("blk.{layer_idx}");

        let attention = Attention::load(
            content,
            reader,
            &prefix,
            num_heads,
            num_kv_heads,
            head_dim,
            rope,
            rms_eps,
            device,
        )?;
        let feed_forward = FeedForward::load(content, reader, &prefix, device)?;

        let attention_norm = RmsNorm::from_qtensor(
            content.tensor(reader, &format!("{prefix}.attn_norm.weight"), device)?,
            rms_eps,
        )?;
        let ffn_norm = RmsNorm::from_qtensor(
            content.tensor(reader, &format!("{prefix}.ffn_norm.weight"), device)?,
            rms_eps,
        )?;

        Ok(Self {
            attention,
            feed_forward,
            attention_norm,
            ffn_norm,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        attention_mask: Option<&Tensor>,
        position_offset: usize,
        kv_cache: &mut KvCache,
    ) -> Result<Tensor> {
        // Pre-norm attention
        let normed = self.attention_norm.forward(hidden_states)?;
        let attention_out =
            self.attention
                .forward(&normed, attention_mask, position_offset, kv_cache)?;
        let hidden_states = (hidden_states + attention_out)?;

        // Pre-norm feed-forward
        let normed = self.ffn_norm.forward(&hidden_states)?;
        let ffn_out = self.feed_forward.forward(&normed)?;
        hidden_states + ffn_out
    }
}

/// Shared model weights that can be used across multiple inference contexts.
pub struct ModelWeights {
    embeddings: Embedding,
    layers: Vec<TransformerLayer>,
    final_norm: RmsNorm,
    output_projection: QMatMul,
    device: Device,
    dtype: DType,
    max_seq_len: usize,
}

impl ModelWeights {
    /// Load model weights from a GGUF file.
    pub fn from_gguf<R: Read + Seek>(
        content: gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        // Helper for metadata access
        let get_metadata = |key: &str| -> Result<&gguf_file::Value> {
            content
                .metadata
                .get(key)
                .ok_or_else(|| candle_core::Error::Msg(format!("Missing metadata key: {key}")))
        };

        // Parse model configuration
        let num_heads = get_metadata("qwen3.attention.head_count")?.to_u32()? as usize;
        let num_kv_heads = get_metadata("qwen3.attention.head_count_kv")?.to_u32()? as usize;
        let head_dim = get_metadata("qwen3.attention.key_length")?.to_u32()? as usize;
        let num_layers = get_metadata("qwen3.block_count")?.to_u32()? as usize;
        let hidden_size = get_metadata("qwen3.embedding_length")?.to_u32()? as usize;
        let max_seq_len = get_metadata("qwen3.context_length")?.to_u32()? as usize;
        let rms_eps = get_metadata("qwen3.attention.layer_norm_rms_epsilon")?.to_f32()? as f64;
        let rope_theta = get_metadata("qwen3.rope.freq_base")?.to_f32()? as f64;

        let dtype = match content.metadata.get("general.dtype") {
            Some(v) => match v.to_u32().unwrap_or(1) {
                0 => DType::F32,
                1 => DType::F16,
                _ => DType::F16,
            },
            None => DType::F16,
        };

        // Load embeddings
        let embed_tensor = content.tensor(reader, "token_embd.weight", device)?;
        let embeddings = Embedding::new(embed_tensor.dequantize(device)?, hidden_size);

        // Create RoPE
        let rope = Arc::new(RoPE::new(dtype, head_dim, max_seq_len, rope_theta, device)?);

        // Load transformer layers
        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            layers.push(TransformerLayer::load(
                &content,
                reader,
                i,
                num_heads,
                num_kv_heads,
                head_dim,
                rope.clone(),
                rms_eps,
                device,
            )?);
        }

        // Load final norm and output projection
        let final_norm = RmsNorm::from_qtensor(
            content.tensor(reader, "output_norm.weight", device)?,
            rms_eps,
        )?;

        let output_tensor = content
            .tensor(reader, "output.weight", device)
            .or_else(|_| content.tensor(reader, "token_embd.weight", device))?; // Fallback to tied weights
        let output_projection = QMatMul::from_qtensor(output_tensor)?;

        Ok(Self {
            embeddings,
            layers,
            final_norm,
            output_projection,
            device: device.clone(),
            dtype,
            max_seq_len,
        })
    }

    /// Create a causal attention mask.
    fn create_causal_mask(
        &self,
        batch_size: usize,
        seq_len: usize,
        position_offset: usize,
    ) -> Result<Tensor> {
        let total_len = seq_len + position_offset;

        // Create position indices
        let row_ids = Tensor::arange(0u32, seq_len as u32, &self.device)?
            .to_dtype(DType::I64)?
            .reshape((seq_len, 1))?
            .broadcast_add(&Tensor::new(&[position_offset as i64], &self.device)?)?;

        let col_ids = Tensor::arange(0u32, total_len as u32, &self.device)?
            .to_dtype(DType::I64)?
            .reshape((1, total_len))?;

        // Create causal mask (can only attend to previous positions)
        let mask = row_ids
            .broadcast_as(&[seq_len, total_len])?
            .ge(&col_ids.broadcast_as(&[seq_len, total_len])?)?;

        // Convert to float mask with -inf (F32) for masked positions
        let neg_inf =
            Tensor::new(&[f32::NEG_INFINITY], &self.device)?.broadcast_as(&[seq_len, total_len])?;
        let zero = Tensor::zeros(&[seq_len, total_len], DType::F32, &self.device)?;

        let float_mask = mask.where_cond(&zero, &neg_inf)?;

        // Add batch and head dimensions
        float_mask
            .unsqueeze(0)?
            .unsqueeze(0)?
            .broadcast_as(&[batch_size, 1, seq_len, total_len])
    }
}

pub enum Qwen3Size {
    Size0_6B,
    Size1_7B,
    Size4B,
    Size8B,
    Size14B,
    Size32B,
}

impl Qwen3Size {
    pub fn to_id(&self) -> (String, String) {
        match self {
            Qwen3Size::Size0_6B => (
                "unsloth/Qwen3-0.6B-GGUF".into(),
                "Qwen3-0.6B-Q4_K_M.gguf".into(),
            ),
            Qwen3Size::Size1_7B => (
                "unsloth/Qwen3-1.7B-GGUF".into(),
                "Qwen3-1.7B-Q4_K_M.gguf".into(),
            ),
            Qwen3Size::Size4B => (
                "unsloth/Qwen3-4B-GGUF".into(),
                "Qwen3-4B-Q4_K_M.gguf".into(),
            ),
            Qwen3Size::Size8B => (
                "unsloth/Qwen3-8B-GGUF".into(),
                "Qwen3-8B-Q4_K_M.gguf".into(),
            ),
            Qwen3Size::Size14B => (
                "unsloth/Qwen3-14B-GGUF".into(),
                "Qwen3-14B-Q4_K_M.gguf".into(),
            ),
            Qwen3Size::Size32B => (
                "unsloth/Qwen3-32B-GGUF".into(),
                "Qwen3-32B-Q4_K_M.gguf".into(),
            ),
        }
    }
}

use crate::pipelines::utils::loaders::{GgufModelLoader, TokenizerLoader};

/// High-level Qwen3 model interface for text generation.
/// This struct manages the shared weights and creates individual contexts.
pub struct Qwen3Model {
    weights: Arc<ModelWeights>,
    reasoning: bool,
}

impl Qwen3Model {
    /// Load a Qwen3 model from a GGUF file.
    pub fn from_gguf<R: Read + Seek>(reader: &mut R, device: &Device) -> Result<Self> {
        let content = gguf_file::Content::read(reader)?;
        let weights = Arc::new(ModelWeights::from_gguf(content, reader, device)?);
        Ok(Self {
            weights,
            reasoning: true,
        })
    }

    /// Load the model from hf
    pub fn from_hf(device: &Device, size: Qwen3Size) -> anyhow::Result<Self> {
        let (repo_id, file_name) = size.to_id();

        // Download the model from hf
        let model_loader = GgufModelLoader::new(&repo_id, &file_name);
        let (mut file, content) = model_loader.load()?;

        let weights = Arc::new(ModelWeights::from_gguf(content, &mut file, device)?);
        Ok(Self {
            weights,
            reasoning: true,
        })
    }

    /// Get the models suggested tokenizer
    pub fn get_tokenizer(&self) -> anyhow::Result<Tokenizer> {
        let tokenizer_loader = TokenizerLoader::new("Qwen/Qwen3-0.6B", "tokenizer.json");
        let tokenizer = tokenizer_loader.load()?;
        Ok(tokenizer)
    }

    /// Create a new inference context with this model.
    /// Each context maintains its own KV cache and position tracking.
    pub fn new_context(&self) -> Context {
        Context::new(self.weights.clone())
    }

    /// Create a new context with custom KV cache size.
    pub fn new_context_with_cache_size(&self, cache_size: usize) -> Context {
        Context::with_cache_size(self.weights.clone(), cache_size)
    }

    /// Get model information.
    pub fn info(&self) -> ModelInfo {
        ModelInfo {
            num_layers: self.weights.layers.len(),
            max_seq_len: self.weights.max_seq_len,
            dtype: self.weights.dtype,
            device: self.weights.device.clone(),
        }
    }
}

/// A single inference context with independent state.
/// Multiple contexts can share the same model weights.
pub struct Context {
    weights: Arc<ModelWeights>,
    kv_caches: Vec<KvCache>,
    position: usize,
}

impl Context {
    /// Create a new context with shared weights.
    pub fn new(weights: Arc<ModelWeights>) -> Self {
        let num_layers = weights.layers.len();
        let kv_caches = (0..num_layers)
            .map(|_| KvCache::new(KV_CACHE_DIMS, DEFAULT_CACHE_SIZE))
            .collect();

        Self {
            weights,
            kv_caches,
            position: 0,
        }
    }

    /// Create a new context with custom KV cache size.
    pub fn with_cache_size(weights: Arc<ModelWeights>, cache_size: usize) -> Self {
        let num_layers = weights.layers.len();
        let kv_caches = (0..num_layers)
            .map(|_| KvCache::new(KV_CACHE_DIMS, cache_size))
            .collect();

        Self {
            weights,
            kv_caches,
            position: 0,
        }
    }

    /// Generate next token logits given input token IDs.
    /// Position is tracked automatically within this context.
    ///
    /// # Arguments
    /// * `input_ids` - Token IDs with shape [batch_size, sequence_length]
    ///
    /// # Returns
    /// Logits for next token prediction with shape [batch_size, vocab_size]
    pub fn generate(&mut self, input_ids: &Tensor) -> Result<Tensor> {
        let (batch_size, seq_len) = input_ids.dims2()?;

        // Use current position as offset
        let position_offset = self.position;

        // Embed input tokens
        let mut hidden_states = self.weights.embeddings.forward(input_ids)?;

        // Create attention mask (only needed for multi-token sequences)
        let attention_mask = if seq_len > 1 {
            Some(
                self.weights
                    .create_causal_mask(batch_size, seq_len, position_offset)?,
            )
        } else {
            None
        };

        // Forward pass through transformer layers
        for (layer_idx, layer) in self.weights.layers.iter().enumerate() {
            hidden_states = layer.forward(
                &hidden_states,
                attention_mask.as_ref(),
                position_offset,
                &mut self.kv_caches[layer_idx],
            )?;
        }

        // Final normalization
        hidden_states = self.weights.final_norm.forward(&hidden_states)?;

        // Extract last token and compute logits
        let last_hidden = hidden_states.narrow(1, seq_len - 1, 1)?;
        let logits = self
            .weights
            .output_projection
            .forward(&last_hidden)?
            .squeeze(1)?;

        // Update position after successful generation
        self.position += seq_len;

        Ok(logits)
    }

    /// Reset context state and position counter.
    pub fn reset(&mut self) {
        for cache in &mut self.kv_caches {
            cache.reset();
        }
        self.position = 0;
    }

    /// Get current position in this context.
    pub fn current_position(&self) -> usize {
        self.position
    }

    /// Manually set position (for advanced use cases).
    pub fn set_position(&mut self, position: usize) {
        self.position = position;
    }

    /// Get model information.
    pub fn info(&self) -> ModelInfo {
        ModelInfo {
            num_layers: self.weights.layers.len(),
            max_seq_len: self.weights.max_seq_len,
            dtype: self.weights.dtype,
            device: self.weights.device.clone(),
        }
    }
}

/// Model information structure.
#[derive(Debug, Clone)]
pub struct ModelInfo {
    pub num_layers: usize,
    pub max_seq_len: usize,
    pub dtype: DType,
    pub device: Device,
}

/*

Pipeline Stuff

*/

use crate::pipelines::text_generation_pipeline::text_generation_model::{
    LanguageModelContext, TextGenerationModel, ToggleableReasoning, ToolCalling,
};

use minijinja::{context, Environment};
use serde_json::Value;

impl LanguageModelContext for Context {
    fn generate(&mut self, input: &Tensor) -> candle_core::Result<Tensor> {
        Context::generate(self, input)
    }

    fn reset(&mut self) {
        Context::reset(self);
    }
}

impl TextGenerationModel for Qwen3Model {
    type Context = crate::models::quantized_qwen3::Context;
    type Options = Qwen3Size;

    fn get_eos_token(&self) -> u32 {
        let eos_token = self
            .get_tokenizer()
            .unwrap()
            .encode("<|im_end|>", true)
            .unwrap();
        return eos_token.get_ids()[0];
    }

    fn new(options: Self::Options) -> Self {
        Qwen3Model::from_hf(&candle_core::Device::Cpu, options).unwrap()
    }

    fn get_tokenizer(&self) -> anyhow::Result<tokenizers::Tokenizer> {
        Qwen3Model::get_tokenizer(self)
    }

    fn apply_chat_template(&self, messages: &[crate::Message]) -> anyhow::Result<String> {
        // Determine thinking mode: check for /think or /no_think in the last user message
        let mut enable_thinking = self.reasoning; // Default to model's reasoning flag

        // Check the last user message for soft switches
        if let Some(last_user_msg) = messages.iter().rev().find(|msg| msg.role() == "user") {
            let content = last_user_msg.content();
            if content.contains("/think") {
                enable_thinking = true;
            } else if content.contains("/no_think") {
                enable_thinking = false;
            }
        }

        // Create a loader for the tokenizer config
        let tokenizer_config_loader = crate::pipelines::utils::loaders::HfLoader::new(
            "Qwen/Qwen3-0.6B",
            "tokenizer_config.json",
        );

        // Loads the tokenizer_config.json file and returns the path to the json file as a PathBuf
        let tokenizer_config_path = tokenizer_config_loader
            .load()
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer config: {}", e))?;
        let tokenizer_config_content = std::fs::read_to_string(tokenizer_config_path)
            .map_err(|e| anyhow::anyhow!("Failed to read tokenizer config: {}", e))?;

        // Parse JSON and get the 'chat_template'
        let config_json: Value = serde_json::from_str(&tokenizer_config_content)
            .map_err(|e| anyhow::anyhow!("Failed to parse tokenizer config: {}", e))?;
        let mut chat_template_str = config_json["chat_template"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Missing 'chat_template' field in tokenizer config"))?
            .to_string();

        // Perform targeted template fixes:
        // 1) Reverse list usage messages[::-1] -> messages|reverse
        chat_template_str = chat_template_str.replace("messages[::-1]", "messages|reverse");

        // 2) Convert Python method calls .startswith and .endswith to Jinja tests
        chat_template_str = chat_template_str.replace(
            ".startswith('<tool_response>')",
            " is startingwith \"<tool_response>\"",
        );
        chat_template_str = chat_template_str.replace(
            ".endswith('</tool_response>')",
            " is endingwith \"<tool_response>\"",
        );

        // 3) Targeted fixes for known problematic lines based on the debug output of chat_template_str

        // Fixing: {%- set content = message.content.split('</think>')[-1].lstrip('\n') %}
        chat_template_str = chat_template_str.replace(
            "{%- set content = message.content.split('</think>')[-1].lstrip('\\n') %}",
            "{%- set content = lstrip(last(split(message.content, \"</think>\")), \"\\n\") %}",
        );

        // Fixing: {%- set reasoning_content = message.content.split('</think>')[0].rstrip('\n').split('<think>')[-1].lstrip('\n') %}
        chat_template_str = chat_template_str.replace(
            "{%- set reasoning_content = message.content.split('</think>')[0].rstrip('\\n').split('<think>')[-1].lstrip('\\n') %}",
            "{%- set reasoning_content = lstrip(last(split(rstrip(first(split(message.content, \"</think>\")), \"\\n\"), \"<think>\")), \"\\n\") %}"
        );

        // Fixing: reasoning_content.strip('\n') within the assistant message block
        chat_template_str = chat_template_str.replace(
            "reasoning_content.strip('\\n')",
            "strip(reasoning_content, '\\n')",
        );

        // Fixing: content.lstrip('\n') at the end of the assistant message block
        chat_template_str =
            chat_template_str.replace("+ content.lstrip('\\n') }}", "+ lstrip(content, '\\n') }}");

        // Create a minijinja environment
        let mut env = Environment::new();

        // Register filters and tests for string operations
        // Qwen3 is the only model i've had to do this with so for, it's weird
        // It's jinja template just has weird pythonic stuff in it
        env.add_test("startingwith", |value: &str, prefix: &str| {
            value.starts_with(prefix)
        });
        env.add_test("endingwith", |value: &str, suffix: &str| {
            value.ends_with(suffix)
        });
        env.add_function("split", |v: String, sep: String| -> Vec<String> {
            v.split(&sep).map(String::from).collect()
        });
        env.add_function("first", |v: Vec<String>| -> String {
            v.first().cloned().unwrap_or_default()
        });
        env.add_function("last", |v: Vec<String>| -> String {
            v.last().cloned().unwrap_or_default()
        });
        env.add_function("lstrip", |s: String, pat: String| -> String {
            let c = pat.chars().next().unwrap_or('\0'); // Get first char or null
            s.trim_start_matches(c).to_string()
        });
        env.add_function("rstrip", |s: String, pat: String| -> String {
            let c = pat.chars().next().unwrap_or('\0');
            s.trim_end_matches(c).to_string()
        });
        env.add_function("strip", |s: String, pat: String| -> String {
            let c = pat.chars().next().unwrap_or('\0');
            s.trim_matches(c).to_string()
        });

        // Add the patched template
        env.add_template("chat", &chat_template_str)
            .map_err(|e| anyhow::anyhow!("Failed to add chat template: {}", e))?;

        // Get the template
        let tmpl = env
            .get_template("chat")
            .map_err(|e| anyhow::anyhow!("Failed to get chat template: {}", e))?;

        // Convert messages to a format that works better with Jinja2
        // Also clean up /think and /no_think from user messages for template rendering
        let messages_dicts: Vec<serde_json::Value> = messages
            .into_iter()
            .map(|msg| {
                let mut content = msg.content().to_string();
                // Remove /think and /no_think from user messages for cleaner output
                if msg.role() == "user" {
                    content = content
                        .replace("/think", "")
                        .replace("/no_think", "")
                        .trim()
                        .to_string();
                }
                serde_json::json!({
                    "role": msg.role(),
                    "content": content,
                })
            })
            .collect();

        // Render the template
        let rendered = tmpl
            .render(context! {
                messages => messages_dicts,
                add_generation_prompt => true,
                enable_thinking => enable_thinking,
            })
            .map_err(|e| anyhow::anyhow!("Failed to render chat template: {}", e))?;

        Ok(rendered)
    }

    fn new_context(&self) -> Context {
        Context::new(self.weights.clone())
    }

    fn clear_context(&self, context: &mut Context) -> anyhow::Result<()> {
        context.reset();
        Ok(())
    }
}

impl ToggleableReasoning for Qwen3Model {
    fn set_reasoning(&mut self, enable: bool) -> anyhow::Result<()> {
        self.reasoning = enable;
        Ok(())
    }
}

impl ToolCalling for Qwen3Model {
    fn register_tool(&mut self, tool: String) {
        todo!()
    }

    fn toggle_tools(&mut self, enable: bool) {
        todo!()
    }
}

/*
use crate::utils::loaders::HfLoader;
use minijinja::{context, Environment};
use serde_json::Value;

use crate::models::generate_tokens_from_prompt;
use crate::pipelines::TextGenerationModel;
use crate::Message;

impl TextGenerationModel for Qwen3Model {
    fn load_tokenizer(&self) -> anyhow::Result<tokenizers::Tokenizer> {
        let tokenizer = self.get_tokenizer()?;

        Ok(tokenizer)
    }

    fn get_eos_token_str(&self) -> &str {
        "<|im_end|>"
    }

    fn format_prompt(&self, prompt: &str) -> String {
        format!("<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n")
    }

    fn format_messages(&self, messages: Vec<Message>) -> anyhow::Result<String> {


    fn prompt_with_tokens(
        &self,
        prompt_tokens: &[u32],
        max_len: usize,
        eos_token: u32,
    ) -> anyhow::Result<Vec<u32>> {
        let mut pipeline_state_guard = self.pipeline_state.write();

        let response_tokens = generate_tokens_from_prompt(
            prompt_tokens,
            &self.config.params,
            &mut *pipeline_state_guard,
            max_len,
            &self.config.device,
            eos_token,
        )?;

        Ok(response_tokens)
    }
}
*/
