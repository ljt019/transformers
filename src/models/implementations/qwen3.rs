//! High-performance Qwen3 implementation with quantization support.
//!
//! This implementation provides efficient inference for Qwen3 models with:
//! - Quantized weights for reduced memory usage
//! - KV caching for autoregressive generation
//! - Multi-context support for concurrent conversations
//! - GPU acceleration via Candle framework
//!

use crate::core::cache::ModelOptions;
use crate::loaders::GenerationConfig;
use crate::models::components::{
    attention::KvCache, 
    Embedding, QMatMul, RmsNorm, VarBuilder,
    common_layers::{RoPE, RoPEParams, Qwen3RoPEParams, FeedForward as CommonFeedForward, Attention as CommonAttention, AttentionConfig}
};
use crate::pipelines::text_generation_pipeline::Tool;
use crate::Message;
use anyhow::anyhow;
use async_trait::async_trait;
use candle_core::{quantized::gguf_file, DType, Device, IndexOp, Module, Result, Tensor, D};
use minijinja::{context, Environment, UndefinedBehavior};
use minijinja_contrib::pycompat;
use std::collections::HashMap;
use std::io::{Read, Seek};
use std::sync::Arc;
use tokenizers::Tokenizer;

// Constants
const DEFAULT_CACHE_SIZE: usize = 64;
const KV_CACHE_DIMS: usize = 2;

// Type aliases for readability
type Qwen3RoPE = RoPE<Qwen3RoPEParams>;
type Qwen3Attention = CommonAttention<Qwen3RoPEParams>;
type FeedForward = CommonFeedForward;

/// Repeats a key or value tensor for grouped query attention
/// The input tensor should have a shape `(batch, num_kv_heads, seq_len, head_dim)`,
pub fn repeat_kv(xs: Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(xs);
    }
    let (b_sz, n_kv_head, seq_len, head_dim) = xs.dims4()?;
    let xs = xs
        .unsqueeze(2)?
        .expand((b_sz, n_kv_head, n_rep, seq_len, head_dim))?
        .reshape((b_sz, n_kv_head * n_rep, seq_len, head_dim))?;
    Ok(xs)
}

/// Transformer layer containing attention and feed-forward sub-layers.
#[derive(Debug)]
struct TransformerLayer {
    attention: Qwen3Attention,
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
        rope: Arc<Qwen3RoPE>,
        rms_eps: f64,
        device: &Device,
    ) -> Result<Self> {
        use crate::models::components::common_layers::WeightNaming;
        
        let prefix = format!("blk.{layer_idx}");
        let naming = WeightNaming::qwen3();

        let attention = Qwen3Attention::load_with_naming(
            content,
            reader,
            &prefix,
            AttentionConfig {
                num_heads,
                num_kv_heads,
                head_dim,
                sliding_window_size: None,
                rms_eps,
            },
            rope,
            device,
            true, // use_qk_norm
            &naming,
        )?;

        let feed_forward = FeedForward::load_with_naming(
            content,
            reader,
            &prefix,
            device,
            &naming,
        )?;

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

/// Main model weights structure containing all layers.
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
    /// Load model weights from GGUF format.
    pub fn from_gguf<R: Read + Seek>(
        content: gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        let metadata = &content.metadata;
        let dtype = metadata
            .get("general.file_type")
            .and_then(|v| v.to_u32().ok())
            .map(|v| match v {
                2 => DType::F32,
                _ => DType::F16,
            })
            .unwrap_or(DType::F32);

        // Model configuration from metadata
        let num_layers = metadata
            .get("llm.block_count")
            .ok_or_else(|| anyhow!("missing layer count"))?
            .to_u32()? as usize;
        let num_heads = metadata
            .get("llm.attention.head_count")
            .ok_or_else(|| anyhow!("missing head count"))?
            .to_u32()? as usize;
        let num_kv_heads = metadata
            .get("llm.attention.head_count_kv")
            .ok_or_else(|| anyhow!("missing kv head count"))?
            .to_u32()? as usize;
        let head_dim = metadata
            .get("llm.rope.dimension_count")
            .ok_or_else(|| anyhow!("missing head dimension"))?
            .to_u32()? as usize;
        let rms_eps = metadata
            .get("llm.attention.layer_norm_rms_epsilon")
            .ok_or_else(|| anyhow!("missing rms epsilon"))?
            .to_f32()? as f64;
        let max_seq_len = metadata
            .get("llm.context_length")
            .and_then(|v| v.to_u32().ok())
            .unwrap_or(2048) as usize;

        // Create shared RoPE instance
        let rope = Arc::new(Qwen3RoPE::new(
            dtype,
            head_dim,
            max_seq_len,
            Qwen3RoPEParams,
            device,
        )?);

        // Load embeddings
        let embeddings = {
            let tensor = content.tensor(reader, "token_embd.weight", device)?;
            let weight = tensor.dequantize(device)?;
            Embedding::new(weight, weight.dim(1)?)
        };

        // Load transformer layers
        let mut layers = Vec::with_capacity(num_layers);
        for layer_idx in 0..num_layers {
            layers.push(TransformerLayer::load(
                &content,
                reader,
                layer_idx,
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
        let output_projection = QMatMul::from_weights(
            content.tensor(reader, "output.weight", device)?,
        )?;

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

    /// Forward pass that returns `[batch, hidden]` embeddings.
    pub fn forward_embedding(
        &self,
        input_ids: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (_, t) = input_ids.dims2()?;
        if t == 0 {
            return Err(candle_core::Error::Msg("Input tensor has zero sequence length".to_string()));
        }
        let mut hidden = self.embeddings.forward(input_ids)?;
        let mut empty_cache = KvCache::new(2, 1024);

        for layer in self.layers.iter() {
            hidden = layer.forward(&hidden, attention_mask, 0, &mut empty_cache)?;
        }
        hidden = self.final_norm.forward(&hidden)?;
        let last = hidden.narrow(1, t - 1, 1)?;
        last.squeeze(1)
    }
  
    /// Forward pass returning logits for all tokens in the sequence.
    pub fn forward_logits(
        &self,
        input_ids: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (_, t) = input_ids.dims2()?;
        if t == 0 {
            return Err(candle_core::Error::Msg("Input tensor has zero sequence length".to_string()));
        }
        let mut hidden = self.embeddings.forward(input_ids)?;
        let mut empty_cache = KvCache::new(2, 1024);

        for layer in self.layers.iter() {
            hidden = layer.forward(&hidden, attention_mask, 0, &mut empty_cache)?;
        }
        hidden = self.final_norm.forward(&hidden)?;
        
        // Get logits for all tokens by applying output projection
        let logits = self.output_projection.forward(&hidden)?;
        Ok(logits)
    }
}

#[derive(Debug, Clone, Copy)]
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

impl std::fmt::Display for Qwen3Size {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            Qwen3Size::Size0_6B => "qwen3-0.6b",
            Qwen3Size::Size1_7B => "qwen3-1.7b",
            Qwen3Size::Size4B => "qwen3-4b",
            Qwen3Size::Size8B => "qwen3-8b",
            Qwen3Size::Size14B => "qwen3-14b",
            Qwen3Size::Size32B => "qwen3-32b",
        };
        write!(f, "{name}")
    }
}

impl crate::core::ModelOptions for Qwen3Size {
    fn cache_key(&self) -> String {
        self.to_string()
    }
}

use crate::loaders::{GgufModelLoader, TokenizerLoader};

/// High-level Qwen3 model interface for text generation.
/// This struct manages the shared weights and creates individual contexts.
#[derive(Clone)]
pub struct Qwen3Model {
    weights: Arc<ModelWeights>,
    reasoning: bool,
    generation_config: crate::core::GenerationConfig,
    tools: Vec<crate::pipelines::text_generation_pipeline::Tool>,
    chat_template_env: Arc<Environment<'static>>,
}

impl Qwen3Model {
    /// Load and prepare the chat template environment
    async fn load_chat_template_env() -> anyhow::Result<Arc<Environment<'static>>> {
        // Load the tokenizer config and extract the chat template
        let tokenizer_config_loader =
            crate::loaders::HfLoader::new("Qwen/Qwen3-0.6B", "tokenizer_config.json");

        let tokenizer_config_path = tokenizer_config_loader.load().await?;
        let tokenizer_config_content = std::fs::read_to_string(tokenizer_config_path)?;
        let config_json: serde_json::Value = serde_json::from_str(&tokenizer_config_content)?;

        let chat_template_str = config_json["chat_template"]
            .as_str()
            .ok_or_else(|| anyhow::anyhow!("Missing 'chat_template' field in tokenizer config"))?;

        let mut chat_template_owned = chat_template_str.to_string();

        // Replace Python list reverse slice with Jinja filter
        chat_template_owned = chat_template_owned.replace("messages[::-1]", "messages|reverse");

        // Patch known problematic arithmetic producing floats
        chat_template_owned = chat_template_owned.replace(
            "(messages|length - 1) - loop.index0",
            "((messages|length - 1)|int - loop.index0|int)",
        );

        // Replace Python negative index access messages[-1] with explicit last element index
        chat_template_owned =
            chat_template_owned.replace("messages[-1]", "messages[(messages|length - 1)]");

        // Build the MiniJinja environment with Python compatibility helpers
        let mut env = Environment::new();
        env.set_undefined_behavior(UndefinedBehavior::Lenient);

        add_to_environment(&mut env);
        env.set_unknown_method_callback(pycompat::unknown_method_callback);

        // Ensure `tojson` filter is available (requires json feature)
        env.add_filter("tojson", minijinja::filters::tojson);

        // Leak the string to get 'static lifetime - this is fine since we're storing it in the model
        let chat_template_static = Box::leak(chat_template_owned.into_boxed_str());
        env.add_template("chat", chat_template_static)?;

        Ok(Arc::new(env))
    }
    /// Load a Qwen3 model from a GGUF file.
    pub async fn from_gguf<R: Read + Seek>(
        reader: &mut R,
        device: &Device,
    ) -> anyhow::Result<Self> {
        let content = gguf_file::Content::read(reader)?;
        let weights = Arc::new(ModelWeights::from_gguf(content, reader, device)?);
        let generation_config = crate::loaders::GenerationConfigLoader::new(
            "Qwen/Qwen3-0.6B",
            "generation_config.json",
        )
        .load()
        .await?;
        let chat_template_env = Self::load_chat_template_env().await?;
        Ok(Self {
            weights,
            reasoning: true,
            generation_config,
            tools: Vec::new(),
            chat_template_env,
        })
    }

    /// Load the model from hf
    pub async fn from_hf(device: &Device, size: Qwen3Size) -> anyhow::Result<Self> {
        let (repo_id, file_name) = size.to_id();

        // Download the model from hf
        let model_loader = GgufModelLoader::new(&repo_id, &file_name);
        let (mut file, content) = model_loader.load().await?;

        // Download the tokenizer config from hf to get the eos token id
        let generation_config = crate::loaders::GenerationConfigLoader::new(
            "Qwen/Qwen3-0.6B",
            "generation_config.json",
        )
        .load()
        .await?;

        let weights = Arc::new(ModelWeights::from_gguf(content, &mut file, device)?);
        let chat_template_env = Self::load_chat_template_env().await?;
        Ok(Self {
            weights,
            reasoning: true,
            generation_config,
            tools: Vec::new(),
            chat_template_env,
        })
    }

    /// Get the models suggested tokenizer
    pub async fn get_tokenizer(&self) -> anyhow::Result<Tokenizer> {
        let tokenizer_loader = TokenizerLoader::new("Qwen/Qwen3-0.6B", "tokenizer.json");
        let tokenizer = tokenizer_loader.load().await?;
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

use crate::pipelines::text_generation_pipeline::model::{
    LanguageModelContext, TextGenerationModel, ToggleableReasoning, ToolCalling,
};

impl LanguageModelContext for Context {
    fn generate(&mut self, input: &Tensor) -> candle_core::Result<Tensor> {
        Context::generate(self, input)
    }

    fn reset(&mut self) {
        Context::reset(self);
    }

    fn position(&self) -> usize {
        self.position
    }

    fn can_continue_from(&self, position: usize) -> bool {
        // Check if we can continue from the given position
        // The cache is valid if the requested position matches our current position
        self.position == position
    }
}

#[async_trait]
impl TextGenerationModel for Qwen3Model {
    type Context = Context;
    type Options = Qwen3Size;

    fn get_eos_token(&self) -> u32 {
        // Return the first EOS token ID from the generation config
        self.generation_config.eos_token_ids[0] as u32
    }

    fn get_eos_tokens(&self) -> Vec<u32> {
        // Return all EOS token IDs for robust termination detection
        self.generation_config
            .eos_token_ids
            .iter()
            .map(|&id| id as u32)
            .collect()
    }

    fn get_max_seq_len(&self) -> usize {
        self.weights.max_seq_len
    }

    async fn new(options: Self::Options) -> anyhow::Result<Self> {
        Qwen3Model::from_hf(&candle_core::Device::Cpu, options).await
    }

    async fn get_tokenizer(&self) -> anyhow::Result<tokenizers::Tokenizer> {
        Qwen3Model::get_tokenizer(self).await
    }

    fn apply_chat_template(&self, messages: &[crate::Message]) -> anyhow::Result<String> {
        // Determine thinking mode
        let mut enable_thinking = self.reasoning;
        if let Some(last_user_msg) = messages.iter().rev().find(|msg| msg.role() == "user") {
            let content = last_user_msg.content();
            if content.contains("/think") {
                enable_thinking = true;
            } else if content.contains("/no_think") {
                enable_thinking = false;
            }
        }

        // Prepare messages (strip /think flags from user content)
        let messages_dicts: Vec<serde_json::Value> = messages
            .iter()
            .map(|msg| {
                let mut content = msg.content().to_string();
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

        // Render the template using the pre-loaded environment
        let rendered = self
            .chat_template_env
            .get_template("chat")?
            .render(context! {
                messages => messages_dicts,
                add_generation_prompt => true,
                enable_thinking => enable_thinking,
                tools => self.registered_tools(),
            })?;

        Ok(rendered)
    }

    fn new_context(&self) -> Context {
        Context::new(self.weights.clone())
    }

    fn clear_context(&self, context: &mut Context) -> anyhow::Result<()> {
        context.reset();
        Ok(())
    }

    fn default_generation_params(&self) -> crate::models::generation::GenerationParams {
        // Recommended Qwen3 inference settings (per official guidance)
        crate::models::generation::GenerationParams {
            temperature: self.generation_config.temperature.unwrap_or(0.6),
            repeat_penalty: self.generation_config.repeat_penalty.unwrap_or(1.1), // presence_penalty analogue
            repeat_last_n: self.generation_config.repeat_last_n.unwrap_or(64),
            seed: 42,
            max_len: 2048, // Qwen3 context length
            top_p: self.generation_config.top_p.unwrap_or(0.95),
            top_k: self.generation_config.top_k.unwrap_or(20) as usize,
            min_p: self.generation_config.min_p.unwrap_or(0.0),
        }
    }
}

impl ToggleableReasoning for Qwen3Model {
    fn set_reasoning(&mut self, enable: bool) -> anyhow::Result<()> {
        self.reasoning = enable;
        Ok(())
    }
}

use crate::core::ToolError;
use crate::pipelines::text_generation_pipeline::model::Tool;
use async_trait::async_trait;

impl ToolCalling for Qwen3Model {
    fn register_tool(&mut self, tool: Tool) -> anyhow::Result<()> {
        // Replace existing tool with same name if present
        if let Some(pos) = self.tools.iter().position(|t| t.name() == tool.name()) {
            self.tools[pos] = tool;
        } else {
            self.tools.push(tool);
        }
        Ok(())
    }

    fn unregister_tool(&mut self, name: &str) -> anyhow::Result<()> {
        if let Some(pos) = self.tools.iter().position(|t| t.name() == name) {
            self.tools.remove(pos);
        }
        Ok(())
    }

    fn clear_tools(&mut self) -> anyhow::Result<()> {
        self.tools.clear();
        Ok(())
    }

    fn registered_tools(&self) -> Vec<Tool> {
        self.tools.clone()
    }

    fn call_tool(
        &mut self,
        tool_name: String,
        parameters: std::collections::HashMap<String, String>,
    ) -> std::result::Result<String, ToolError> {
        if let Some(tool) = self.tools.iter().find(|t| t.name() == tool_name) {
            tool.call(parameters)
        } else {
            Err(ToolError::Message(format!(
                "Tool '{tool_name}' is not registered"
            )))
        }
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
