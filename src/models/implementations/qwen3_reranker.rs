use std::io::{Read, Seek};
use std::sync::{Arc, Mutex};

use candle_core::quantized::{gguf_file, QMatMul, QTensor};
use candle_core::{DType, Device, Result, Tensor};
use candle_nn::{kv_cache::KvCache, Activation, Embedding, Module};
use tokenizers::Tokenizer;

use crate::loaders::{GgufModelLoader, TokenizerLoader};
use crate::models::{repeat_kv, RmsNorm};
use crate::pipelines::reranker_pipeline::model::RerankModel;
use crate::pipelines::reranker_pipeline::pipeline::RerankResult;

#[derive(Debug, Clone, Copy)]
pub enum Qwen3RerankSize {
    Size0_6B,
    Size4B,
    Size8B,
}

impl Qwen3RerankSize {
    pub fn to_id(&self) -> (String, String) {
        match self {
            Qwen3RerankSize::Size0_6B => (
                "Mungert/Qwen3-Reranker-0.6B-GGUF".into(),
                "Qwen3-Reranker-0.6B-q4_k_m.gguf".into(),
            ),
            Qwen3RerankSize::Size4B => (
                "Mungert/Qwen3-Reranker-4B-GGUF".into(),
                "Qwen3-Reranker-4B-q4_k_m.gguf".into(),
            ),
            Qwen3RerankSize::Size8B => (
                "Mungert/Qwen3-Reranker-8B-GGUF".into(),
                "Qwen3-Reranker-8B-q4_k_m.gguf".into(),
            ),
        }
    }
}

impl std::fmt::Display for Qwen3RerankSize {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let name = match self {
            Qwen3RerankSize::Size0_6B => "qwen3-reranker-0.6b",
            Qwen3RerankSize::Size4B => "qwen3-reranker-4b",
            Qwen3RerankSize::Size8B => "qwen3-reranker-8b",
        };
        write!(f, "{name}")
    }
}

impl crate::core::ModelOptions for Qwen3RerankSize {
    fn cache_key(&self) -> String {
        self.to_string()
    }
}

/*
Minimal Model Implementation
*/

struct Gguf<R: Read + Seek> {
    ct: gguf_file::Content,
    reader: R,
    device: Device,
}

impl<R: Read + Seek> Gguf<R> {
    fn new(ct: gguf_file::Content, reader: R, device: Device) -> Self {
        Self { ct, reader, device }
    }

    fn qmatmul(&mut self, name: &str) -> Result<QMatMul> {
        let ws = self.ct.tensor(&mut self.reader, name, &self.device)?;
        QMatMul::from_qtensor(ws)
    }

    fn rms_norm(&mut self, name: &str, eps: f64) -> Result<RmsNorm> {
        let ws = self.ct.tensor(&mut self.reader, name, &self.device)?;
        RmsNorm::from_qtensor(ws, eps)
    }

    fn metadata(&self) -> &std::collections::HashMap<String, gguf_file::Value> {
        &self.ct.metadata
    }

    fn tensor(&mut self, name: &str) -> Result<QTensor> {
        self.ct.tensor(&mut self.reader, name, &self.device)
    }
}

#[derive(Debug, Clone)]
struct MlpWeights {
    gate_proj: QMatMul,
    up_proj: QMatMul,
    down_proj: QMatMul,
    act_fn: Activation,
    span: tracing::Span,
}

impl MlpWeights {
    fn new<R: Read + Seek>(gg: &mut Gguf<R>, prefix: &str) -> Result<Self> {
        let gate_proj = gg.qmatmul(&format!("{prefix}.ffn_gate.weight"))?;
        let up_proj = gg.qmatmul(&format!("{prefix}.ffn_up.weight"))?;
        let down_proj = gg.qmatmul(&format!("{prefix}.ffn_down.weight"))?;
        let act_fn = Activation::Silu;
        let span = tracing::span!(tracing::Level::TRACE, "mlp");
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            act_fn,
            span,
        })
    }
}

impl Module for MlpWeights {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let _enter = self.span.enter();
        let gate = self.gate_proj.forward(x)?.apply(&self.act_fn)?;
        let up = self.up_proj.forward(x)?;
        let gated = (gate * up)?;
        self.down_proj.forward(&gated)
    }
}

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(
        dtype: DType,
        head_dim: usize,
        max_position_embeddings: usize,
        rope_theta: f64,
        dev: &Device,
    ) -> Result<Self> {
        let dim = head_dim;
        let max_seq_len = max_position_embeddings;
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / rope_theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(dtype)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(dtype)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?,
            cos: freqs.cos()?,
        })
    }

    /// Apply RoPE (q, k shape: B x H x L x D)
    fn apply(&self, q: &Tensor, k: &Tensor, offset: usize) -> Result<(Tensor, Tensor)> {
        let (_, _, seq_len, _) = q.dims4()?;
        let cos = self.cos.narrow(0, offset, seq_len)?.to_dtype(q.dtype())?;
        let sin = self.sin.narrow(0, offset, seq_len)?.to_dtype(q.dtype())?;
        let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

#[derive(Debug, Clone)]
struct AttentionWeights {
    q_proj: QMatMul,
    k_proj: QMatMul,
    v_proj: QMatMul,
    o_proj: QMatMul,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    rotary_emb: Arc<RotaryEmbedding>,
    kv_cache: KvCache,
    span_attn: tracing::Span,
}

impl AttentionWeights {
    fn new<R: Read + Seek>(
        gg: &mut Gguf<R>,
        num_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        rms_norm_eps: f64,
        rotary_emb: Arc<RotaryEmbedding>,
        prefix: &str,
    ) -> Result<Self> {
        let num_kv_groups = num_heads / num_kv_heads;

        let q_proj = gg.qmatmul(&format!("{prefix}.attn_q.weight"))?;
        let k_proj = gg.qmatmul(&format!("{prefix}.attn_k.weight"))?;
        let v_proj = gg.qmatmul(&format!("{prefix}.attn_v.weight"))?;
        let o_proj = gg.qmatmul(&format!("{prefix}.attn_output.weight"))?;

        let q_norm = gg.rms_norm(&format!("{prefix}.attn_q_norm.weight"), rms_norm_eps)?;
        let k_norm = gg.rms_norm(&format!("{prefix}.attn_k_norm.weight"), rms_norm_eps)?;

        // Initialize KV cache with 512 tokens capacity to reduce initial memory allocation.
        // The cache will grow in chunks of 512 tokens when needed.
        let kv_cache = KvCache::new(2, 512);

        let span_attn = tracing::span!(tracing::Level::TRACE, "attn");

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            num_kv_groups,
            head_dim,
            rotary_emb,
            kv_cache,
            span_attn,
        })
    }

    fn forward(&mut self, x: &Tensor, attn_mask: Option<&Tensor>, offset: usize) -> Result<Tensor> {
        let _enter = self.span_attn.enter();
        let (b, l, _) = x.dims3()?;

        let q = self.q_proj.forward(x)?;
        let k = self.k_proj.forward(x)?;
        let v = self.v_proj.forward(x)?;

        let q = q
            .reshape((b, l, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b, l, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b, l, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        let q_flat = q.flatten(0, 2)?;
        let k_flat = k.flatten(0, 2)?;

        let q_flat = self.q_norm.forward(&q_flat)?;
        let k_flat = self.k_norm.forward(&k_flat)?;
        let q = q_flat.reshape((b, self.num_heads, l, self.head_dim))?;
        let k = k_flat.reshape((b, self.num_kv_heads, l, self.head_dim))?;

        let (q, k) = self.rotary_emb.apply(&q, &k, offset)?;

        // Reset KV cache if we're at the first position
        if offset == 0 {
            self.kv_cache.reset();
        }
        let (k, v) = self.kv_cache.append(&k.contiguous()?, &v.contiguous()?)?;

        // Make tensor contiguous to avoid some strided copies
        let k = k.contiguous()?;
        let v = v.contiguous()?;

        let k = repeat_kv(k, self.num_kv_groups)?.contiguous()?;
        let v = repeat_kv(v, self.num_kv_groups)?.contiguous()?;

        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let mut scores = (q.matmul(&k.transpose(2, 3)?)? * scale)?;
        if let Some(m) = attn_mask {
            let m_dtype = m.dtype();
            let scores_dtype = scores.dtype();
            let mask = if m_dtype != scores_dtype {
                m.to_dtype(scores_dtype)?
            } else {
                m.clone()
            };
            scores = scores.broadcast_add(&mask)?;
        }
        let probs = candle_nn::ops::softmax_last_dim(&scores)?;
        let ctx = probs.matmul(&v)?; // (B, H, L, D)
        let reshaped_ctx = ctx
            .transpose(1, 2)?
            .reshape((b, l, self.num_heads * self.head_dim))?;
        self.o_proj.forward(&reshaped_ctx)
    }
}

#[derive(Debug, Clone)]
struct LayerWeights {
    self_attn: AttentionWeights,
    mlp: MlpWeights,
    ln1: RmsNorm,
    ln2: RmsNorm,
}

impl LayerWeights {
    fn new<R: Read + Seek>(
        gg: &mut Gguf<R>,
        num_attention_heads: usize,
        num_key_value_heads: usize,
        head_dim: usize,
        rms_norm_eps: f64,
        rotary: Arc<RotaryEmbedding>,
        layer_idx: usize,
    ) -> Result<Self> {
        let prefix = format!("blk.{layer_idx}");

        let ln1 = gg.rms_norm(&format!("{prefix}.attn_norm.weight"), rms_norm_eps)?;
        let ln2 = gg.rms_norm(&format!("{prefix}.ffn_norm.weight"), rms_norm_eps)?;
        let self_attn = AttentionWeights::new(
            gg,
            num_attention_heads,
            num_key_value_heads,
            head_dim,
            rms_norm_eps,
            rotary,
            &prefix,
        )?;
        let mlp = MlpWeights::new(gg, &prefix)?;
        Ok(Self {
            self_attn,
            mlp,
            ln1,
            ln2,
        })
    }

    fn forward(&mut self, x: &Tensor, mask: Option<&Tensor>, offset: usize) -> Result<Tensor> {
        let h = self.ln1.forward(x)?;
        let h = self.self_attn.forward(&h, mask, offset)?;
        let x = (x + h)?;
        let h2 = self.ln2.forward(&x)?;
        let h2 = h2.apply(&self.mlp)?;
        x + h2
    }
}

#[derive(Debug, Clone)]
struct ModelWeights {
    embed_tokens: Embedding,
    layers: Vec<LayerWeights>,
    norm: RmsNorm,
    lm_head: QMatMul,
    device: Device,
    dtype: DType,
    span: tracing::Span,
    span_output: tracing::Span,
}

impl ModelWeights {
    fn from_gguf<R: Read + Seek>(
        ct: gguf_file::Content,
        reader: &mut R,
        device: &Device,
    ) -> Result<Self> {
        let mut gg = Gguf::new(ct, reader, device.clone());
        let md_get = |s: &str| match gg.metadata().get(s) {
            None => candle_core::bail!("cannot find {s} in metadata"),
            Some(v) => Ok(v),
        };

        let num_attention_heads = md_get("qwen3.attention.head_count")?.to_u32()? as usize;
        let num_kv_heads = md_get("qwen3.attention.head_count_kv")?.to_u32()? as usize;
        let head_dim = md_get("qwen3.attention.key_length")?.to_u32()? as usize;
        let num_layers = md_get("qwen3.block_count")?.to_u32()? as usize;
        let hidden_size = md_get("qwen3.embedding_length")?.to_u32()? as usize;
        let max_position_embeddings = md_get("qwen3.context_length")?.to_u32()? as usize;
        let rms_norm_eps = md_get("qwen3.attention.layer_norm_rms_epsilon")?.to_f32()? as f64;
        let rope_freq_base = md_get("qwen3.rope.freq_base")?.to_f32()? as f64;

        let dtype = match gg.metadata().get("general.dtype") {
            Some(v) => match v.to_u32() {
                Ok(0) => DType::F32,
                Ok(1) => DType::F16,
                _ => DType::F16,
            },
            None => DType::F16,
        };

        let embed_tensor = gg.tensor("token_embd.weight")?;
        let embed_tokens = Embedding::new(embed_tensor.dequantize(device)?, hidden_size);

        let rotary = Arc::new(RotaryEmbedding::new(
            dtype,
            head_dim,
            max_position_embeddings,
            rope_freq_base,
            device,
        )?);

        let mut layers = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            layers.push(LayerWeights::new(
                &mut gg,
                num_attention_heads,
                num_kv_heads,
                head_dim,
                rms_norm_eps,
                rotary.clone(),
                i,
            )?);
        }

        let norm = gg.rms_norm("output_norm.weight", rms_norm_eps)?;
        // Load output projection tensor, falling back to tied embeddings like gemma3
        let lm_head_tensor = match gg.tensor("output.weight") {
            Ok(tensor) => tensor,
            Err(_) => gg.tensor("token_embd.weight")?,
        };
        let lm_head = QMatMul::from_qtensor(lm_head_tensor)?;
        let span = tracing::span!(tracing::Level::TRACE, "model");
        let span_output = tracing::span!(tracing::Level::TRACE, "output");
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            device: device.clone(),
            dtype,
            span,
            span_output,
        })
    }

    fn forward(&mut self, input: &Tensor, offset: usize) -> Result<Tensor> {
        let _enter = self.span.enter();
        let (b, l) = input.dims2()?;
        let mut h = self.embed_tokens.forward(input)?;
        let causal_mask = if l == 1 {
            None
        } else {
            Some(self.causal_mask(b, l, offset, None)?)
        };
        for layer in &mut self.layers {
            h = layer.forward(&h, causal_mask.as_ref(), offset)?;
        }
        let h = self.norm.forward(&h)?;
        let _enter = self.span_output.enter();
        let last_hidden = h.narrow(1, l - 1, 1)?;
        self.lm_head.forward(&last_hidden)?.squeeze(1)
    }

    fn causal_mask(
        &self,
        b: usize,
        tgt: usize,
        offset: usize,
        sw: Option<usize>,
    ) -> Result<Tensor> {
        let minf = f32::NEG_INFINITY;
        let mask: Vec<_> = (0..tgt)
            .flat_map(|i| {
                (0..(tgt + offset)).map(move |j| {
                    let past_ok = j <= i + offset;
                    let sw_ok = match sw {
                        Some(w) => (i + offset) as i64 - j as i64 <= w as i64,
                        None => true,
                    };
                    if past_ok && sw_ok {
                        0.
                    } else {
                        minf
                    }
                })
            })
            .collect();
        Tensor::from_slice(&mask, (b, 1, tgt, tgt + offset), &self.device)?.to_dtype(self.dtype)
    }
}

/// Qwen3 model for reranking text pairs using cross-encoder architecture.
#[derive(Clone)]
pub struct Qwen3RerankModel {
    weights: Arc<Mutex<ModelWeights>>,
    device: Device,
}

impl Qwen3RerankModel {
    pub async fn from_hf(device: &Device, size: Qwen3RerankSize) -> anyhow::Result<Self> {
        let (repo_id, file_name) = size.to_id();
        let loader = GgufModelLoader::new(&repo_id, &file_name);
        let (mut file, content) = loader.load().await?;
        let weights = Arc::new(Mutex::new(ModelWeights::from_gguf(
            content, &mut file, device,
        )?));
        Ok(Self {
            weights,
            device: device.clone(),
        })
    }

    pub async fn get_tokenizer(&self) -> anyhow::Result<Tokenizer> {
        let loader = TokenizerLoader::new("Qwen/Qwen3-0.6B", "tokenizer.json");
        loader.load().await
    }

    /// Rerank a list of documents against a query using cross-encoder architecture.
    /// Returns a list of (document_index, relevance_score) pairs sorted by relevance.
    pub fn rerank_documents(
        &self,
        tokenizer: &Tokenizer,
        query: &str,
        documents: &[&str],
    ) -> anyhow::Result<Vec<RerankResult>> {
        let mut scored_docs = Vec::new();

        for (idx, document) in documents.iter().enumerate() {
            let score = self.compute_relevance_score(tokenizer, query, document)?;
            scored_docs.push(RerankResult { index: idx, score });
        }

        // Sort by relevance score in descending order
        scored_docs.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(scored_docs)
    }

    /// Compute relevance score for a query-document pair using cross-encoder.
    fn compute_relevance_score(
        &self,
        tokenizer: &Tokenizer,
        query: &str,
        document: &str,
    ) -> anyhow::Result<f32> {
        let instruction = "Given a query, retrieve relevant passages that answer the query";

        // Format with system prompt and proper delimiters
        let prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n";
        let suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";

        let prompt_str = format!(
            "{prefix}<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}{suffix}"
        );

        let tokens = tokenizer
            .encode(prompt_str, true)
            .map_err(anyhow::Error::msg)?;

        let token_ids = tokens.get_ids();

        // Run a single forward pass to get logits
        let input = Tensor::new(token_ids, &self.device)?.unsqueeze(0)?;

        let logits = self.weights.lock().unwrap().forward(&input, 0)?;
        let logits = logits.squeeze(0)?;

        // Get token IDs for "yes" and "no" for scoring
        let yes_token_id = *tokenizer.get_vocab(true).get("yes").unwrap() as u32;
        let no_token_id = *tokenizer.get_vocab(true).get("no").unwrap() as u32;

        // Extract logits for "yes" and "no" tokens
        let yes_logit = logits.get(yes_token_id as usize)?.to_scalar::<f32>()?;
        let no_logit = logits.get(no_token_id as usize)?.to_scalar::<f32>()?;

        // Compute score as difference (equivalent to binary classification)
        // Higher score means more relevant
        let score = yes_logit - no_logit;

        Ok(score)
    }

    /// Batch reranking for better efficiency with multiple queries.
    pub fn batch_rerank(
        &self,
        tokenizer: &Tokenizer,
        queries: &[&str],
        documents: &[&str],
    ) -> anyhow::Result<Vec<Vec<RerankResult>>> {
        let mut results = Vec::new();

        for query in queries {
            let ranked_docs = self.rerank_documents(tokenizer, query, documents)?;
            results.push(ranked_docs);
        }

        Ok(results)
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}

impl RerankModel for Qwen3RerankModel {
    type Options = Qwen3RerankSize;

    fn new(options: Self::Options, device: Device) -> anyhow::Result<Self> {
        futures::executor::block_on(Self::from_hf(&device, options))
    }

    fn rerank(
        &self,
        tokenizer: &Tokenizer,
        query: &str,
        documents: &[&str],
    ) -> anyhow::Result<Vec<RerankResult>> {
        self.rerank_documents(tokenizer, query, documents)
    }

    fn get_tokenizer(_options: Self::Options) -> anyhow::Result<Tokenizer> {
        let loader = TokenizerLoader::new("Qwen/Qwen3-0.6B", "tokenizer.json");
        futures::executor::block_on(loader.load())
    }

    fn batch_rerank(
        &self,
        tokenizer: &Tokenizer,
        queries: &[&str],
        documents: &[&str],
    ) -> anyhow::Result<Vec<Vec<RerankResult>>> {
        Qwen3RerankModel::batch_rerank(self, tokenizer, queries, documents)
    }

    fn device(&self) -> &Device {
        &self.device
    }
}
