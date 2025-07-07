use std::sync::Arc;

use candle_core::{DType, Device, Result, Tensor};
use tokenizers::Tokenizer;

use super::qwen3::ModelWeights;

#[derive(Debug, Clone, Copy)]
pub enum Qwen3EmbeddingSize {
    Size0_6B,
    Size4B,
    Size8B,
}

impl std::fmt::Display for Qwen3EmbeddingSize {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Qwen3EmbeddingSize::Size0_6B => write!(f, "0.6B"),
            Qwen3EmbeddingSize::Size4B => write!(f, "4B"),
            Qwen3EmbeddingSize::Size8B => write!(f, "8B"),
        }
    }
}

// Allow embedding size to be used as a cache key just like other model option enums.
impl crate::core::ModelOptions for Qwen3EmbeddingSize {
    fn cache_key(&self) -> String {
        self.to_string()
    }
}

use crate::loaders::{GgufModelLoader, TokenizerLoader};

fn embed_id(size: Qwen3EmbeddingSize) -> (String, String) {
    match size {
        Qwen3EmbeddingSize::Size0_6B => (
            "Qwen/Qwen3-Embedding-0.6B-GGUF".into(),
            "Qwen3-Embedding-0.6B-Q8_0.gguf".into(),
        ),
        Qwen3EmbeddingSize::Size4B => (
            "Qwen/Qwen3-Embedding-4B-GGUF".into(),
            "Qwen3-Embedding-4B-Q8_0.gguf".into(),
        ),
        Qwen3EmbeddingSize::Size8B => (
            "Qwen/Qwen3-Embedding-8B-GGUF".into(),
            "Qwen3-Embedding-8B-Q8_0.gguf".into(),
        ),
    }
}

/// Qwen3 model for generating text embeddings.
#[derive(Clone)]
pub struct Qwen3EmbeddingModel {
    weights: Arc<ModelWeights>,
    device: Device,
}

impl Qwen3EmbeddingModel {
    pub async fn from_hf(device: &Device, size: Qwen3EmbeddingSize) -> anyhow::Result<Self> {
        let (repo_id, file_name) = embed_id(size);
        let loader = GgufModelLoader::new(&repo_id, &file_name);
        let (mut file, content) = loader.load().await?;
        let weights = Arc::new(ModelWeights::from_gguf(content, &mut file, device)?);
        Ok(Self {
            weights,
            device: device.clone(),
        })
    }

    pub async fn get_tokenizer(&self) -> anyhow::Result<Tokenizer> {
        let loader = TokenizerLoader::new("Qwen/Qwen3-0.6B", "tokenizer.json");
        loader.load().await
    }

    /// Generate an embedding for the provided text.
    pub fn embed(&self, tokenizer: &Tokenizer, text: &str) -> anyhow::Result<Vec<f32>> {
        self.embed_with_instruction(tokenizer, None, text)
    }

    /// Generate an embedding with optional instruction concatenation.
    pub fn embed_with_instruction(
        &self,
        tokenizer: &Tokenizer,
        instruction: Option<&str>,
        text: &str,
    ) -> anyhow::Result<Vec<f32>> {
        const EOS: &str = "<|endoftext|>";
        let input_text = match instruction {
            Some(instr) => format!("{} {}", instr, text),
            None => text.to_string(),
        };
        let encoded = tokenizer
            .encode(format!("{input_text}{EOS}"), false)
            .map_err(anyhow::Error::msg)?;
        let ids = encoded.get_ids();
        if ids.is_empty() {
            return Err(anyhow::anyhow!(
                "Tokenizer produced empty token sequence for text: '{}'",
                input_text
            ));
        }
        let input = Tensor::new(ids, &self.device)?.unsqueeze(0)?;
        // Attention mask is not required for a single-sequence forward pass. Omitting it avoids
        // shape-broadcast issues on very short inputs.
        let emb = self.weights.forward_embedding(&input, None)?;
        let emb = l2_normalise(emb)?;
        // Squeeze the batch dimension if it exists
        let emb = if emb.dims().len() > 1 && emb.dims()[0] == 1 {
            emb.squeeze(0)?
        } else {
            emb
        };
        Ok(emb.to_vec1::<f32>()?)
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}

fn l2_normalise(t: Tensor) -> Result<Tensor> {
    let norm = t.sqr()?.sum_keepdim(candle_core::D::Minus1)?.sqrt()?;
    t.broadcast_div(&norm)
}

fn create_causal_mask(
    device: &Device,
    batch_size: usize,
    seq_len: usize,
    position_offset: usize,
) -> Result<Tensor> {
    let total_len = seq_len + position_offset;

    let row_ids = Tensor::arange(0u32, seq_len as u32, device)?
        .to_dtype(DType::I64)?
        .reshape((seq_len, 1))?
        .broadcast_add(&Tensor::new(&[position_offset as i64], device)?)?;

    let col_ids = Tensor::arange(0u32, total_len as u32, device)?
        .to_dtype(DType::I64)?
        .reshape((1, total_len))?;

    let mask = row_ids
        .broadcast_as(&[seq_len, total_len])?
        .ge(&col_ids.broadcast_as(&[seq_len, total_len])?)?;

    let neg_inf = Tensor::new(&[f32::NEG_INFINITY], device)?.broadcast_as(&[seq_len, total_len])?;
    let zero = Tensor::zeros(&[seq_len, total_len], DType::F32, device)?;

    let float_mask = mask.where_cond(&zero, &neg_inf)?;

    float_mask
        .unsqueeze(0)?
        .unsqueeze(0)?
        .broadcast_as(&[batch_size, 1, seq_len, total_len])
}

use crate::pipelines::embedding_pipeline::embedding_model::EmbeddingModel;

impl EmbeddingModel for Qwen3EmbeddingModel {
    type Options = Qwen3EmbeddingSize;

    fn new(options: Self::Options, device: Device) -> anyhow::Result<Self> {
        futures::executor::block_on(Self::from_hf(&device, options))
    }

    fn embed(&self, tokenizer: &Tokenizer, text: &str) -> anyhow::Result<Vec<f32>> {
        Qwen3EmbeddingModel::embed(self, tokenizer, text)
    }

    fn get_tokenizer(_options: Self::Options) -> anyhow::Result<Tokenizer> {
        let loader = TokenizerLoader::new("Qwen/Qwen3-0.6B", "tokenizer.json");
        futures::executor::block_on(loader.load())
    }

    fn device(&self) -> &Device {
        &self.device
    }
}
