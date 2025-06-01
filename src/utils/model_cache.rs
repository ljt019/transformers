//! Model cache key for identifying cached models across different configurations.

use anyhow::Result;
use candle_core::Device;
use serde::{Deserialize, Serialize};
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};

/// Configuration key for caching models based on model path and device
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ModelCacheKey {
    pub model_path: String,
    pub device_type: String,
    pub device_id: Option<usize>,
    pub model_hash: u64,
}

impl ModelCacheKey {
    /// Create a new cache key from model path and device
    pub fn new(model_path: &str, device: &Device) -> Result<Self> {
        let mut hasher = DefaultHasher::new();
        model_path.hash(&mut hasher);
        let model_hash = hasher.finish();

        let (device_type, device_id) = match device {
            Device::Cpu => ("CPU".to_string(), None),
            Device::Cuda(_cuda_device) => {
                // TODO: Fix device indexing for multi-GPU setups
                // Currently hard-coded to 0 because candle_core::Device doesn't expose
                // the actual device index via a public API. This can cause incorrect
                // cache key mapping in multi-GPU setups where different devices should
                // have separate cache entries.
                ("CUDA".to_string(), Some(0))
            }
            Device::Metal(_metal_device) => {
                // TODO: Fix device indexing for multi-MPS setups
                // Currently hard-coded to 0 because candle_core::Device doesn't expose
                // the actual device index via a public API. This can cause incorrect
                // cache key mapping in multi-MPS setups where different devices should
                // have separate cache entries.
                ("Metal".to_string(), Some(0))
            }
        };

        Ok(Self {
            model_path: model_path.to_string(),
            device_type,
            device_id,
            model_hash,
        })
    }
}
