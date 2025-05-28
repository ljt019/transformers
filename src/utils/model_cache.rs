use crate::pipelines::text_generation_pipeline::{Gemma3Size, ModelOptions, Phi4Size, Qwen3Size};
use candle_core::Device;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::sync::{Arc, Mutex, Weak};

/// A unique identifier for a cached model based on its configuration
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ModelCacheKey {
    model_family: String,
    model_size: String,
    device_type: String,
    device_id: Option<usize>,
}

impl ModelCacheKey {
    pub fn new(model_options: &ModelOptions, device: &Device) -> Self {
        let (model_family, model_size) = match model_options {
            ModelOptions::Qwen3(size) => {
                let size_str = match size {
                    Qwen3Size::Size0_6B => "0_6B",
                    Qwen3Size::Size1_7B => "1_7B",
                    Qwen3Size::Size4B => "4B",
                    Qwen3Size::Size8B => "8B",
                    Qwen3Size::Size14B => "14B",
                    Qwen3Size::Size32B => "32B",
                };
                ("Qwen3".to_string(), size_str.to_string())
            }
            ModelOptions::Gemma3(size) => {
                let size_str = match size {
                    Gemma3Size::Size1B => "1B",
                    Gemma3Size::Size4B => "4B",
                    Gemma3Size::Size12B => "12B",
                    Gemma3Size::Size27B => "27B",
                };
                ("Gemma3".to_string(), size_str.to_string())
            }
            ModelOptions::Phi4(size) => {
                let size_str = match size {
                    Phi4Size::Size14B => "14B",
                };
                ("Phi4".to_string(), size_str.to_string())
            }
        };

        let (device_type, device_id) = match device {
            Device::Cpu => ("CPU".to_string(), None),
            Device::Cuda(_cuda_device) => ("CUDA".to_string(), Some(0)), // Simplified for now
            Device::Metal(_metal_device) => ("Metal".to_string(), Some(0)), // Simplified for now
        };

        Self {
            model_family,
            model_size,
            device_type,
            device_id,
        }
    }
}

/// Container for shared model data that can be shared across multiple pipelines
pub trait SharedModelData: Send + Sync {}

/// Global cache for model instances
pub struct ModelCache {
    cache: Mutex<HashMap<ModelCacheKey, Box<dyn std::any::Any + Send + Sync>>>,
}

impl ModelCache {
    fn new() -> Self {
        Self {
            cache: Mutex::new(HashMap::new()),
        }
    }

    /// Get or load a model from the cache
    pub fn get_or_load<T, F>(&self, key: ModelCacheKey, loader: F) -> anyhow::Result<Arc<T>>
    where
        T: Send + Sync + 'static,
        F: FnOnce() -> anyhow::Result<T>,
    {
        let mut cache = self.cache.lock().unwrap();

        // Check if we have a cached weak reference
        if let Some(any_weak) = cache.get(&key) {
            // Try to downcast to Weak<T>
            if let Some(weak_ref) = any_weak.downcast_ref::<Weak<T>>() {
                // Try to upgrade the weak reference
                if let Some(strong_ref) = weak_ref.upgrade() {
                    // Model is still alive, return it
                    return Ok(strong_ref);
                }
            }
            // Either wrong type or dead reference, remove it
            cache.remove(&key);
        }

        // Cache miss or dead reference, load new model
        drop(cache); // Release lock while loading
        let model_data = loader()?;
        let arc_data = Arc::new(model_data);

        // Store weak reference in cache
        let weak_ref = Arc::downgrade(&arc_data);
        let mut cache = self.cache.lock().unwrap();
        cache.insert(key, Box::new(weak_ref));

        Ok(arc_data)
    }

    /// Clear all cached models (useful for testing or memory management)
    pub fn clear(&self) {
        let mut cache = self.cache.lock().unwrap();
        cache.clear();
    }

    /// Remove dead weak references from the cache
    pub fn cleanup_dead_references(&self) {
        let mut cache = self.cache.lock().unwrap();
        cache.retain(|_, weak_any| {
            // This is a simplified check - in a real implementation you'd need
            // to check each type specifically. For now, we'll rely on natural cleanup
            // when get_or_load is called.
            true
        });
    }
}

// Global singleton instance
static GLOBAL_CACHE: std::sync::OnceLock<ModelCache> = std::sync::OnceLock::new();

/// Get the global model cache instance
pub fn get_global_cache() -> &'static ModelCache {
    GLOBAL_CACHE.get_or_init(|| ModelCache::new())
}
