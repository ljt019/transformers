use crate::models::raw::models::{quantized_gemma3, quantized_phi3, quantized_qwen3};
use crate::utils::model_cache::ModelCacheKey;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, Weak};

/// Global cache for shared model instances
pub struct SharedModelCache {
    qwen3_weights: Mutex<HashMap<ModelCacheKey, Weak<quantized_qwen3::Weights>>>,
    gemma3_weights: Mutex<HashMap<ModelCacheKey, Weak<quantized_gemma3::Weights>>>,
    phi4_weights: Mutex<HashMap<ModelCacheKey, Weak<quantized_phi3::Weights>>>,
}

impl SharedModelCache {
    fn new() -> Self {
        Self {
            qwen3_weights: Mutex::new(HashMap::new()),
            gemma3_weights: Mutex::new(HashMap::new()),
            phi4_weights: Mutex::new(HashMap::new()),
        }
    }

    /// Get or load Qwen3 weights from the cache
    pub fn get_or_load_qwen3_weights<F>(
        &self,
        key: ModelCacheKey,
        loader: F,
    ) -> anyhow::Result<Arc<quantized_qwen3::Weights>>
    where
        F: FnOnce() -> anyhow::Result<quantized_qwen3::Weights>,
    {
        // First check: try to get from cache
        {
            let mut cache = self.qwen3_weights.lock().unwrap();
            if let Some(weak_ref) = cache.get(&key) {
                if let Some(strong_ref) = weak_ref.upgrade() {
                    // Model is still alive, return it
                    println!("Using cached Qwen3 weights - shared across pipelines!");
                    return Ok(strong_ref);
                }
                // Dead reference, remove it
                cache.remove(&key);
            }
        }

        // Cache miss or dead reference, load new model outside the lock
        println!("Loading new Qwen3 weights into cache...");
        let weights = loader()?;
        let arc_weights = Arc::new(weights);

        // Second check: re-acquire lock and check again before inserting
        {
            let mut cache = self.qwen3_weights.lock().unwrap();
            // Check if another thread loaded the model while we were loading
            if let Some(weak_ref) = cache.get(&key) {
                if let Some(strong_ref) = weak_ref.upgrade() {
                    // Another thread loaded it, use their version
                    println!("Using Qwen3 weights loaded by another thread!");
                    return Ok(strong_ref);
                }
                // Dead reference, remove it
                cache.remove(&key);
            }

            // Store our loaded model in cache
            let weak_ref = Arc::downgrade(&arc_weights);
            cache.insert(key, weak_ref);
        }

        Ok(arc_weights)
    }

    /// Get or load Gemma3 weights from the cache
    pub fn get_or_load_gemma3_weights<F>(
        &self,
        key: ModelCacheKey,
        loader: F,
    ) -> anyhow::Result<Arc<quantized_gemma3::Weights>>
    where
        F: FnOnce() -> anyhow::Result<quantized_gemma3::Weights>,
    {
        // First check: try to get from cache
        {
            let mut cache = self.gemma3_weights.lock().unwrap();
            if let Some(weak_ref) = cache.get(&key) {
                if let Some(strong_ref) = weak_ref.upgrade() {
                    // Model is still alive, return it
                    println!("Using cached Gemma3 weights - shared across pipelines!");
                    return Ok(strong_ref);
                }
                // Dead reference, remove it
                cache.remove(&key);
            }
        }

        // Cache miss or dead reference, load new model outside the lock
        println!("Loading new Gemma3 weights into cache...");
        let weights = loader()?;
        let arc_weights = Arc::new(weights);

        // Second check: re-acquire lock and check again before inserting
        {
            let mut cache = self.gemma3_weights.lock().unwrap();
            // Check if another thread loaded the model while we were loading
            if let Some(weak_ref) = cache.get(&key) {
                if let Some(strong_ref) = weak_ref.upgrade() {
                    // Another thread loaded it, use their version
                    println!("Using Gemma3 weights loaded by another thread!");
                    return Ok(strong_ref);
                }
                // Dead reference, remove it
                cache.remove(&key);
            }

            // Store our loaded model in cache
            let weak_ref = Arc::downgrade(&arc_weights);
            cache.insert(key, weak_ref);
        }

        Ok(arc_weights)
    }

    /// Get or load Phi4 weights from the cache
    pub fn get_or_load_phi4_weights<F>(
        &self,
        key: ModelCacheKey,
        loader: F,
    ) -> anyhow::Result<Arc<quantized_phi3::Weights>>
    where
        F: FnOnce() -> anyhow::Result<quantized_phi3::Weights>,
    {
        // First check: try to get from cache
        {
            let mut cache = self.phi4_weights.lock().unwrap();
            if let Some(weak_ref) = cache.get(&key) {
                if let Some(strong_ref) = weak_ref.upgrade() {
                    // Model is still alive, return it
                    println!("Using cached Phi4 weights - shared across pipelines!");
                    return Ok(strong_ref);
                }
                // Dead reference, remove it
                cache.remove(&key);
            }
        }

        // Cache miss or dead reference, load new model outside the lock
        println!("Loading new Phi4 weights into cache...");
        let weights = loader()?;
        let arc_weights = Arc::new(weights);

        // Second check: re-acquire lock and check again before inserting
        {
            let mut cache = self.phi4_weights.lock().unwrap();
            // Check if another thread loaded the model while we were loading
            if let Some(weak_ref) = cache.get(&key) {
                if let Some(strong_ref) = weak_ref.upgrade() {
                    // Another thread loaded it, use their version
                    println!("Using Phi4 weights loaded by another thread!");
                    return Ok(strong_ref);
                }
                // Dead reference, remove it
                cache.remove(&key);
            }

            // Store our loaded model in cache
            let weak_ref = Arc::downgrade(&arc_weights);
            cache.insert(key, weak_ref);
        }

        Ok(arc_weights)
    }
}

// Global singleton
static GLOBAL_SHARED_MODEL_CACHE: std::sync::OnceLock<SharedModelCache> =
    std::sync::OnceLock::new();

pub fn get_global_shared_model_cache() -> &'static SharedModelCache {
    GLOBAL_SHARED_MODEL_CACHE.get_or_init(|| SharedModelCache::new())
}
