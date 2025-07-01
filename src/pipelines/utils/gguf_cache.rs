use crate::loaders::GgufModelLoader;
use candle_core::quantized::gguf_file;
use candle_core::Device;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, Weak};

/// Cache key for GGUF files based on repository and filename
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct GgufCacheKey {
    repo: String,
    filename: String,
}

/// Cached GGUF data that can be shared across model instances
#[derive(Debug)]
pub struct CachedGgufData {
    pub file_data: Vec<u8>,
}

/// Global cache for GGUF file contents
pub struct GgufCache {
    cache: Mutex<HashMap<GgufCacheKey, Weak<CachedGgufData>>>,
}

impl GgufCache {
    fn new() -> Self {
        Self {
            cache: Mutex::new(HashMap::new()),
        }
    }

    /// Get or load GGUF data from cache
    pub fn get_or_load(&self, repo: &str, filename: &str) -> anyhow::Result<Arc<CachedGgufData>> {
        let key = GgufCacheKey {
            repo: repo.to_string(),
            filename: filename.to_string(),
        };

        // First check: try to get from cache
        {
            let mut cache = self.cache.lock().unwrap();
            if let Some(weak_ref) = cache.get(&key) {
                if let Some(strong_ref) = weak_ref.upgrade() {
                    // GGUF data is still alive, return it
                    return Ok(strong_ref);
                }
                // Dead reference, remove it
                cache.remove(&key);
            }
        }

        // Cache miss or dead reference, load new GGUF data outside the lock
        let gguf_loader = GgufModelLoader::new(repo, filename);
        let (mut gguf_file, _) = gguf_loader.load()?;

        // Read the entire file into memory for caching
        use std::io::{Read, Seek, SeekFrom};
        gguf_file.seek(SeekFrom::Start(0))?;
        let mut file_data = Vec::new();
        gguf_file.read_to_end(&mut file_data)?;

        let cached_data = Arc::new(CachedGgufData { file_data });

        // Second check: re-acquire lock and check again before inserting
        {
            let mut cache = self.cache.lock().unwrap();
            // Check if another thread loaded the GGUF data while we were loading
            if let Some(weak_ref) = cache.get(&key) {
                if let Some(strong_ref) = weak_ref.upgrade() {
                    // Another thread loaded it, use their version
                    return Ok(strong_ref);
                }
                // Dead reference, remove it
                cache.remove(&key);
            }

            // Store our loaded GGUF data in cache
            let weak_ref = Arc::downgrade(&cached_data);
            cache.insert(key, weak_ref);
        }

        Ok(cached_data)
    }

    /// Clear all cached GGUF data
    pub fn clear(&self) {
        let mut cache = self.cache.lock().unwrap();
        cache.clear();
    }
}

// Global singleton instance
static GLOBAL_GGUF_CACHE: std::sync::OnceLock<GgufCache> = std::sync::OnceLock::new();

/// Get the global GGUF cache instance
pub fn get_global_gguf_cache() -> &'static GgufCache {
    GLOBAL_GGUF_CACHE.get_or_init(|| GgufCache::new())
}

/// Helper function to create model weights from cached GGUF data
pub fn create_model_weights_from_cache<T, F>(
    repo: &str,
    filename: &str,
    device: &Device,
    weights_builder: F,
) -> anyhow::Result<T>
where
    F: FnOnce(gguf_file::Content, &mut std::io::Cursor<&[u8]>, &Device) -> anyhow::Result<T>,
{
    let cache = get_global_gguf_cache();
    let cached_data = cache.get_or_load(repo, filename)?;

    // Create a cursor from the cached file data and re-parse the content
    let mut cursor = std::io::Cursor::new(cached_data.file_data.as_slice());
    let content = gguf_file::Content::read(&mut cursor)?;

    // Reset cursor position for the weights builder
    cursor.set_position(0);

    // Build the model weights using the re-parsed content
    weights_builder(content, &mut cursor, device)
}
