# Reranker Performance Fix Summary

## Problem Identified

Your reranker implementation was running ~2.5x slower than the reference implementation (46.7s vs 18.9s) due to unnecessary overhead from KV caching and causal masking.

## Root Causes

1. **KV Cache Reset on Every Forward Pass**
   - The model was resetting the KV cache every time (`offset == 0`)
   - Since `forward()` was always called with `offset = 0`, the cache was cleared for every document
   - This added overhead without any caching benefit

2. **Unnecessary Causal Masking**
   - The model created causal attention masks for each forward pass
   - Reranking processes the full sequence at once and doesn't need causal masking
   - Creating these masks added computational overhead

3. **KV Cache Memory Operations**
   - Even without reuse, the KV cache operations (`append`, `contiguous()`) added overhead
   - For single-pass inference (like reranking), KV caching is unnecessary

## Changes Made

### 1. Removed KV Cache Usage in Attention
In `AttentionWeights::forward()`:
```rust
// Before:
if offset == 0 {
    self.kv_cache.reset();
}
let (k, v) = self.kv_cache.append(&k.contiguous()?, &v.contiguous()?)?;

// After:
// For reranking, we don't need KV cache as each document is processed independently
// Directly use k and v without caching
let k = k.contiguous()?;
let v = v.contiguous()?;
```

### 2. Removed Causal Masking
In `ModelWeights::forward()`:
```rust
// Before:
let causal_mask = if l == 1 {
    None
} else {
    Some(self.causal_mask(b, l, offset, None)?)
};

// After:
// For reranking, we don't need causal masking as we process the full sequence
let causal_mask = None;
```

### 3. Added Configuration Flag (Optional)
Added `use_kv_cache` flag to allow reverting to old behavior for comparison:
```rust
pub struct Qwen3RerankModel {
    // ...
    use_kv_cache: bool, // Default: false
}
```

## Expected Performance Improvement

Based on the changes:
- **Before**: ~46.7s for 4 documents
- **After**: Expected ~20-25s (similar to reference implementation)
- **Speedup**: ~2x faster

The optimized implementation should now perform similarly to the reference implementation since both:
- Process documents in a single forward pass
- Don't use KV caching
- Don't apply causal masking

## How to Test

Run your reranker example again:
```bash
cargo run --release --example reranker
```

You should see significantly improved performance, with the total time reduced by approximately 50%.

## Why This Works

1. **Reranking is Not Autoregressive**: Unlike text generation, reranking processes the full query-document pair in one pass
2. **No Sequential Dependencies**: Each document is scored independently, so caching previous computations doesn't help
3. **Simplified Attention**: Without causal masking, attention computation is more efficient

These optimizations align your implementation with the reference code's approach, which is optimal for the reranking use case.