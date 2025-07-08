use candle_core::quantized::gguf_file;
use std::fs::File;

fn main() -> anyhow::Result<()> {
    // Path to GGUF file
    let path = std::env::home_dir().unwrap()
        .join(".cache/huggingface/hub/models--Mungert--Qwen3-Reranker-0.6B-GGUF/blobs/66867f47323e058f9dbfe24a13268859a84d9e9a8bb89ad0789c7c52131267e2");
    
    println!("Opening GGUF file: {}", path.display());
    let mut file = File::open(&path)?;
    
    // Read GGUF content metadata only
    let start = std::time::Instant::now();
    let content = gguf_file::Content::read(&mut file)?;
    println!("Read GGUF metadata in: {:?}", start.elapsed());
    
    // Print metadata
    if let Some(layers) = content.metadata.get("qwen3.block_count") {
        println!("Number of layers: {:?}", layers);
    }
    
    // Count tensors
    println!("\nTotal tensors in file: {}", content.tensor_infos.len());
    
    // List first 20 tensor names
    println!("\nFirst 20 tensor names:");
    for (i, (name, _)) in content.tensor_infos.iter().enumerate() {
        if i >= 20 { break; }
        println!("  {}: {}", i, name);
    }
    
    // Count tensors by prefix
    let mut layer_tensors = 0;
    let mut other_tensors = 0;
    for (name, _) in &content.tensor_infos {
        if name.starts_with("blk.") {
            layer_tensors += 1;
        } else {
            other_tensors += 1;
        }
    }
    
    println!("\nTensor breakdown:");
    println!("  Layer tensors (blk.*): {}", layer_tensors);
    println!("  Other tensors: {}", other_tensors);
    
    Ok(())
}