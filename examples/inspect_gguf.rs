use anyhow::Result;
use std::fs::File;
use candle_core::quantized::gguf_file;

#[tokio::main]
async fn main() -> Result<()> {
    println!("🔍 ANALYZING GGUF MODEL STRUCTURE");
    println!("Model: Qwen3-Reranker-0.6B-q4_k_m.gguf");
    println!();
    
    // Load the GGUF file directly to examine its structure
    let file_path = "Qwen3-Reranker-0.6B-q4_k_m.gguf";
    
    println!("📂 Opening GGUF file...");
    let mut file = File::open(file_path)?;
    
    // Parse GGUF content
    let content = gguf_file::Content::read(&mut file)?;
    
    println!("✅ GGUF file loaded successfully");
    println!();
    
    // Examine metadata
    println!("📋 METADATA:");
    println!("  Total metadata entries: {}", content.metadata.len());
    
    // Key metadata to look for
    let key_metadata = [
        "general.architecture",
        "general.name", 
        "general.dtype",
        "qwen3.attention.head_count",
        "qwen3.embedding_length",
        "qwen3.block_count",
        "qwen3.context_length",
        "tokenizer.ggml.model",
    ];
    
    for key in &key_metadata {
        if let Some(value) = content.metadata.get(*key) {
            println!("  {}: {:?}", key, value);
        }
    }
    
    println!();
    
    // Examine tensor information
    println!("🔢 TENSORS:");
    println!("  Total tensors: {}", content.tensor_infos.len());
    println!();
    
    // Look for specific tensor patterns that indicate model type
    let mut has_lm_head = false;
    let mut has_classifier = false;
    let mut output_layer_info = None;
    
    println!("📊 TENSOR ANALYSIS:");
    
    // Group tensors by category
    let mut embedding_tensors = Vec::new();
    let mut layer_tensors = Vec::new();
    let mut output_tensors = Vec::new();
    let mut norm_tensors = Vec::new();
    let mut other_tensors = Vec::new();
    
    for (name, info) in &content.tensor_infos {
        let shape_str = format!("{:?}", info.shape);
        
        if name.contains("embed") {
            embedding_tensors.push((name.clone(), shape_str));
        } else if name.contains("blk.") {
            layer_tensors.push((name.clone(), shape_str));
        } else if name.contains("output") || name.contains("lm_head") || name.contains("classifier") {
            output_tensors.push((name.clone(), shape_str));
            
            if name.contains("lm_head") || (name.contains("output") && !name.contains("norm")) {
                has_lm_head = true;
                output_layer_info = Some((name.clone(), info.shape.clone()));
            }
            if name.contains("classifier") {
                has_classifier = true;
                output_layer_info = Some((name.clone(), info.shape.clone()));
            }
        } else if name.contains("norm") {
            norm_tensors.push((name.clone(), shape_str));
        } else {
            other_tensors.push((name.clone(), shape_str));
        }
    }
    
    println!("  📝 Embedding tensors: {}", embedding_tensors.len());
    for (name, shape) in &embedding_tensors {
        println!("    {}: {}", name, shape);
    }
    
    println!("  🔄 Layer tensors: {}", layer_tensors.len());
    if layer_tensors.len() > 10 {
        println!("    (showing first 5 and last 5)");
        for (name, shape) in layer_tensors.iter().take(5) {
            println!("    {}: {}", name, shape);
        }
        println!("    ...");
        for (name, shape) in layer_tensors.iter().rev().take(5).rev() {
            println!("    {}: {}", name, shape);
        }
    } else {
        for (name, shape) in &layer_tensors {
            println!("    {}: {}", name, shape);
        }
    }
    
    println!("  📤 Output tensors: {}", output_tensors.len());
    for (name, shape) in &output_tensors {
        println!("    {}: {}", name, shape);
    }
    
    println!("  📏 Normalization tensors: {}", norm_tensors.len());
    for (name, shape) in &norm_tensors {
        println!("    {}: {}", name, shape);
    }
    
    if !other_tensors.is_empty() {
        println!("  🔧 Other tensors: {}", other_tensors.len());
        for (name, shape) in &other_tensors {
            println!("    {}: {}", name, shape);
        }
    }
    
    println!();
    
    // Analysis conclusion
    println!("🎯 ARCHITECTURE ANALYSIS:");
    
    if let Some((output_name, output_shape)) = output_layer_info {
        println!("  Primary output layer: {}", output_name);
        println!("  Output shape: {:?}", output_shape);
        
        // Analyze output shape to determine model type  
        let dims = output_shape.dims();
        if dims.len() == 2 {
            let (dim1, dim2) = (dims[0], dims[1]);
            
            if dim2 > 100000 {
                println!("  🔍 Analysis: Large vocabulary size ({}) indicates LANGUAGE MODELING HEAD", dim2);
                println!("  ❌ This is NOT a proper reranker classification head");
                println!("  ⚠️  The model is using vocabulary logits for classification");
            } else if dim2 < 10 {
                println!("  🔍 Analysis: Small output size ({}) indicates CLASSIFICATION HEAD", dim2);
                println!("  ✅ This appears to be a proper reranker model");
            } else {
                println!("  🔍 Analysis: Medium output size ({}) - unclear model type", dim2);
            }
        }
    } else {
        println!("  ❓ Could not identify primary output layer");
    }
    
    println!();
    println!("🏁 CONCLUSION:");
    if has_classifier {
        println!("  ✅ Model has classification head - proper reranker architecture");
    } else if has_lm_head {
        println!("  ❌ Model has language modeling head - NOT a proper reranker");
        println!("  🔧 Current implementation is treating LM head as reranker (incorrect)");
        println!("  💡 This explains the 'yes'/'no' token extraction approach");
    } else {
        println!("  ❓ Unclear model architecture from tensor names");
    }
    
    Ok(())
}