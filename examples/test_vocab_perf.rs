use std::time::Instant;
use transformers::loaders::TokenizerLoader;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let loader = TokenizerLoader::new("Qwen/Qwen3-0.6B", "tokenizer.json");
    let tokenizer = loader.load().await?;
    
    // Time get_vocab
    let start = Instant::now();
    let vocab = tokenizer.get_vocab(true);
    println!("First get_vocab: {:?}", start.elapsed());
    println!("Vocab size: {}", vocab.len());
    
    // Time subsequent calls
    let start = Instant::now();
    for _ in 0..1000 {
        let _ = tokenizer.get_vocab(true);
    }
    println!("1000 get_vocab calls: {:?}", start.elapsed());
    
    // Time token lookups
    let vocab = tokenizer.get_vocab(true);
    let start = Instant::now();
    for _ in 0..1000 {
        let _ = vocab.get("yes");
        let _ = vocab.get("no");
    }
    println!("1000 token lookups: {:?}", start.elapsed());
    
    Ok(())
}