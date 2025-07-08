use anyhow::Result;
use transformers::models::implementations::Qwen3RerankSize;
use transformers::pipelines::reranker_pipeline::*;
use transformers::pipelines::utils::DeviceSelectable;

#[tokio::main]
async fn main() -> Result<()> {
    println!("🕵️ Investigating Qwen3 Reranker Scoring Bug");
    println!("Testing current implementation behavior");
    println!();
    
    // Build the reranker pipeline
    let rerank_pipe = RerankPipelineBuilder::qwen3(Qwen3RerankSize::Size0_6B)
        .cpu()
        .build()
        .await?;
    
    // Create test cases with clear relevance expectations
    let query = "What is machine learning and how does it work?";
    let documents = vec![
        "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed. It works by using algorithms to analyze data patterns.",
        "Deep learning is a type of machine learning that uses neural networks with multiple layers to process and learn from data, mimicking how the human brain works.",
        "Supervised learning is a machine learning approach where algorithms learn from labeled training data to make predictions on new, unseen data.",
        "Cooking pasta requires boiling water, adding salt, and timing the cooking process carefully to achieve the right texture and flavor.",
        "The weather forecast shows sunny skies with a high of 75°F and low humidity throughout the day.",
        "Gardening tips include watering plants regularly, providing adequate sunlight, and using proper soil nutrients for healthy growth.",
    ];
    
    println!("Query: {}", query);
    println!();
    
    println!("📊 Documents with expected relevance:");
    for (i, doc) in documents.iter().enumerate() {
        let is_relevant = doc.to_lowercase().contains("machine learning") || 
                         doc.to_lowercase().contains("neural network") ||
                         doc.to_lowercase().contains("deep learning") ||
                         doc.to_lowercase().contains("supervised learning");
        
        println!("  {}. [{}] {}", 
                 i + 1, 
                 if is_relevant { "RELEVANT" } else { "IRRELEVANT" },
                 truncate(doc, 60));
    }
    
    println!();
    println!("🔍 Current implementation ranking:");
    
    // Get ranking from current implementation
    let results = rerank_pipe.rerank(query, &documents).await?;
    
    for (rank, result) in results.iter().enumerate() {
        let doc = &documents[result.index];
        let is_relevant = doc.to_lowercase().contains("machine learning") || 
                         doc.to_lowercase().contains("neural network") ||
                         doc.to_lowercase().contains("deep learning") ||
                         doc.to_lowercase().contains("supervised learning");
        
        println!("  {}. [score: {:.4}] [{}] {}", 
                 rank + 1, 
                 result.score,
                 if is_relevant { "RELEVANT" } else { "IRRELEVANT" },
                 truncate(doc, 60));
    }
    
    println!();
    println!("🎯 ANALYSIS:");
    
    // Check if relevant documents are ranked higher
    let relevant_docs_in_top_3 = results.iter().take(3).filter(|r| {
        let doc = &documents[r.index];
        doc.to_lowercase().contains("machine learning") || 
        doc.to_lowercase().contains("neural network") ||
        doc.to_lowercase().contains("deep learning") ||
        doc.to_lowercase().contains("supervised learning")
    }).count();
    
    let irrelevant_docs_in_top_3 = 3 - relevant_docs_in_top_3;
    
    println!("  Relevant documents in top 3: {}/3", relevant_docs_in_top_3);
    println!("  Irrelevant documents in top 3: {}/3", irrelevant_docs_in_top_3);
    
    if relevant_docs_in_top_3 >= 2 {
        println!("  ✅ Current implementation appears to work correctly");
        println!("  ✅ Relevant documents are ranked higher");
        println!("  📝 Using 'no' probability as relevance score is correct for GGUF models");
    } else {
        println!("  ❌ Current implementation appears to be broken");
        println!("  ❌ Irrelevant documents are ranked higher than relevant ones");
        println!("  🔧 BUG: Should probably use 'yes' probability instead of 'no' probability");
    }
    
    println!();
    println!("🔬 DETAILED SCORE ANALYSIS:");
    
    // Calculate average scores for relevant vs irrelevant docs
    let mut relevant_scores = Vec::new();
    let mut irrelevant_scores = Vec::new();
    
    for result in &results {
        let doc = &documents[result.index];
        let is_relevant = doc.to_lowercase().contains("machine learning") || 
                         doc.to_lowercase().contains("neural network") ||
                         doc.to_lowercase().contains("deep learning") ||
                         doc.to_lowercase().contains("supervised learning");
        
        if is_relevant {
            relevant_scores.push(result.score);
        } else {
            irrelevant_scores.push(result.score);
        }
    }
    
    let avg_relevant = relevant_scores.iter().sum::<f32>() / relevant_scores.len() as f32;
    let avg_irrelevant = irrelevant_scores.iter().sum::<f32>() / irrelevant_scores.len() as f32;
    
    println!("  Average score for relevant documents: {:.4}", avg_relevant);
    println!("  Average score for irrelevant documents: {:.4}", avg_irrelevant);
    
    if avg_relevant > avg_irrelevant {
        println!("  ✅ Relevant documents have higher average scores");
    } else {
        println!("  ❌ Irrelevant documents have higher average scores - this indicates a bug");
    }
    
    Ok(())
}

fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len])
    }
}