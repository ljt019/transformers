use anyhow::Result;
use candle_core::{Device, Tensor};
use std::sync::Arc;
use transformers::models::implementations::qwen3::ModelWeights;
use transformers::models::implementations::Qwen3RerankSize;
use transformers::loaders::{GgufModelLoader, TokenizerLoader};
use tokenizers::Tokenizer;

/// Test both scoring approaches to identify which is correct
pub struct ScoringInvestigator {
    weights: Arc<ModelWeights>,
    device: Device,
}

impl ScoringInvestigator {
    pub async fn from_hf(device: &Device, size: Qwen3RerankSize) -> anyhow::Result<Self> {
        let (repo_id, file_name) = match size {
            Qwen3RerankSize::Size0_6B => (
                "DevQuasar/Qwen.Qwen3-Reranker-0.6B-GGUF".to_string(),
                "Qwen.Qwen3-Reranker-0.6B.Q4_K_M.gguf".to_string(),
            ),
            _ => return Err(anyhow::anyhow!("Using 0.6B for testing")),
        };
        
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

    /// Test both scoring approaches: YES prob vs NO prob
    pub fn compare_scoring_approaches(
        &self,
        tokenizer: &Tokenizer,
        query: &str,
        documents: &[&str],
    ) -> anyhow::Result<ScoringComparison> {
        let mut results = Vec::new();
        
        for (idx, doc) in documents.iter().enumerate() {
            let probs = self.get_probabilities(tokenizer, query, doc)?;
            results.push(DocumentScore {
                index: idx,
                document: doc.to_string(),
                yes_prob: probs.yes_prob,
                no_prob: probs.no_prob,
                yes_logit: probs.yes_logit,
                no_logit: probs.no_logit,
            });
        }
        
        // Sort by YES probability (official approach)
        let mut yes_ranked = results.clone();
        yes_ranked.sort_by(|a, b| b.yes_prob.partial_cmp(&a.yes_prob).unwrap());
        
        // Sort by NO probability (current implementation)
        let mut no_ranked = results.clone();
        no_ranked.sort_by(|a, b| b.no_prob.partial_cmp(&a.no_prob).unwrap());
        
        Ok(ScoringComparison {
            query: query.to_string(),
            yes_prob_ranking: yes_ranked,
            no_prob_ranking: no_ranked,
        })
    }
    
    fn get_probabilities(
        &self,
        tokenizer: &Tokenizer,
        query: &str,
        document: &str,
    ) -> anyhow::Result<ProbabilityResult> {
        let instruction = "Given a query and a document, determine whether the document is relevant to the query";
        let input_text = format!(
            "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {document}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        );
        
        let encoded = tokenizer.encode(input_text, false).map_err(anyhow::Error::msg)?;
        let ids = encoded.get_ids();
        let input = Tensor::new(ids, &self.device)?.unsqueeze(0)?;
        
        let logits = self.weights.forward_logits(&input, None)?;
        let (_, seq_len, _vocab_size) = logits.dims3()?;
        let last_token_logits = logits.narrow(1, seq_len - 1, 1)?.squeeze(1)?;
        
        let yes_token_id = tokenizer.encode("yes", false).map_err(anyhow::Error::msg)?.get_ids()[0];
        let no_token_id = tokenizer.encode("no", false).map_err(anyhow::Error::msg)?.get_ids()[0];
        
        let yes_logit = last_token_logits.narrow(1, yes_token_id as usize, 1)?.squeeze(1)?.squeeze(0)?.to_scalar::<f32>()?;
        let no_logit = last_token_logits.narrow(1, no_token_id as usize, 1)?.squeeze(1)?.squeeze(0)?.to_scalar::<f32>()?;
        
        let exp_yes = yes_logit.exp();
        let exp_no = no_logit.exp();
        let total = exp_yes + exp_no;
        
        Ok(ProbabilityResult {
            yes_prob: exp_yes / total,
            no_prob: exp_no / total,
            yes_logit,
            no_logit,
        })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct DocumentScore {
    pub index: usize,
    pub document: String,
    pub yes_prob: f32,
    pub no_prob: f32,
    pub yes_logit: f32,
    pub no_logit: f32,
}

#[derive(Debug)]
pub struct ScoringComparison {
    pub query: String,
    pub yes_prob_ranking: Vec<DocumentScore>,
    pub no_prob_ranking: Vec<DocumentScore>,
}

#[derive(Debug)]
pub struct ProbabilityResult {
    pub yes_prob: f32,
    pub no_prob: f32,
    pub yes_logit: f32,
    pub no_logit: f32,
}

impl ScoringComparison {
    pub fn analyze_correctness(&self) -> ScoringAnalysis {
        println!("🔍 SCORING BEHAVIOR ANALYSIS");
        println!("Query: {}", self.query);
        println!();
        
        println!("📊 YES Probability Ranking (Official Method):");
        for (rank, doc) in self.yes_prob_ranking.iter().enumerate() {
            println!("  {}. [YES: {:.4}, NO: {:.4}] {}", 
                     rank + 1, doc.yes_prob, doc.no_prob, 
                     truncate(&doc.document, 60));
        }
        
        println!();
        println!("📊 NO Probability Ranking (Current Implementation):");
        for (rank, doc) in self.no_prob_ranking.iter().enumerate() {
            println!("  {}. [YES: {:.4}, NO: {:.4}] {}", 
                     rank + 1, doc.yes_prob, doc.no_prob, 
                     truncate(&doc.document, 60));
        }
        
        println!();
        
        // Check which method ranks relevant documents higher
        let mut yes_method_score = 0;
        let mut no_method_score = 0;
        
        // Find ML-related documents (should rank high)
        for (rank, doc) in self.yes_prob_ranking.iter().enumerate() {
            if doc.document.to_lowercase().contains("machine learning") || 
               doc.document.to_lowercase().contains("neural network") ||
               doc.document.to_lowercase().contains("deep learning") {
                yes_method_score += 10 - rank; // Higher rank = better score
            }
        }
        
        for (rank, doc) in self.no_prob_ranking.iter().enumerate() {
            if doc.document.to_lowercase().contains("machine learning") || 
               doc.document.to_lowercase().contains("neural network") ||
               doc.document.to_lowercase().contains("deep learning") {
                no_method_score += 10 - rank; // Higher rank = better score
            }
        }
        
        println!("🎯 RELEVANCE RANKING ANALYSIS:");
        println!("  YES probability method score: {}", yes_method_score);
        println!("  NO probability method score: {}", no_method_score);
        
        let yes_approach_correct = yes_method_score > no_method_score;
        let no_approach_correct = no_method_score > yes_method_score;
        
        println!();
        if yes_approach_correct {
            println!("  ✅ YES probability approach ranks relevant documents higher");
            println!("  ❌ Current implementation using NO probability is INCORRECT");
            println!("  🔧 BUG CONFIRMED: Should use YES probability in line 193");
        } else if no_approach_correct {
            println!("  ✅ NO probability approach ranks relevant documents higher");
            println!("  ✅ Current implementation using NO probability is CORRECT");
            println!("  📝 GGUF models have inverted behavior vs official docs");
        } else {
            println!("  ❓ Both methods perform similarly - inconclusive");
        }
        
        ScoringAnalysis {
            yes_approach_correct,
            no_approach_correct,
            rankings_differ: self.yes_prob_ranking != self.no_prob_ranking,
        }
    }
}

#[derive(Debug)]
pub struct ScoringAnalysis {
    pub yes_approach_correct: bool,
    pub no_approach_correct: bool,
    pub rankings_differ: bool,
}

fn truncate(s: &str, max_len: usize) -> String {
    if s.len() <= max_len {
        s.to_string()
    } else {
        format!("{}...", &s[..max_len])
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("🕵️ Investigating Qwen3 Reranker Scoring Bug");
    println!("Testing whether current implementation or official docs are correct");
    println!();
    
    let device = Device::Cpu;
    let investigator = ScoringInvestigator::from_hf(&device, Qwen3RerankSize::Size0_6B).await?;
    let tokenizer = investigator.get_tokenizer().await?;
    
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
    
    let comparison = investigator.compare_scoring_approaches(&tokenizer, query, &documents)?;
    let analysis = comparison.analyze_correctness();
    
    println!();
    println!("🔬 FINAL INVESTIGATION CONCLUSION:");
    if analysis.yes_approach_correct && !analysis.no_approach_correct {
        println!("  ❌ CURRENT IMPLEMENTATION IS BUGGY");
        println!("  ✅ Should use YES probability (as per official docs)");
        println!("  🔧 Fix: Change qwen3_reranker.rs:193 from Ok(no_prob) to Ok(yes_prob)");
    } else if !analysis.yes_approach_correct && analysis.no_approach_correct {
        println!("  ✅ CURRENT IMPLEMENTATION IS CORRECT");
        println!("  ❌ Official documentation appears to be wrong for GGUF models");
        println!("  📝 GGUF quantized models have inverted behavior");
    } else {
        println!("  ❓ RESULTS INCONCLUSIVE - Both methods perform similarly");
    }
    
    Ok(())
}