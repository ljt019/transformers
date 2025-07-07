use anyhow::Result;
use transformers::pipelines::reranker_pipeline::RerankPipelineBuilder;
use transformers::models::implementations::qwen3_reranker::Qwen3RerankSize;

#[tokio::main]
async fn main() -> Result<()> {
    // Create a reranker pipeline using Qwen3-Reranker-0.6B
    let pipeline = RerankPipelineBuilder::qwen3(Qwen3RerankSize::Size0_6B)
        .cpu()
        .build()
        .await?;

    // Test cases with different domains and complexity levels
    let test_cases = vec![
        (
            "What is machine learning?",
            vec![
                "Machine learning is a subset of artificial intelligence that focuses on algorithms and statistical models.",
                "Cooking recipes often require precise measurements and timing for best results.",
                "Supervised learning uses labeled data to train models for prediction tasks.",
                "The weather today is sunny and warm with light clouds.",
                "Deep learning is a type of machine learning using neural networks with multiple layers.",
                "Artificial intelligence encompasses machine learning, natural language processing, and computer vision.",
                "Python is a popular programming language for data science and machine learning applications.",
                "Basketball is a team sport played on a rectangular court with two hoops.",
                "Reinforcement learning agents learn through interaction with their environment.",
                "Medieval history covers the period from the 5th to the 15th century.",
            ]
        ),
        (
            "How do I cook pasta?",
            vec![
                "Bring a large pot of salted water to a boil, add pasta and cook until al dente.",
                "Machine learning algorithms require large datasets for training.",
                "Pasta comes in many shapes like spaghetti, penne, fusilli, and rigatoni.",
                "The stock market experienced volatility due to economic uncertainty.",
                "Salt the pasta water generously - it should taste like seawater.",
                "Neural networks consist of interconnected nodes that process information.",
                "Drain the pasta and reserve some pasta water for sauce consistency.",
                "Climate change affects global weather patterns and ecosystems.",
                "Fresh pasta cooks faster than dried pasta, usually in 2-3 minutes.",
                "Quantum computing uses quantum mechanical phenomena for computation.",
            ]
        ),
        (
            "What are the benefits of renewable energy?",
            vec![
                "Solar panels convert sunlight into electricity with no emissions during operation.",
                "The best pizza toppings include pepperoni, mushrooms, and bell peppers.",
                "Wind turbines generate clean electricity from wind power.",
                "Machine learning models can predict energy consumption patterns.",
                "Renewable energy sources help reduce greenhouse gas emissions and combat climate change.",
                "Hydroelectric power harnesses flowing water to generate electricity.",
                "Video games have become increasingly popular as entertainment media.",
                "Renewable energy creates jobs in manufacturing, installation, and maintenance.",
                "Geothermal energy taps into the Earth's internal heat for power generation.",
                "Social media platforms connect people across the globe.",
            ]
        ),
        (
            "How do neural networks work?",
            vec![
                "Neural networks are inspired by biological neurons and process information through interconnected layers.",
                "Gardening requires understanding soil types, watering schedules, and plant care.",
                "Backpropagation is the algorithm used to train neural networks by adjusting weights.",
                "Travel insurance protects against unexpected costs during trips.",
                "Activation functions like ReLU and sigmoid determine neuron output in neural networks.",
                "Convolutional neural networks are particularly effective for image recognition tasks.",
                "Movie theaters offer immersive entertainment experiences with large screens and surround sound.",
                "Gradient descent optimization helps neural networks learn from training data.",
                "Recurrent neural networks can process sequential data like text and time series.",
                "Cooking techniques vary across different cultures and cuisines.",
            ]
        ),
    ];

    // Evaluate each test case
    for (i, (query, documents)) in test_cases.iter().enumerate() {
        println!("=== Test Case {} ===", i + 1);
        println!("Query: {}", query);
        println!();

        // Rerank the documents
        let ranked_results = pipeline.rerank(query, documents)?;

        println!("All ranked documents:");
        for (rank, (doc_idx, score)) in ranked_results.iter().enumerate() {
            println!("{}. [Score: {:.4}] {}", rank + 1, score, documents[*doc_idx]);
        }

        // Get top-3 results
        let top_3 = pipeline.rerank_top_k(query, documents, 3)?;
        println!("\nTop 3 results:");
        for (rank, (doc_idx, score)) in top_3.iter().enumerate() {
            println!("{}. [Score: {:.4}] {}", rank + 1, score, documents[*doc_idx]);
        }

        // Calculate relevance metrics
        let relevant_docs = identify_relevant_docs(i, documents);
        let precision_at_3 = calculate_precision_at_k(&top_3, &relevant_docs, 3);
        let ndcg_at_3 = calculate_ndcg_at_k(&top_3, &relevant_docs, 3);

        println!("\nMetrics:");
        println!("Precision@3: {:.4}", precision_at_3);
        println!("NDCG@3: {:.4}", ndcg_at_3);
        println!("Relevant docs: {:?}", relevant_docs);
        println!("{}", "=".repeat(80));
        println!();
    }

    Ok(())
}

/// Identify which documents are relevant for each test case
fn identify_relevant_docs(test_case_idx: usize, documents: &[&str]) -> Vec<usize> {
    match test_case_idx {
        0 => { // Machine learning query
            documents.iter().enumerate()
                .filter_map(|(i, doc)| {
                    if doc.contains("machine learning") || doc.contains("supervised learning") || 
                       doc.contains("deep learning") || doc.contains("artificial intelligence") ||
                       doc.contains("neural networks") || doc.contains("reinforcement learning") ||
                       doc.contains("Python") && doc.contains("data science") {
                        Some(i)
                    } else {
                        None
                    }
                })
                .collect()
        },
        1 => { // Cooking pasta query
            documents.iter().enumerate()
                .filter_map(|(i, doc)| {
                    if doc.contains("pasta") || doc.contains("boil") || doc.contains("cook") ||
                       doc.contains("salt") && doc.contains("water") || doc.contains("drain") {
                        Some(i)
                    } else {
                        None
                    }
                })
                .collect()
        },
        2 => { // Renewable energy query
            documents.iter().enumerate()
                .filter_map(|(i, doc)| {
                    if doc.contains("solar") || doc.contains("wind") || doc.contains("renewable") ||
                       doc.contains("hydroelectric") || doc.contains("geothermal") ||
                       doc.contains("emissions") || doc.contains("clean electricity") {
                        Some(i)
                    } else {
                        None
                    }
                })
                .collect()
        },
        3 => { // Neural networks query
            documents.iter().enumerate()
                .filter_map(|(i, doc)| {
                    if doc.contains("neural networks") || doc.contains("backpropagation") ||
                       doc.contains("activation functions") || doc.contains("convolutional") ||
                       doc.contains("recurrent") || doc.contains("gradient descent") {
                        Some(i)
                    } else {
                        None
                    }
                })
                .collect()
        },
        _ => vec![]
    }
}

/// Calculate Precision@K
fn calculate_precision_at_k(ranked_results: &[(usize, f32)], relevant_docs: &[usize], k: usize) -> f32 {
    let top_k = ranked_results.iter().take(k).map(|(idx, _)| *idx).collect::<Vec<_>>();
    let relevant_in_top_k = top_k.iter().filter(|&&idx| relevant_docs.contains(&idx)).count();
    relevant_in_top_k as f32 / k as f32
}

/// Calculate NDCG@K (Normalized Discounted Cumulative Gain)
fn calculate_ndcg_at_k(ranked_results: &[(usize, f32)], relevant_docs: &[usize], k: usize) -> f32 {
    let dcg = ranked_results.iter().take(k).enumerate()
        .map(|(rank, (doc_idx, _))| {
            let relevance = if relevant_docs.contains(doc_idx) { 1.0 } else { 0.0 };
            relevance / ((rank + 2) as f32).log2() // +2 because rank is 0-indexed and we want log2(rank+1)
        })
        .sum::<f32>();
    
    // Calculate IDCG (Ideal DCG) - assumes all relevant docs are ranked first
    let ideal_dcg = (0..k.min(relevant_docs.len())).map(|rank| {
        1.0 / ((rank + 2) as f32).log2()
    }).sum::<f32>();
    
    if ideal_dcg == 0.0 {
        0.0
    } else {
        dcg / ideal_dcg
    }
}