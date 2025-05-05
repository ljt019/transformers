use anyhow::Result;
use transformers::pipelines::sentiment_analysis_pipeline::*;

fn main() -> Result<()> {
    println!("Building sentiment analysis pipeline...");

    // 1. Create the pipeline, selecting the desired model size.
    // The default model is clapAI/modernBERT-base-multilingual-sentiment.
    let pipeline = SentimentAnalysisPipelineBuilder::new(SentimentModernBertSize::Base)
        // Optionally override model_id, revision, specify local files, or force CPU:
        // .model_id("another/sentiment-model")
        // .revision("v1.0")
        // .tokenizer_file("path/to/tokenizer.json")
        // .cpu()
        .build()?;

    println!("Pipeline built successfully.");

    // 2. Define some test sentences with different expected sentiments
    let sentences = [
        "I absolutely loved this movie! The acting was superb.", // Expected: positive
        "The service was okay, but the food could have been better.", // Expected: neutral (or slightly negative)
        "This is the worst experience I've ever had. Terrible customer service.", // Expected: negative
        "Ce film est absolument incroyable!", // French: positive
        "El servicio fue bastante decepcionante.", // Spanish: negative
        "I'm not sure what to think about this product. It's okay, but I don't love it.", // Expected: neutral
    ];

    // 3. Analyze each sentence
    for (i, sentence) in sentences.iter().enumerate() {
        println!("\nAnalyzing sentiment of: '{}'", sentence);

        // Get the sentiment classification label
        let sentiment_label = pipeline.predict(sentence)?;

        println!("--- Result {} ---", i + 1);
        println!("Text: {}", sentence);
        println!("Predicted Sentiment: {}", sentiment_label);
        println!("--- End {} ---", i + 1);
    }

    Ok(())
}
