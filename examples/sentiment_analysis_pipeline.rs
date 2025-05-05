use anyhow::Result;
use transformers::pipelines::sentiment_analysis_pipeline::*;

fn main() -> Result<()> {
    println!("Building sentiment analysis pipeline...");

    // 1. Create the pipeline, selecting the desired model size.
    let pipeline = SentimentAnalysisPipelineBuilder::new(SentimentModernBertSize::Base).build()?;

    println!("Pipeline built successfully.");

    // 2. Define some test sentences with different expected sentiments
    let sentences = [
        "I absolutely loved this movie! The acting was superb.", // Expected: positive
        "This is the worst experience I've ever had. Terrible customer service.", // Expected: negative
        "I'm not sure what to think about this product. It's okay, but I don't love it.", // Expected: neutral
        "Ce film est absolument incroyable!", // Expected: positive (French)
        "El servicio fue bastante decepcionante.", // Expected: negative (Spanish)
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
