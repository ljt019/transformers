use transformers::pipelines::text_generation_pipeline::*;
use futures::StreamExt;

fn main() -> anyhow::Result<()> {
    // Build the text generation pipeline
    let pipeline = TextGenerationPipelineBuilder::qwen3(Qwen3Size::Size0_6B)
        .max_len(200)
        .temperature(0.7)
        .build()?;

    // Build an XML parser to watch for 'think' and 'tool_response' tags
    let xml_parser = XmlParserBuilder::new()
        .register_tag("think")
        .register_tag("tool_response")
        .register_tag("analysis")
        .build();

    println!("=== XML Parsing Example ===\n");

    // Example 1: Non-streaming with XML parsing
    println!("1. Non-streaming XML parsing:");
    let prompt = "Think about the capital of France before answering. Use <think> tags for your reasoning.";
    
    // Simulate what the model might output
    let mock_response = "<think>The user is asking about the capital of France. I know this is Paris, which is a major city and the political center of France.</think>\n\nThe capital of France is **Paris**. It's been the capital since 987 AD and is home to about 2.1 million people in the city proper.";
    
    // Parse the mock response to demonstrate the API
    let events = xml_parser.parse_complete(mock_response);
    
    for event in &events {
        match event {
            Event::Tagged { tag, content } => {
                println!("  📋 [{}]: {}", tag, content);
            }
            Event::Content(content) => {
                println!("  💬 Content: {}", content);
            }
        }
    }

    println!("\n2. Streaming XML parsing simulation:");
    
    // Simulate streaming tokens
    let streaming_tokens = vec![
        "<think>", "Let", " me", " think", " about", " this", "...</think>",
        "\n\nThe", " answer", " is", " **Paris**."
    ];
    
    xml_parser.reset(); // Reset for new parsing session
    
    for token in streaming_tokens {
        let events = xml_parser.parse_token(token);
        for event in events {
            match event {
                Event::Tagged { tag, content } => {
                    println!("  🔄 [{}]: {}", tag, content);
                }
                Event::Content(content) => {
                    println!("  🔄 Content: {}", content);
                }
            }
        }
    }
    
    // Flush any remaining events
    let final_events = xml_parser.flush();
    for event in final_events {
        match event {
            Event::Tagged { tag, content } => {
                println!("  ✅ [{}]: {}", tag, content);
            }
            Event::Content(content) => {
                println!("  ✅ Content: {}", content);
            }
        }
    }

    println!("\n3. Real pipeline integration (commented out - requires model):");
    println!("   // Generate with XML parsing (non-streaming)");
    println!("   // let events = pipeline.completion_with_xml_parser(prompt, &xml_parser)?;");
    println!("   // for event in events {{ ... }}");
    println!();
    println!("   // Generate with XML parsing (streaming)");
    println!("   // let mut stream = pipeline.completion_stream_with_xml_parser(prompt, &xml_parser)?;");
    println!("   // while let Some(event) = stream.next().await {{ ... }}");

    /*
    // Uncomment these lines to test with a real model:
    
    // Non-streaming version
    let events = pipeline.completion_with_xml_parser(prompt, &xml_parser)?;
    println!("\nReal pipeline results:");
    for event in &events {
        match event {
            Event::Tagged { tag, content } => {
                println!("  📋 [{}]: {}", tag, content);
            }
            Event::Content(content) => {
                println!("  💬 Content: {}", content);
            }
        }
    }

    // Streaming version
    let mut stream = pipeline.completion_stream_with_xml_parser(prompt, &xml_parser)?;
    println!("\nStreaming results:");
    while let Some(event) = stream.next().await {
        match event {
            Event::Tagged { tag, content } => {
                println!("  🔄 [{}]: {}", tag, content);
            }
            Event::Content(content) => {
                print!("{}", content);
                std::io::stdout().flush()?;
            }
        }
    }
    */

    Ok(())
}

// Helper function to demonstrate API patterns
#[allow(dead_code)]
fn demonstrate_api_patterns() -> anyhow::Result<()> {
    // Build XML parser with fluent API
    let xml_parser = XmlParserBuilder::new()
        .register_tag("think")
        .register_tags(vec!["tool_response", "analysis", "summary"])
        .build();

    // The parser can handle various XML patterns:
    let test_cases = vec![
        // Basic case
        "<think>Simple thinking</think>Content after",
        
        // Multiple tags
        "<think>First</think>Between<analysis>Second</analysis>End",
        
        // Nested unregistered tags (treated as content within registered tags)
        "<think>I need to <em>emphasize</em> this point</think>",
        
        // Malformed XML (unclosed tags)
        "<think>Unclosed thinking tag",
        
        // Self-closing tags (ignored since they have no content)
        "<think/>Regular content",
        
        // Mixed content
        "Start<think>Thinking</think>Middle<unknown>Not tracked</unknown>End",
    ];

    println!("API Pattern Examples:");
    for (i, test) in test_cases.iter().enumerate() {
        println!("\nTest case {}: {}", i + 1, test);
        let events = xml_parser.parse_complete(test);
        for event in events {
            match event {
                Event::Tagged { tag, content } => {
                    println!("  → [{}]: '{}'", tag, content);
                }
                Event::Content(content) => {
                    println!("  → Content: '{}'", content);
                }
            }
        }
    }

    Ok(())
}

#[tokio::main]
async fn async_main() -> anyhow::Result<()> {
    main()?;
    demonstrate_api_patterns()?;
    Ok(())
}