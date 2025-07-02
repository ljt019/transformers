use transformers::pipelines::text_generation_pipeline::{XmlParserBuilder, Event};

fn main() {
    println!("=== XML Parser Demo ===\n");

    // Create an XML parser that watches for 'think' and 'tool_response' tags
    let xml_parser = XmlParserBuilder::new()
        .register_tag("think")
        .register_tag("tool_response")
        .build();

    // Example 1: Basic parsing
    println!("Example 1: Basic parsing");
    let text1 = "<think>Processing request...</think>The answer is 42.";
    let events1 = xml_parser.parse_complete(text1);
    
    println!("Input: {}", text1);
    println!("Events:");
    for event in &events1 {
        match event {
            Event::Tagged { tag, content } => {
                println!("  - Tagged[{}]: {}", tag, content);
            }
            Event::Content(content) => {
                println!("  - Content: {}", content);
            }
        }
    }
    println!();

    // Example 2: Multiple tags
    println!("Example 2: Multiple tags");
    let text2 = "<think>Let me calculate...</think>Result: <tool_response>Calculation complete: 100</tool_response>";
    let events2 = xml_parser.parse_complete(text2);
    
    println!("Input: {}", text2);
    println!("Events:");
    for event in &events2 {
        match event {
            Event::Tagged { tag, content } => {
                println!("  - Tagged[{}]: {}", tag, content);
            }
            Event::Content(content) => {
                println!("  - Content: {}", content);
            }
        }
    }
    println!();

    // Example 3: Unregistered tags (treated as content)
    println!("Example 3: Unregistered tags");
    let text3 = "<think>Thinking...</think><unregistered>This tag is not registered</unregistered>Done.";
    let events3 = xml_parser.parse_complete(text3);
    
    println!("Input: {}", text3);
    println!("Events:");
    for event in &events3 {
        match event {
            Event::Tagged { tag, content } => {
                println!("  - Tagged[{}]: {}", tag, content);
            }
            Event::Content(content) => {
                println!("  - Content: {}", content);
            }
        }
    }
    println!();

    // Example 4: Streaming parsing
    println!("Example 4: Streaming parsing");
    let tokens = vec![
        "<think>",
        "Processing",
        " your",
        " request",
        "...</think>",
        "The result",
        " is: ",
        "<tool_response>",
        "Success!",
        "</tool_response>",
    ];
    
    xml_parser.reset(); // Reset for new parsing session
    let mut streaming_events = Vec::new();
    
    println!("Streaming tokens:");
    for token in &tokens {
        print!("{}", token);
        let events = xml_parser.parse_token(token);
        streaming_events.extend(events);
    }
    println!();
    
    // Flush any remaining events
    streaming_events.extend(xml_parser.flush());
    
    println!("Events:");
    for event in &streaming_events {
        match event {
            Event::Tagged { tag, content } => {
                println!("  - Tagged[{}]: {}", tag, content);
            }
            Event::Content(content) => {
                println!("  - Content: {}", content);
            }
        }
    }
    println!();

    // Example 5: Nested tags (inner unregistered tag inside registered tag)
    println!("Example 5: Nested tags");
    let text5 = "<think>I'm <em>really</em> thinking hard!</think>Done.";
    let events5 = xml_parser.parse_complete(text5);
    
    println!("Input: {}", text5);
    println!("Events:");
    for event in &events5 {
        match event {
            Event::Tagged { tag, content } => {
                println!("  - Tagged[{}]: {}", tag, content);
            }
            Event::Content(content) => {
                println!("  - Content: {}", content);
            }
        }
    }
    println!();

    // Example 6: Using event helper methods
    println!("Example 6: Event helper methods");
    let text6 = "<think>Helper test</think>Regular content";
    let events6 = xml_parser.parse_complete(text6);
    
    for event in &events6 {
        println!("Event content: '{}', Tag: {:?}", 
                 event.get_content(), 
                 event.tag());
    }
}