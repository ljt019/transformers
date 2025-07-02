use transformers::pipelines::text_generation_pipeline::xml_parser::{Event, Tag, XmlParser, XmlParserBuilder};

fn main() -> anyhow::Result<()> {
    // This example focuses on the XML parser Tag Registry Pattern
    // In a real application, you would use this with a TextGenerationPipeline

    // Create a tag registry for type safety
    struct Tags {
        think: Tag,
        tool_response: Tag,
        analysis: Tag,
    }

    impl Tags {
        fn new() -> (Self, XmlParser) {
            let mut builder = XmlParserBuilder::new();
            
            let tags = Tags {
                think: builder.register_tag("think"),
                tool_response: builder.register_tag("tool_response"),
                analysis: builder.register_tag("analysis"),
            };
            
            (tags, builder.build())
        }
    }

    // Create tags and parser together
    let (tags, xml_parser) = Tags::new();

    // Example text with XML tags
    let text = "<think>Let me analyze this problem...</think>The answer is 42. <tool_response>Calculation complete</tool_response>";
    
    // Parse the text
    let events = xml_parser.parse_complete(text);

    // Type-safe tag matching
    for event in &events {
        match event {
            Event::Tagged { tag, content } if tag == &tags.think => {
                println!("THOUGHTS START");
                println!("{}", content);
                println!("THOUGHTS END\n");
            }
            Event::Tagged { tag, content } if tag == &tags.tool_response => {
                println!("TOOL RESPONSE: {}\n", content);
            }
            Event::Tagged { tag, content } if tag == &tags.analysis => {
                println!("ANALYSIS: {}\n", content);
            }
            Event::Tagged { tag, .. } => {
                println!("ERROR: Unknown tag - {}", tag.name());
            }
            Event::Content(content) => {
                println!("CONTENT: {}", content);
            }
        }
    }

    // Alternative: Using match with tag names (still type-safe due to Tag equality implementations)
    println!("\n=== Alternative matching approach ===");
    for event in &events {
        match event {
            Event::Tagged { tag, content } => {
                match tag.name() {
                    "think" => println!("[Thinking] {}", content),
                    "tool_response" => println!("[Tool] {}", content),
                    "analysis" => println!("[Analysis] {}", content),
                    _ => println!("[Unknown: {}] {}", tag.name(), content),
                }
            }
            Event::Content(content) => {
                println!("{}", content);
            }
        }
    }

    // Example with streaming
    println!("\n=== Streaming example ===");
    let tokens = vec![
        "<think>", 
        "Processing", 
        " request", 
        "...</think>",
        "Result: ",
        "<tool_response>",
        "Success!",
        "</tool_response>"
    ];
    
    xml_parser.reset(); // Reset for new parsing session
    for token in &tokens {
        print!("{}", token); // Show the token
        let events = xml_parser.parse_token(token);
        for event in events {
            match event {
                Event::Tagged { tag, content } if tag == &tags.think => {
                    print!(" [Got thinking event: {}]", content);
                }
                Event::Tagged { tag, content } if tag == &tags.tool_response => {
                    print!(" [Got tool response: {}]", content);
                }
                _ => {}
            }
        }
    }
    
    // Flush any remaining events
    let final_events = xml_parser.flush();
    for event in final_events {
        match event {
            Event::Content(content) => println!("\nFinal content: {}", content),
            _ => {}
        }
    }

    Ok(())
}