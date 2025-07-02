use transformers::pipelines::text_generation_pipeline::{
    xml_parser::{Event, Tag, XmlParser, XmlParserBuilder},
    StreamOutput,
};
use futures::StreamExt;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Example demonstrating StreamOutput enum handling
    
    // Create a tag registry
    struct Tags {
        think: Tag,
        tool_response: Tag,
    }
    
    impl Tags {
        fn new() -> (Self, XmlParser) {
            let mut builder = XmlParserBuilder::new();
            
            let tags = Tags {
                think: builder.register_tag("think"),
                tool_response: builder.register_tag("tool_response"),
            };
            
            (tags, builder.build())
        }
    }
    
    let (tags, xml_parser) = Tags::new();
    
    // Simulate a stream of StreamOutput items
    // In real usage, this would come from TextGenerationPipeline::completion_stream()
    let stream_items = vec![
        StreamOutput::Text("Let me ".to_string()),
        StreamOutput::Event(Event::Tagged { 
            tag: tags.think.clone(), 
            content: "analyze this problem".to_string() 
        }),
        StreamOutput::Text(" find the answer.\n".to_string()),
        StreamOutput::Event(Event::Tagged {
            tag: tags.tool_response.clone(),
            content: "Calculation: 2 + 2 = 4".to_string()
        }),
        StreamOutput::Text("\nThe answer is 4.".to_string()),
    ];
    
    println!("=== Processing StreamOutput Items ===\n");
    
    // Process the stream
    for (i, output) in stream_items.iter().enumerate() {
        println!("Item {}: {:?}", i + 1, output);
        
        match output {
            StreamOutput::Text(text) => {
                println!("  -> Raw text: {}", text);
            }
            StreamOutput::Event(event) => {
                match event {
                    Event::Tagged { tag, content } if tag == &tags.think => {
                        println!("  -> [THINKING] {}", content);
                    }
                    Event::Tagged { tag, content } if tag == &tags.tool_response => {
                        println!("  -> [TOOL OUTPUT] {}", content);
                    }
                    Event::Tagged { tag, content } => {
                        println!("  -> [{}] {}", tag.name(), content);
                    }
                    Event::Content(content) => {
                        println!("  -> Content: {}", content);
                    }
                }
            }
        }
        println!();
    }
    
    // Demonstrate extracting full text from mixed StreamOutput
    println!("\n=== Extracting Full Text ===");
    let full_text: String = stream_items.iter()
        .map(|output| output.as_text())
        .collect();
    println!("Full text: {}", full_text);
    
    // Demonstrate filtering for specific events
    println!("\n=== Filtering for Specific Events ===");
    let thinking_events: Vec<_> = stream_items.iter()
        .filter_map(|output| match output {
            StreamOutput::Event(Event::Tagged { tag, content }) if tag == &tags.think => {
                Some(content.as_str())
            }
            _ => None
        })
        .collect();
    
    println!("Thinking events: {:?}", thinking_events);
    
    // Demonstrate async stream processing (simulated)
    println!("\n=== Async Stream Processing ===");
    let stream = futures::stream::iter(stream_items);
    
    let mut stream = Box::pin(stream);
    while let Some(output) = stream.next().await {
        match output {
            StreamOutput::Text(text) => {
                print!("{}", text);
                // Simulate real-time streaming
                std::io::Write::flush(&mut std::io::stdout())?;
                tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
            }
            StreamOutput::Event(Event::Tagged { tag, content }) => {
                if tag == &tags.think {
                    print!(" <thinking: {}> ", content);
                } else if tag == &tags.tool_response {
                    print!("\n[TOOL: {}]\n", content);
                }
                std::io::Write::flush(&mut std::io::stdout())?;
                tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;
            }
            _ => {}
        }
    }
    
    println!("\n\nDone!");
    
    Ok(())
}