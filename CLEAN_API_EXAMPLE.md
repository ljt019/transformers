# Clean XML Parser API Example

This document shows how the clean API would work once fully implemented.

## The Clean API Design

Instead of having separate methods like `completion_with_xml_parser()`, the API allows you to register an XML parser on the pipeline, and then the standard methods automatically return parsed events when a parser is registered.

```rust
use transformers::pipelines::text_generation_pipeline::*;
use futures::StreamExt;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Build the text generation pipeline
    let mut pipeline = TextGenerationPipelineBuilder::qwen3(Qwen3Size::Size0_6B)
        .max_len(200)
        .temperature(0.7)
        .build()?;

    // Build an XML parser to watch for specific tags
    let xml_parser = XmlParserBuilder::new()
        .register_tag("think")
        .register_tag("tool_response")
        .register_tag("analysis")
        .build();

    // Register the XML parser with the pipeline
    pipeline.register_xml_parser(xml_parser);

    // Now completion returns Output enum instead of String
    let result = pipeline.completion("What is the capital of France?")?;

    // Handle the Output enum
    match &result {
        Output::Events(events) => {
            println!("Parsed {} events:", events.len());
            for event in events {
                match event {
                    Event::Tagged { tag, content } => {
                        println!("  [{}]: {}", tag, content);
                    }
                    Event::Content(content) => {
                        println!("  Content: {}", content);
                    }
                }
            }
        }
        Output::Text(text) => {
            // This branch is taken if no XML parser is registered
            println!("Raw text: {}", text);
        }
    }

    // You can always get the raw text regardless of output type
    println!("\nFull text: {}", result.as_text());

    // Example with streaming
    println!("\n=== Streaming Example ===");
    let mut stream = pipeline.completion_stream("Explain quantum computing")?;
    
    while let Some(output) = stream.next().await {
        match output {
            StreamOutput::Event(event) => {
                // Handle individual XML events as they arrive
                match event {
                    Event::Tagged { tag, content } => {
                        println!("Event [{}]: {}", tag, content);
                    }
                    Event::Content(content) => {
                        print!("{}", content);
                    }
                }
            }
            StreamOutput::Text(text) => {
                // This is used when no XML parser is registered
                print!("{}", text);
            }
        }
    }

    Ok(())
}
```

## Benefits of This API

1. **Single API Surface**: Users don't need to learn different methods for XML parsing vs regular completion
2. **Opt-in Parsing**: XML parsing only happens when explicitly registered
3. **Type Safety**: The Output/StreamOutput enums make it clear what type of data you're working with
4. **Backwards Compatible**: Existing code can continue to work by not registering a parser
5. **Flexible**: Can easily switch between parsed and unparsed output by registering/unregistering the parser

## Implementation Status

- ✅ Core XML parser functionality
- ✅ Output and StreamOutput enums
- ✅ register_xml_parser() method
- ✅ Non-streaming completion methods updated
- ⚠️  Streaming methods partially implemented
- ❌ Tool calling integration needs updates
- ❌ Full example not yet compilable

The core concept is implemented but needs additional work to handle all edge cases with streaming and tool calling.