# XML Parser Clean API Implementation Status

## Summary

The user requested a cleaner API where:
- An XML parser is registered on the pipeline with `pipeline.register_xml_parser(xml_parser)`
- The same `completion()` and `completion_stream()` methods return either Events or Strings based on whether an XML parser is registered
- No need for separate methods like `completion_with_xml_parser()`

## What Was Implemented

### ✅ Core Infrastructure
1. **Added `xml_parser: Option<XmlParser>` field** to `TextGenerationPipeline`
2. **Added `register_xml_parser()` method** to register an XML parser on the pipeline
3. **Created `Output` enum** that can represent either `Text(String)` or `Events(Vec<Event>)`
4. **Created `StreamOutput` enum** for streaming with `Text(String)` or `Event(Event)` variants
5. **Updated non-streaming completion methods** to return `Output` instead of `String`

### ✅ Partially Implemented
1. **Non-streaming methods** (`completion`, `prompt_completion`, `message_completion`) now:
   - Check if an XML parser is registered
   - Return `Output::Events` if parser exists, `Output::Text` otherwise
   - Use internal `_internal` methods for the actual text generation

2. **Removed old XML parser methods** like `completion_with_xml_parser` to clean up the API

## What Still Needs Work

### ❌ Streaming Methods
The streaming methods (`completion_stream`, `prompt_completion_stream`, `message_completion_stream`) need to:
- Return `Stream<Item = StreamOutput>` instead of `Stream<Item = String>`
- Handle XML parsing in the stream if a parser is registered

### ❌ Tool Calling Integration
The tool calling methods still expect strings and need updates to work with the new `Output`/`StreamOutput` types:
- `completion_with_tools` 
- `completion_stream_with_tools`
- And related methods

### ❌ Type Compatibility Issues
Several compilation errors remain due to:
- Methods expecting `String` but receiving `Output` or `StreamOutput`
- Stream type mismatches between different branches
- Tool calling code that processes strings directly

## Recommended Next Steps

1. **Complete streaming implementation**: Update all streaming methods to properly return `StreamOutput`
2. **Update tool calling**: Modify tool methods to work with the new output types
3. **Add helper methods**: Create convenience methods to extract text from Output/StreamOutput when needed
4. **Update examples**: Create comprehensive examples showing the new API usage
5. **Testing**: Add tests for the new functionality

## Example Usage (When Complete)

```rust
use transformers::pipelines::text_generation_pipeline::*;

// Build pipeline and register XML parser
let mut pipeline = TextGenerationPipelineBuilder::new()
    .model(...) 
    .build()?;

let xml_parser = XmlParserBuilder::new()
    .register_tag("think")
    .register_tag("tool_response")
    .build();

pipeline.register_xml_parser(xml_parser);

// Non-streaming: returns Output enum
let result = pipeline.completion("What is 2+2?")?;
match result {
    Output::Events(events) => {
        for event in events {
            // Handle parsed XML events
        }
    }
    Output::Text(text) => {
        // Handle raw text
    }
}

// Streaming: returns Stream<Item = StreamOutput>
let mut stream = pipeline.completion_stream("Explain AI")?;
while let Some(output) = stream.next().await {
    match output {
        StreamOutput::Event(event) => { /* handle event */ }
        StreamOutput::Text(text) => { /* handle text chunk */ }
    }
}
```

## Current Status

The implementation is approximately **40% complete**. The core infrastructure and non-streaming methods are mostly done, but the streaming methods and tool calling integration need significant work to fully support the clean API design.