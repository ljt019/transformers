# XML Parser Feature with Tag Registry Pattern

## Overview
This feature adds XML parsing capabilities to the text generation pipeline, allowing extraction of content from specific XML tags in model outputs. The implementation uses a Tag Registry Pattern for type-safe tag handling.

## Core Components

### 1. Tag Registry Pattern
The Tag Registry Pattern provides type-safe tag handling by returning `Tag` handles when registering tags:

```rust
// Instead of string-based registration
let parser = XmlParserBuilder::new()
    .register_tag("think")
    .build();

// Tag Registry Pattern returns handles
let mut builder = XmlParserBuilder::new();
let think_tag = builder.register_tag("think");
let tool_response_tag = builder.register_tag("tool_response");
let parser = builder.build();
```

### 2. Tag Structure
```rust
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Tag {
    name: String,
    id: usize,
}
```

Tags provide:
- Unique ID for fast comparison
- String name for display/debugging
- Equality implementations for convenient matching

### 3. Event Enum
```rust
pub enum Event {
    Tagged { tag: Tag, content: String },
    Content(String),
}
```

### 4. Type-Safe Pattern Matching
```rust
match event {
    Event::Tagged { tag, content } if tag == &think_tag => {
        // Handle thinking content
    }
    Event::Tagged { tag, content } if tag == &tool_response_tag => {
        // Handle tool response
    }
    Event::Content(content) => {
        // Handle regular content
    }
}
```

## API Examples

### Basic Usage
```rust
// Create parser with tag registry
let mut builder = XmlParserBuilder::new();
let think_tag = builder.register_tag("think");
let parser = builder.build();

// Parse complete text
let events = parser.parse_complete("<think>Hello</think>World");

// Process events
for event in events {
    match event {
        Event::Tagged { tag, content } if tag == &think_tag => {
            println!("Thinking: {}", content);
        }
        Event::Content(content) => {
            println!("Content: {}", content);
        }
        _ => {}
    }
}
```

### Streaming Usage
```rust
parser.reset(); // Reset for new parsing session

// Parse tokens as they arrive
for token in tokens {
    let events = parser.parse_token(token);
    for event in events {
        // Process events immediately
    }
}

// Flush any remaining content
let final_events = parser.flush();
```

### Recommended Pattern: Tag Registry Struct
```rust
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

// Usage
let (tags, parser) = Tags::new();
```

## Integration with Pipeline

The XML parser can be registered on a TextGenerationPipeline:

```rust
let mut pipeline = TextGenerationPipelineBuilder::new()
    .model(model)
    .build()?;

let (tags, xml_parser) = Tags::new();
pipeline.register_xml_parser(xml_parser);
```

### Future Work: Automatic Event Parsing
While not yet implemented, the design supports future automatic parsing where:
- `completion()` would return `Output::Events(Vec<Event>)` when parser is registered
- `completion_stream()` would yield `StreamOutput::Event(Event)` items

## Benefits of Tag Registry Pattern

1. **Type Safety**: No string duplication between registration and matching
2. **Compile-Time Checks**: Typos in tag names are caught at compile time
3. **Performance**: Tag comparison uses ID equality checks
4. **Maintainability**: All tags defined in one place
5. **Refactoring**: Changing tag names only requires updating registration

## Implementation Details

- **Thread-Safe**: Parser uses `Arc<Mutex<>>` for safe concurrent access
- **Streaming Support**: Character-by-character parsing with state machine
- **Nested Tags**: Proper handling of nested XML structures
- **Error Resilience**: Gracefully handles malformed XML
- **Reset Capability**: Can reset state between parsing sessions

## Testing
Comprehensive test suite covers:
- Basic parsing
- Streaming parsing
- Multiple tags
- Nested tags
- Unregistered tags
- Malformed XML

All tests pass with the Tag Registry Pattern implementation.