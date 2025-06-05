# Text Generation Pipeline Architecture Implementation Plan

## Overview
Implement a trait-based architecture that allows different models to expose different capabilities (reasoning, tool calling, etc.) through type-safe interfaces.

## Core Design Decisions
1. **Capability Declaration**: In ModelOptions (compile-time known)
2. **Builder Pattern**: Generic builder with associated types
3. **Multi-Capability Support**: Dedicated combination structs for each capability set

## Implementation Phases

### Phase 1: Define Core Traits and Types

#### 1.1 Create Capability Enums and Structs
```rust
// In src/pipelines/capabilities.rs
pub enum ReasoningSupport {
    None,
    AlwaysOn,
    Toggleable,
}

pub struct ModelCapabilities {
    pub reasoning: ReasoningSupport,
    pub tool_calling: bool,
    pub streaming: bool,
}
```

#### 1.2 Define Pipeline Traits
```rust
// In src/pipelines/traits.rs
pub trait TextGenerationPipeline {
    fn prompt_completion(&self, prompt: &str, max_length: usize) -> anyhow::Result<String>;
    fn message_completion(&self, messages: Vec<Message>, max_length: usize) -> anyhow::Result<String>;
}

pub trait TextGenerationPipelineWithReasoning: TextGenerationPipeline {
    fn get_reasoning_trace(&self) -> Option<&str>;
}

pub trait TextGenerationPipelineWithToggleableReasoning: TextGenerationPipeline {
    fn enable_reasoning(&mut self);
    fn disable_reasoning(&mut self);
    fn is_reasoning_enabled(&self) -> bool;
}

pub trait TextGenerationPipelineWithTools: TextGenerationPipeline {
    fn register_tool(&mut self, tool: Tool) -> anyhow::Result<()>;
    fn unregister_tool(&mut self, tool_name: &str) -> anyhow::Result<()>;
    fn list_tools(&self) -> Vec<&Tool>;
    fn call_with_tools(&self, prompt: &str, max_length: usize) -> anyhow::Result<ToolCallResult>;
}
```

#### 1.3 Define Tool Types
```rust
// In src/pipelines/tools.rs
pub struct Tool {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value, // JSON Schema
}

pub struct ToolCall {
    pub tool_name: String,
    pub arguments: serde_json::Value,
}

pub struct ToolCallResult {
    pub response: String,
    pub tool_calls: Vec<ToolCall>,
}
```

### Phase 2: Create Concrete Pipeline Types

#### 2.1 Basic Pipeline (No Special Capabilities)
```rust
// In src/pipelines/basic_pipeline.rs
pub struct BasicPipeline {
    model: Box<dyn TextGenerationModel>,
    tokenizer: Tokenizer,
    eos_token_id: u32,
}

impl TextGenerationPipeline for BasicPipeline {
    // Implementation
}
```

#### 2.2 Reasoning Pipeline (Always-On Reasoning)
```rust
// In src/pipelines/reasoning_pipeline.rs
pub struct ReasoningPipeline {
    model: Box<dyn TextGenerationModel>,
    tokenizer: Tokenizer,
    eos_token_id: u32,
    reasoning_trace: Option<String>,
}

impl TextGenerationPipeline for ReasoningPipeline {
    // Implementation
}

impl TextGenerationPipelineWithReasoning for ReasoningPipeline {
    // Implementation
}
```

#### 2.3 Toggleable Reasoning Pipeline
```rust
// In src/pipelines/toggleable_reasoning_pipeline.rs
pub struct ToggleableReasoningPipeline {
    model: Box<dyn TextGenerationModel>,
    tokenizer: Tokenizer,
    eos_token_id: u32,
    reasoning_enabled: bool,
}

impl TextGenerationPipeline for ToggleableReasoningPipeline {
    // Implementation
}

impl TextGenerationPipelineWithToggleableReasoning for ToggleableReasoningPipeline {
    // Implementation
}
```

#### 2.4 Tool Calling Pipeline
```rust
// In src/pipelines/tool_calling_pipeline.rs
pub struct ToolCallingPipeline {
    model: Box<dyn TextGenerationModel>,
    tokenizer: Tokenizer,
    eos_token_id: u32,
    tools: Vec<Tool>,
}

impl TextGenerationPipeline for ToolCallingPipeline {
    // Implementation
}

impl TextGenerationPipelineWithTools for ToolCallingPipeline {
    // Implementation
}
```

#### 2.5 Combined Pipelines (Multi-Capability)
```rust
// In src/pipelines/combined_pipelines.rs

// For models like Qwen3 with toggleable reasoning + tools
pub struct ToggleableReasoningToolsPipeline {
    model: Box<dyn TextGenerationModel>,
    tokenizer: Tokenizer,
    eos_token_id: u32,
    reasoning_enabled: bool,
    tools: Vec<Tool>,
}

impl TextGenerationPipeline for ToggleableReasoningToolsPipeline {
    // Implementation
}

impl TextGenerationPipelineWithToggleableReasoning for ToggleableReasoningToolsPipeline {
    // Implementation
}

impl TextGenerationPipelineWithTools for ToggleableReasoningToolsPipeline {
    // Implementation
}
```

### Phase 3: Implement Associated Types System

#### 3.1 Model Options Trait
```rust
// In src/pipelines/model_options.rs
pub trait ModelOptionsType: Clone {
    type Pipeline;
    
    fn capabilities(&self) -> ModelCapabilities;
    fn build_pipeline(
        self,
        params: GenerationParams,
    ) -> anyhow::Result<Self::Pipeline>;
}
```

#### 3.2 Implement for Each Model Type

##### Model Capabilities Summary:
- **Qwen3**: Toggleable reasoning + Tool calling → `ToggleableReasoningToolsPipeline`
- **Gemma3**: Tool calling only → `ToolCallingPipeline`
- **Phi4**: Basic generation only → `BasicPipeline`

```rust
// Qwen3 - Toggleable reasoning + Tool calling
#[derive(Clone)]
pub struct Qwen3ModelOptions {
    pub size: Qwen3Size,
}

impl ModelOptionsType for Qwen3ModelOptions {
    type Pipeline = ToggleableReasoningToolsPipeline;
    
    fn capabilities(&self) -> ModelCapabilities {
        ModelCapabilities {
            reasoning: ReasoningSupport::Toggleable,
            tool_calling: true,
            streaming: false,
        }
    }
    
    fn build_pipeline(
        self,
        params: GenerationParams,
    ) -> anyhow::Result<Self::Pipeline> {
        // Build Qwen3 model and create pipeline
        let model = /* load Qwen3 model */;
        Ok(ToggleableReasoningToolsPipeline {
            model,
            tokenizer,
            eos_token_id,
            reasoning_enabled: false, // Default to off
            tools: Vec::new(),
        })
    }
}

// Gemma3 - Tool calling only
#[derive(Clone)]
pub struct Gemma3ModelOptions {
    pub size: Gemma3Size,
}

impl ModelOptionsType for Gemma3ModelOptions {
    type Pipeline = ToolCallingPipeline;
    
    fn capabilities(&self) -> ModelCapabilities {
        ModelCapabilities {
            reasoning: ReasoningSupport::None,
            tool_calling: true,
            streaming: false,
        }
    }
    
    fn build_pipeline(
        self,
        params: GenerationParams,
    ) -> anyhow::Result<Self::Pipeline> {
        // Build Gemma3 model and create pipeline
        let model = /* load Gemma3 model */;
        Ok(ToolCallingPipeline {
            model,
            tokenizer,
            eos_token_id,
            tools: Vec::new(),
        })
    }
}

// Phi4 - Basic generation only
#[derive(Clone)]
pub struct Phi4ModelOptions {
    pub size: Phi4Size,
}

impl ModelOptionsType for Phi4ModelOptions {
    type Pipeline = BasicPipeline;
    
    fn capabilities(&self) -> ModelCapabilities {
        ModelCapabilities {
            reasoning: ReasoningSupport::None,
            tool_calling: false,
            streaming: false,
        }
    }
    
    fn build_pipeline(
        self,
        params: GenerationParams,
    ) -> anyhow::Result<Self::Pipeline> {
        // Build Phi4 model and create pipeline
        let model = /* load Phi4 model */;
        Ok(BasicPipeline {
            model,
            tokenizer,
            eos_token_id,
        })
    }
}
```

### Phase 4: Update Builder

#### 4.1 Generic Builder
```rust
// In src/pipelines/text_generation_pipeline.rs
pub struct TextGenerationPipelineBuilder<M: ModelOptionsType> {
    model_options: M,
    temperature: Option<f64>,
    repeat_penalty: Option<f32>,
    repeat_last_n: Option<usize>,
    seed: Option<u64>,
}

impl<M: ModelOptionsType> TextGenerationPipelineBuilder<M> {
    pub fn new(model_options: M) -> Self {
        Self {
            model_options,
            temperature: None,
            repeat_penalty: None,
            repeat_last_n: None,
            seed: None,
        }
    }
    
    // Builder methods remain the same
    pub fn temperature(mut self, temperature: f64) -> Self {
        self.temperature = Some(temperature);
        self
    }
    
    pub fn repeat_penalty(mut self, repeat_penalty: f32) -> Self {
        self.repeat_penalty = Some(repeat_penalty);
        self
    }
    
    pub fn repeat_last_n(mut self, repeat_last_n: usize) -> Self {
        self.repeat_last_n = Some(repeat_last_n);
        self
    }
    
    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
    
    pub fn build(self) -> anyhow::Result<M::Pipeline> {
        let temperature = self.temperature.unwrap_or(crate::DEFAULT_TEMPERATURE);
        let repeat_penalty = self.repeat_penalty.unwrap_or(crate::DEFAULT_REPEAT_PENALTY);
        let repeat_last_n = self.repeat_last_n.unwrap_or(crate::DEFAULT_REPEAT_LAST_N);
        let seed = self.seed.unwrap_or(crate::DEFAULT_SEED);
        
        let generation_params = GenerationParams::new(temperature, repeat_penalty, repeat_last_n, seed);
        
        // build_pipeline will handle all model loading, tokenizer creation, etc.
        self.model_options.build_pipeline(generation_params)
    }
}
```

### Phase 5: Update Model Trait

#### 5.1 Extend TextGenerationModel Trait
```rust
// In src/pipelines/mod.rs
pub trait TextGenerationModel {
    // Existing methods
    fn load_tokenizer(&self) -> anyhow::Result<tokenizers::Tokenizer>;
    fn get_eos_token_str(&self) -> &str;
    fn format_prompt(&self, prompt: &str) -> String;
    fn format_messages(&self, messages: Vec<Message>) -> anyhow::Result<String>;
    fn prompt_with_tokens(
        &self,
        tokens: &[u32],
        max_length: usize,
        eos_token: u32,
    ) -> anyhow::Result<Vec<u32>>;
    
    // New methods for advanced capabilities
    fn format_prompt_with_reasoning(&self, prompt: &str, reasoning_enabled: bool) -> String {
        // Default implementation
        self.format_prompt(prompt)
    }
    
    fn format_prompt_with_tools(&self, prompt: &str, tools: &[Tool]) -> String {
        // Default implementation
        self.format_prompt(prompt)
    }
    
    fn parse_tool_calls(&self, response: &str) -> anyhow::Result<Vec<ToolCall>> {
        // Default implementation returns empty vec
        Ok(vec![])
    }
}
```

### Phase 6: Migration Strategy

1. **No backward compatibility**: Remove the old `ModelOptions` enum and `TextGenerationPipeline` struct entirely
2. **Clean break**: Users must migrate to the new model-specific options
3. **Documentation**: Comprehensive examples showing how to use the new API

### Phase 7: Testing Strategy

1. **Unit tests** for each pipeline type
2. **Integration tests** for model-specific capabilities
3. **Compile-time tests** to ensure type safety
4. **Example programs** demonstrating each capability

## Implementation Order

1. Start with traits and basic types (Phase 1)
2. **DELETE the old ModelOptions enum and TextGenerationPipeline struct**
3. Implement BasicPipeline as proof of concept (Phase 2.1)
4. Add ModelOptionsType trait and implement for one model (Phase 3)
5. Update builder to be generic (Phase 4)
6. Implement remaining pipeline types (Phase 2.2-2.5)
7. Migrate all models to new system (Phase 3.2)
8. Add tests and documentation

## Key Benefits

1. **Type Safety**: Users can only call methods their chosen model supports
2. **Discoverability**: IDE autocomplete shows available methods
3. **Extensibility**: Easy to add new capabilities without breaking existing code
4. **Performance**: No runtime overhead for capability checking
5. **User Experience**: Simple API - choose model, get appropriate pipeline

## Potential Challenges

1. **Complexity**: More types and traits to maintain
2. **Documentation**: Need clear docs for each pipeline type
3. **Breaking Changes**: This will break all existing user code (and that's okay)
4. **Testing**: More combinations to test

## Future Extensions

1. **Streaming**: Add `TextGenerationPipelineWithStreaming` trait
2. **Batch Processing**: Add batch methods to base trait
3. **Fine-tuning**: Add capability for models that support it
4. **Multi-modal**: Extend for image/audio inputs

## Important Implementation Notes

1. **Method Implementations**: All capability-specific methods (tool registration, reasoning toggle, etc.) should use `todo!()` for now:
   ```rust
   impl TextGenerationPipelineWithToggleableReasoning for ToggleableReasoningToolsPipeline {
       fn enable_reasoning(&mut self) {
           todo!("Reasoning support not yet implemented")
       }
       
       fn disable_reasoning(&mut self) {
           todo!("Reasoning support not yet implemented")
       }
       
       fn is_reasoning_enabled(&self) -> bool {
           todo!("Reasoning support not yet implemented")
       }
   }
   ```

2. **Model Loading**: The actual model loading logic in `build_pipeline` methods will need to be adapted from the existing code but modified to return the appropriate pipeline type.

3. **No Backward Compatibility**: Delete the old `ModelOptions` enum and `TextGenerationPipeline` struct. This is a breaking change and that's fine.

4. **Error Handling**: Ensure all `todo!()` macros have descriptive messages for clarity during development.

## Additional Considerations

### Trait Object Support
Since we're using associated types, users won't be able to store different pipeline types in the same collection. If this is needed, we could provide:
```rust
// A trait object wrapper for dynamic dispatch
pub struct DynamicTextGenerationPipeline {
    inner: Box<dyn TextGenerationPipeline>,
}
```

### User API Examples

#### Basic Usage (Phi4)
```rust
use transformers::pipelines::{TextGenerationPipelineBuilder, Phi4ModelOptions, Phi4Size};

let pipeline = TextGenerationPipelineBuilder::new(
    Phi4ModelOptions { size: Phi4Size::Size14B }
)
.temperature(0.7)
.build()?;

// Only basic methods available
let response = pipeline.prompt_completion("Hello, world!", 100)?;
```

#### Tool Calling (Gemma3)
```rust
use transformers::pipelines::{TextGenerationPipelineBuilder, Gemma3ModelOptions, Gemma3Size, Tool};

let mut pipeline = TextGenerationPipelineBuilder::new(
    Gemma3ModelOptions { size: Gemma3Size::Size4B }
)
.build()?;

// Tool methods are available
pipeline.register_tool(Tool {
    name: "calculator".to_string(),
    description: "Performs basic math".to_string(),
    parameters: serde_json::json!({
        "type": "object",
        "properties": {
            "expression": { "type": "string" }
        }
    }),
})?;

let result = pipeline.call_with_tools("What is 2+2?", 100)?;
```

#### Full Featured (Qwen3)
```rust
use transformers::pipelines::{TextGenerationPipelineBuilder, Qwen3ModelOptions, Qwen3Size};

let mut pipeline = TextGenerationPipelineBuilder::new(
    Qwen3ModelOptions { size: Qwen3Size::Size8B }
)
.build()?;

// Both reasoning and tool methods available
pipeline.enable_reasoning();
pipeline.register_tool(my_tool)?;

let result = pipeline.call_with_tools("Solve this step by step: ...", 500)?;
```

### File Organization
```
src/pipelines/
├── mod.rs                    # Traits and re-exports
├── traits.rs                 # All pipeline traits
├── capabilities.rs           # Capability enums/structs
├── tools.rs                  # Tool-related types
├── model_options.rs          # ModelOptionsType trait and impls
├── basic_pipeline.rs         # BasicPipeline impl
├── tool_calling_pipeline.rs  # ToolCallingPipeline impl
├── combined_pipelines.rs     # ToggleableReasoningToolsPipeline impl
└── text_generation_pipeline.rs # Builder (updated to be generic)
```
