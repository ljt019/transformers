#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Tool {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value, // JSON Schema
}

impl Tool {
    pub fn new(name: String, description: String, parameters: serde_json::Value) -> Self {
        Self {
            name,
            description,
            parameters,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCall {
    pub tool_name: String,
    pub arguments: serde_json::Value,
}

impl ToolCall {
    pub fn new(tool_name: String, arguments: serde_json::Value) -> Self {
        Self {
            tool_name,
            arguments,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolCallResult {
    pub response: String,
    pub tool_calls: Vec<ToolCall>,
}

impl ToolCallResult {
    pub fn new(response: String, tool_calls: Vec<ToolCall>) -> Self {
        Self {
            response,
            tool_calls,
        }
    }

    pub fn simple_response(response: String) -> Self {
        Self {
            response,
            tool_calls: Vec::new(),
        }
    }
}
