use thiserror::Error;

/// Generic error type returned by any tool function.
#[derive(Debug, Error)]
pub enum ToolError {
    /// A domain-specific failure – message chosen by the tool author.
    #[error("{0}")]
    Message(String),

    /// The tool invocation failed because the caller supplied malformed
    /// parameters (wrong type, missing fields, etc.).
    #[error("parameter decoding failed: {0}")]
    Format(String),
}