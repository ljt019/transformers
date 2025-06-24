mod utils;

pub mod models;
pub mod pipelines;

pub(crate) const DEFAULT_TEMPERATURE: f64 = 0.7;
pub(crate) const DEFAULT_REPEAT_PENALTY: f32 = 1.1;
pub(crate) const DEFAULT_REPEAT_LAST_N: usize = 64;
pub(crate) const DEFAULT_SEED: u64 = 299792458;

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
/// An individual message in a chat.
pub struct Message {
    role: String,
    content: String,
}

impl Message {
    /// Create a new system message.
    ///
    /// System messages are used to provide instructions to the model.
    /// It's not recommended to use more than one of these in a given chat.
    pub fn system(content: &str) -> Self {
        Self {
            role: "system".to_string(),
            content: content.to_string(),
        }
    }

    /// Create a new user message.
    ///
    /// User messages are used to send messages from the user to the model.
    pub fn user(content: &str) -> Self {
        Self {
            role: "user".to_string(),
            content: content.to_string(),
        }
    }

    /// Create a new assistant message.
    ///
    /// Assistant messages are used to store responses from the model.
    pub fn assistant(content: &str) -> Self {
        Self {
            role: "assistant".to_string(),
            content: content.to_string(),
        }
    }

    /// Get the role of the message.
    pub fn role(&self) -> &str {
        &self.role
    }

    /// Get the content of the message.
    pub fn content(&self) -> &str {
        &self.content
    }
}
