#![allow(warnings)]
#![allow(unused_imports)]
#![allow(dead_code)]
#![allow(unused_variables)]
#![allow(unused_mut)]

mod loaders;
pub mod models;
pub mod pipelines;

pub(crate) const DEFAULT_TEMPERATURE: f64 = 0.7;
pub(crate) const DEFAULT_REPEAT_PENALTY: f32 = 1.1;
pub(crate) const DEFAULT_REPEAT_LAST_N: usize = 64;
pub(crate) const DEFAULT_SEED: u64 = 299792458;

// Re-export the `#[tool]` procedural macro so users can simply write
// `use transformers::tool;` and annotate their functions without adding an
// explicit dependency on the `tool_macro` crate.
// The macro lives in the separate `tool_macro` crate to avoid a proc-macro/
// normal crate cyclic dependency, but re-exporting it here keeps the public
// API surface of `transformers` ergonomic.

pub use tool_macro::tool;

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

/// Trait extension for Vec<Message> that provides convenient methods for
/// accessing common message types without verbose iterator chains.
pub trait MessageVecExt {
    /// Get the content of the last user message in the conversation.
    ///
    /// # Returns
    /// - `Some(&str)` - The content of the last user message if found
    /// - `None` - If no user messages exist in the conversation
    fn last_user(&self) -> Option<&str>;

    /// Get the content of the last assistant message in the conversation.
    ///
    /// # Returns
    /// - `Some(&str)` - The content of the last assistant message if found
    /// - `None` - If no assistant messages exist in the conversation
    fn last_assistant(&self) -> Option<&str>;

    /// Get the content of the system message in the conversation.
    ///
    /// # Returns
    /// - `Some(&str)` - The content of the system message if found
    /// - `None` - If no system message exists in the conversation
    fn system(&self) -> Option<&str>;
}

impl<T: AsRef<[Message]>> MessageVecExt for T {
    fn last_user(&self) -> Option<&str> {
        self.as_ref()
            .iter()
            .rev()
            .find(|message| message.role() == "user")
            .map(|msg| msg.content())
    }

    fn last_assistant(&self) -> Option<&str> {
        self.as_ref()
            .iter()
            .rev()
            .find(|message| message.role() == "assistant")
            .map(|msg| msg.content())
    }

    fn system(&self) -> Option<&str> {
        self.as_ref()
            .iter()
            .find(|message| message.role() == "system")
            .map(|msg| msg.content())
    }
}

/// Prelude module for convenient imports.
///
/// This module re-exports the most commonly used types and traits from the transformers crate.
/// Import this module to get access to `Message` and `MessageVecExt` together:
///
/// ```rust
/// use transformers::prelude::*;
/// ```
pub mod prelude {
    pub use crate::{Message, MessageVecExt};
}
