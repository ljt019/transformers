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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_message_vec_ext_last_user() {
        let messages = vec![
            Message::system("You are helpful"),
            Message::user("First question"),
            Message::assistant("First answer"),
            Message::user("Second question"),
        ];

        assert_eq!(messages.last_user(), Some("Second question"));
    }

    #[test]
    fn test_message_vec_ext_last_assistant() {
        let messages = vec![
            Message::system("You are helpful"),
            Message::user("Question"),
            Message::assistant("First answer"),
            Message::user("Follow up"),
            Message::assistant("Second answer"),
        ];

        assert_eq!(messages.last_assistant(), Some("Second answer"));
    }

    #[test]
    fn test_message_vec_ext_system() {
        let messages = vec![
            Message::system("You are helpful"),
            Message::user("Question"),
            Message::assistant("Answer"),
        ];

        assert_eq!(messages.system(), Some("You are helpful"));
    }

    #[test]
    fn test_message_vec_ext_empty() {
        let messages: Vec<Message> = vec![];
        assert_eq!(messages.last_user(), None);
        assert_eq!(messages.last_assistant(), None);
        assert_eq!(messages.system(), None);
    }
}
