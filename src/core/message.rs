#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
/// Role of a message in a chat conversation.
pub enum Role {
    /// System messages provide instructions to the model.
    System,
    /// User messages are sent from the user to the model.
    User,
    /// Assistant messages are responses from the model.
    Assistant,
}

impl Role {
    /// Returns the string representation of the role.
    pub fn as_str(&self) -> &'static str {
        match self {
            Role::System => "system",
            Role::User => "user",
            Role::Assistant => "assistant",
        }
    }
}

impl std::fmt::Display for Role {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
/// An individual message in a chat.
pub struct Message {
    role: Role,
    content: String,
}

impl Message {
    /// Create a new system message.
    ///
    /// System messages are used to provide instructions to the model.
    /// It's not recommended to use more than one of these in a given chat.
    pub fn system(content: &str) -> Self {
        Self {
            role: Role::System,
            content: content.to_string(),
        }
    }

    /// Create a new user message.
    ///
    /// User messages are used to send messages from the user to the model.
    pub fn user(content: &str) -> Self {
        Self {
            role: Role::User,
            content: content.to_string(),
        }
    }

    /// Create a new assistant message.
    ///
    /// Assistant messages are used to store responses from the model.
    pub fn assistant(content: &str) -> Self {
        Self {
            role: Role::Assistant,
            content: content.to_string(),
        }
    }

    /// Get the role of the message.
    pub fn role(&self) -> &Role {
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
            .find(|message| message.role() == &Role::User)
            .map(|msg| msg.content())
    }

    fn last_assistant(&self) -> Option<&str> {
        self.as_ref()
            .iter()
            .rev()
            .find(|message| message.role() == &Role::Assistant)
            .map(|msg| msg.content())
    }

    fn system(&self) -> Option<&str> {
        self.as_ref()
            .iter()
            .find(|message| message.role() == &Role::System)
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
