use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};

/// A handle to a registered XML tag
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Tag {
    name: String,
    id: usize,
}

impl Tag {
    /// Get the name of this tag
    pub fn name(&self) -> &str {
        &self.name
    }
    
    /// Get the unique ID of this tag
    pub fn id(&self) -> usize {
        self.id
    }
}

impl PartialEq<str> for Tag {
    fn eq(&self, other: &str) -> bool {
        self.name == other
    }
}

impl PartialEq<&str> for Tag {
    fn eq(&self, other: &&str) -> bool {
        self.name == *other
    }
}

impl PartialEq<String> for Tag {
    fn eq(&self, other: &String) -> bool {
        self.name == *other
    }
}

impl PartialEq<&Tag> for Tag {
    fn eq(&self, other: &&Tag) -> bool {
        self == *other
    }
}

/// An event emitted by the XML parser containing the tag and content
#[derive(Debug, Clone, PartialEq)]
pub enum Event {
    /// Content inside a specific XML tag
    Tagged { tag: Tag, content: String },
    /// Content outside any registered XML tags  
    Content(String),
}

impl Event {
    /// Create a new tagged event (for internal use)
    fn tagged_internal(tag: Tag, content: impl Into<String>) -> Self {
        Self::Tagged {
            tag,
            content: content.into(),
        }
    }

    /// Create a new content event
    pub fn content(content: impl Into<String>) -> Self {
        Self::Content(content.into())
    }

    /// Get the content string regardless of event type
    pub fn get_content(&self) -> &str {
        match self {
            Event::Tagged { content, .. } => content,
            Event::Content(content) => content,
        }
    }

    /// Get the tag name if this is a tagged event
    pub fn tag(&self) -> Option<&str> {
        match self {
            Event::Tagged { tag, .. } => Some(tag.name()),
            Event::Content(_) => None,
        }
    }

    /// Get the internal tag handle if needed
    pub fn tag_handle(&self) -> Option<&Tag> {
        match self {
            Event::Tagged { tag, .. } => Some(tag),
            Event::Content(_) => None,
        }
    }
}

/// Builder for creating an XmlParser with specific tags to watch for
#[derive(Debug)]
pub struct XmlParserBuilder {
    tags: Vec<String>,
    next_id: usize,
}

impl Default for XmlParserBuilder {
    fn default() -> Self {
        Self {
            tags: Vec::new(),
            next_id: 0,
        }
    }
}

impl XmlParserBuilder {
    /// Create a new XmlParserBuilder
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a tag to watch for during parsing and return a handle to it
    pub fn register_tag(&mut self, tag: impl Into<String>) -> Tag {
        let name = tag.into();
        let tag_handle = Tag {
            name: name.clone(),
            id: self.next_id,
        };
        self.next_id += 1;
        self.tags.push(name);
        tag_handle
    }

    /// Build the XmlParser
    pub fn build(self) -> XmlParser {
        let mut tag_map = HashMap::new();
        let mut tags_set = HashSet::new();
        
        for (id, name) in self.tags.into_iter().enumerate() {
            tags_set.insert(name.clone());
            tag_map.insert(name.clone(), Tag { name, id });
        }
        
        XmlParser::new(tags_set, tag_map)
    }
}

/// Parser state for tracking XML tag parsing
#[derive(Debug, Clone)]
struct ParserState {
    /// Stack of currently open tags (tag name and accumulated content)
    open_tags: Vec<(String, String)>,
    /// Buffer for content outside any registered tags
    content_buffer: String,
    /// Buffer for partial tag parsing
    tag_buffer: String,
    /// Whether we're currently inside a tag
    in_tag: bool,
}

impl Default for ParserState {
    fn default() -> Self {
        Self {
            open_tags: Vec::new(),
            content_buffer: String::new(),
            tag_buffer: String::new(),
            in_tag: false,
        }
    }
}

/// XML parser that can process streaming or complete text and emit events
#[derive(Debug, Clone)]
pub struct XmlParser {
    /// Set of tags this parser is configured to watch for
    registered_tags: HashSet<String>,
    /// Map from tag names to Tag handles
    tag_map: HashMap<String, Tag>,
    /// Current parser state (wrapped for interior mutability)
    state: Arc<Mutex<ParserState>>,
}

impl XmlParser {
    /// Create a new XmlParser with the specified tags to watch for
    pub fn new(tags: HashSet<String>, tag_map: HashMap<String, Tag>) -> Self {
        Self {
            registered_tags: tags,
            tag_map,
            state: Arc::new(Mutex::new(ParserState::default())),
        }
    }

    /// Reset the parser state (useful for new generations)
    pub fn reset(&self) {
        *self.state.lock().unwrap() = ParserState::default();
    }

    /// Parse a complete text and return all events
    pub fn parse_complete(&self, text: &str) -> Vec<Event> {
        self.reset();
        let mut events = Vec::new();
        
        for char in text.chars() {
            if let Some(event) = self.process_char(char) {
                events.push(event);
            }
        }
        
        // Flush any remaining content
        events.extend(self.flush());
        events
    }

    /// Parse a streaming token and return any events that are ready
    pub fn parse_token(&self, token: &str) -> Vec<Event> {
        let mut events = Vec::new();
        
        for char in token.chars() {
            if let Some(event) = self.process_char(char) {
                events.push(event);
            }
        }
        
        events
    }

    /// Process a single character and return an event if one is ready
    fn process_char(&self, c: char) -> Option<Event> {
        let mut state = self.state.lock().unwrap();
        
        match c {
            '<' => {
                // Start of a potential tag
                state.in_tag = true;
                state.tag_buffer.clear();
                state.tag_buffer.push(c);
                None
            }
            '>' if state.in_tag => {
                // End of tag
                state.tag_buffer.push(c);
                state.in_tag = false;
                
                let tag_content = state.tag_buffer.clone();
                state.tag_buffer.clear();
                
                self.handle_tag(&mut state, &tag_content)
            }
            _ if state.in_tag => {
                // Inside a tag, accumulate
                state.tag_buffer.push(c);
                None
            }
            _ => {
                // Regular content character
                // Check if we're inside any registered tag
                if let Some((_, ref mut content)) = state.open_tags.last_mut() {
                    content.push(c);
                } else {
                    state.content_buffer.push(c);
                }
                None
            }
        }
    }

    /// Handle a complete tag and return an event if one is ready
    fn handle_tag(&self, state: &mut ParserState, tag_content: &str) -> Option<Event> {
        if let Some(tag_name) = self.parse_tag_name(tag_content) {
            if self.registered_tags.contains(&tag_name) {
                if tag_content.starts_with("</") {
                    // Closing tag - find matching opening tag
                    if let Some(pos) = state.open_tags.iter().rposition(|(name, _)| name == &tag_name) {
                        let (_, content) = state.open_tags.remove(pos);
                        
                        // Get the Tag handle from our map
                        if let Some(tag_handle) = self.tag_map.get(&tag_name) {
                            return Some(Event::tagged_internal(tag_handle.clone(), content));
                        }
                    }
                } else if !tag_content.ends_with("/>") {
                    // Opening tag (not self-closing)
                    // First emit any pending content if we're at top level
                    if state.open_tags.is_empty() && !state.content_buffer.is_empty() {
                        let content = state.content_buffer.clone();
                        state.content_buffer.clear();
                        // We need to push the tag AFTER emitting content
                        state.open_tags.push((tag_name, String::new()));
                        return Some(Event::content(content));
                    }
                    
                    state.open_tags.push((tag_name, String::new()));
                }
            } else {
                // Tag not registered, treat as content
                if state.open_tags.is_empty() {
                    state.content_buffer.push_str(tag_content);
                } else {
                    // We're inside a registered tag, add to its content
                    if let Some((_, ref mut content)) = state.open_tags.last_mut() {
                        content.push_str(tag_content);
                    }
                }
            }
        } else {
            // Invalid tag, treat as content
            if state.open_tags.is_empty() {
                state.content_buffer.push_str(tag_content);
            } else {
                // We're inside a registered tag, add to its content
                if let Some((_, ref mut content)) = state.open_tags.last_mut() {
                    content.push_str(tag_content);
                }
            }
        }

        None
    }

    /// Extract tag name from tag content (e.g., "<think>" -> "think", "</think>" -> "think")
    fn parse_tag_name(&self, tag_content: &str) -> Option<String> {
        if tag_content.len() < 3 || !tag_content.starts_with('<') || !tag_content.ends_with('>') {
            return None;
        }

        let inner = &tag_content[1..tag_content.len() - 1];
        
        if inner.starts_with('/') {
            // Closing tag
            let name = &inner[1..];
            Some(name.split_whitespace().next()?.to_string())
        } else {
            // Opening tag or self-closing tag
            let name = inner.split_whitespace().next()?;
            if name.ends_with('/') {
                Some(name[..name.len() - 1].to_string())
            } else {
                Some(name.to_string())
            }
        }
    }

    /// Flush any remaining content and return events
    pub fn flush(&self) -> Vec<Event> {
        let mut state = self.state.lock().unwrap();
        let mut events = Vec::new();

        // Emit any remaining content
        if !state.content_buffer.is_empty() {
            events.push(Event::content(state.content_buffer.clone()));
            state.content_buffer.clear();
        }

        // Emit any unclosed tags as content (malformed XML)
        for (tag_name, content) in state.open_tags.drain(..) {
            if !content.is_empty() {
                if let Some(tag_handle) = self.tag_map.get(&tag_name) {
                    events.push(Event::tagged_internal(tag_handle.clone(), content));
                }
            }
        }

        events
    }

    /// Get the currently registered tags
    pub fn registered_tags(&self) -> &HashSet<String> {
        &self.registered_tags
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_basic_parsing() {
        let mut builder = XmlParserBuilder::new();
        let think_tag = builder.register_tag("think");
        let parser = builder.build();

        let text = "<think>Hello world</think>Regular content";
        let events = parser.parse_complete(text);

        assert_eq!(events.len(), 2);
        match &events[0] {
            Event::Tagged { tag, content } => {
                assert_eq!(tag, &think_tag);
                assert_eq!(content, "Hello world");
            }
            _ => panic!("Expected tagged event"),
        }
        assert_eq!(events[1], Event::content("Regular content"));
    }

    #[test]
    fn test_streaming_parsing() {
        let mut builder = XmlParserBuilder::new();
        let think_tag = builder.register_tag("think");
        let parser = builder.build();

        let tokens = vec!["<think>", "Hello", " ", "world", "</think>", "Regular"];
        let mut all_events = Vec::new();

        for token in tokens {
            let events = parser.parse_token(token);
            all_events.extend(events);
        }
        all_events.extend(parser.flush());

        assert_eq!(all_events.len(), 2);
        match &all_events[0] {
            Event::Tagged { tag, content } => {
                assert_eq!(tag, &think_tag);
                assert_eq!(content, "Hello world");
            }
            _ => panic!("Expected tagged event"),
        }
        assert_eq!(all_events[1], Event::content("Regular"));
    }

    #[test]
    fn test_multiple_tags() {
        let mut builder = XmlParserBuilder::new();
        let think_tag = builder.register_tag("think");
        let tool_response_tag = builder.register_tag("tool_response");
        let parser = builder.build();

        let text = "<think>Thinking</think>Content<tool_response>Response</tool_response>";
        let events = parser.parse_complete(text);

        assert_eq!(events.len(), 3);
        match &events[0] {
            Event::Tagged { tag, content } => {
                assert_eq!(tag, &think_tag);
                assert_eq!(content, "Thinking");
            }
            _ => panic!("Expected tagged event"),
        }
        assert_eq!(events[1], Event::content("Content"));
        match &events[2] {
            Event::Tagged { tag, content } => {
                assert_eq!(tag, &tool_response_tag);
                assert_eq!(content, "Response");
            }
            _ => panic!("Expected tagged event"),
        }
    }

    #[test]
    fn test_unregistered_tags() {
        let mut builder = XmlParserBuilder::new();
        let think_tag = builder.register_tag("think");
        let parser = builder.build();

        let text = "<think>Registered</think><other>Not registered</other>";
        let events = parser.parse_complete(text);

        assert_eq!(events.len(), 2);
        match &events[0] {
            Event::Tagged { tag, content } => {
                assert_eq!(tag, &think_tag);
                assert_eq!(content, "Registered");
            }
            _ => panic!("Expected tagged event"),
        }
        assert_eq!(events[1], Event::content("<other>Not registered</other>"));
    }

    #[test]
    fn test_malformed_xml() {
        let mut builder = XmlParserBuilder::new();
        let think_tag = builder.register_tag("think");
        let parser = builder.build();

        let text = "<think>Unclosed tag content";
        let events = parser.parse_complete(text);

        assert_eq!(events.len(), 1);
        match &events[0] {
            Event::Tagged { tag, content } => {
                assert_eq!(tag, &think_tag);
                assert_eq!(content, "Unclosed tag content");
            }
            _ => panic!("Expected tagged event"),
        }
    }
}