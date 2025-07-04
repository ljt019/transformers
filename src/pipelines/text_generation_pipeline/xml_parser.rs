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

/// Parts of a tag emitted as events during parsing
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TagParts {
    /// Opening of a tag
    Start,
    /// Content inside a tag or outside any tag
    Content,
    /// Closing of a tag
    End,
}

/// An event emitted by the XML parser
#[derive(Debug, Clone, PartialEq)]
pub struct Event {
    tag: Option<Tag>,
    part: TagParts,
    content: String,
}

impl Event {
    fn new(tag: Option<Tag>, part: TagParts, content: impl Into<String>) -> Self {
        Self {
            tag,
            part,
            content: content.into(),
        }
    }

    /// Create a new content event outside any tag
    pub fn content(content: impl Into<String>) -> Self {
        Self::new(None, TagParts::Content, content)
    }

    /// Create a new start event for a tag
    pub fn start(tag: Tag) -> Self {
        Self::new(Some(tag), TagParts::Start, "")
    }

    /// Create a new end event for a tag
    pub fn end(tag: Tag) -> Self {
        Self::new(Some(tag), TagParts::End, "")
    }

    /// Create a new content event inside a tag
    fn tagged_internal(tag: Tag, content: impl Into<String>) -> Self {
        Self::new(Some(tag), TagParts::Content, content)
    }

    /// Get the content string
    pub fn get_content(&self) -> &str {
        &self.content
    }

    /// Get the tag name if present
    pub fn tag(&self) -> Option<&str> {
        self.tag.as_ref().map(|t| t.name())
    }

    /// Get the internal tag handle
    pub fn tag_handle(&self) -> Option<&Tag> {
        self.tag.as_ref()
    }

    /// Get the part of the tag this event corresponds to
    pub fn part(&self) -> TagParts {
        self.part
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
    /// Length of top-level content that has already been emitted downstream. This lets us
    /// stream only the newly arrived slice on every call to `parse_token`.
    emitted_top_len: usize,
    /// For each open tag name we keep the number of characters that have already been
    /// emitted so we can stream incremental updates without duplication.
    emitted_tag_lens: std::collections::HashMap<String, usize>,
}

impl Default for ParserState {
    fn default() -> Self {
        Self {
            open_tags: Vec::new(),
            content_buffer: String::new(),
            tag_buffer: String::new(),
            in_tag: false,
            emitted_top_len: 0,
            emitted_tag_lens: std::collections::HashMap::new(),
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
            let mut evs = self.process_char(char);
            events.append(&mut evs);
        }

        events.extend(self.flush());
        events
    }

    /// Parse a streaming token and return any events that are ready
    pub fn parse_token(&self, token: &str) -> Vec<Event> {
        let mut events = Vec::new();

        for char in token.chars() {
            let mut evs = self.process_char(char);
            events.append(&mut evs);
        }

        // Emit the newly appended slice either for top-level content or for the currently
        // innermost open tag (if any). This enables true streaming behaviour: callers get
        // incremental updates instead of the whole block at close time.
        {
            let mut state = self.state.lock().unwrap();

            if state.open_tags.is_empty() {
                // Outside of any registered tag.
                let current_len = state.content_buffer.len();
                if current_len > state.emitted_top_len {
                    let mut new_slice = &state.content_buffer[state.emitted_top_len..];

                    // If this is the very first top-level content emission, strip leading newlines.
                    if state.emitted_top_len == 0 {
                        new_slice = new_slice.trim_start_matches('\n');
                    }

                    // Skip emitting if it is now empty or just whitespace/newlines.
                    let content_to_emit = if new_slice.trim().is_empty() {
                        "".to_string()
                    } else {
                        new_slice.to_string()
                    };

                    if !content_to_emit.is_empty() {
                        events.push(Event::content(content_to_emit));
                    }
                    state.emitted_top_len = current_len;
                }
            } else {
                // Inside the innermost registered tag → stream its delta.
                if let Some((tag_name_ref, content_ref)) = state.open_tags.last() {
                    let tag_name = tag_name_ref.clone();
                    let content = content_ref.clone();
                    let total_len = content.len();

                    let already_emitted = *state.emitted_tag_lens.get(&tag_name).unwrap_or(&0);

                    if total_len > already_emitted {
                        let new_slice = &content[already_emitted..];

                        // Strip leading newlines from the first emission of tag content
                        if already_emitted == 0 {
                            let trimmed = new_slice.trim_start_matches('\n');
                            if !trimmed.is_empty() {
                                if let Some(tag_handle) = self.tag_map.get(&tag_name) {
                                    events
                                        .push(Event::tagged_internal(tag_handle.clone(), trimmed));
                                }
                            }
                            // Update emitted length to account for any trimmed newlines
                            state.emitted_tag_lens.insert(tag_name.clone(), total_len);
                        } else {
                            // Not the first emission, emit as-is
                            if let Some(tag_handle) = self.tag_map.get(&tag_name) {
                                events.push(Event::tagged_internal(tag_handle.clone(), new_slice));
                            }
                            state.emitted_tag_lens.insert(tag_name.clone(), total_len);
                        }
                    }
                }
            }
        }

        events
    }

    /// Process a single character and return an event if one is ready
    fn process_char(&self, c: char) -> Vec<Event> {
        let mut events = Vec::new();
        let mut state = self.state.lock().unwrap();

        match c {
            '<' => {
                state.in_tag = true;
                state.tag_buffer.clear();
                state.tag_buffer.push(c);
            }
            '>' if state.in_tag => {
                state.tag_buffer.push(c);
                state.in_tag = false;

                let tag_content = state.tag_buffer.clone();
                state.tag_buffer.clear();

                events.extend(self.handle_tag(&mut state, &tag_content));
            }
            _ if state.in_tag => {
                state.tag_buffer.push(c);
            }
            _ => {
                if let Some((_, ref mut content)) = state.open_tags.last_mut() {
                    content.push(c);
                } else {
                    state.content_buffer.push(c);
                }
            }
        }

        events
    }

    /// Handle a complete tag and return an event if one is ready
    fn handle_tag(&self, state: &mut ParserState, tag_content: &str) -> Vec<Event> {
        let mut events = Vec::new();

        if let Some(tag_name) = self.parse_tag_name(tag_content) {
            if self.registered_tags.contains(&tag_name) {
                if tag_content.starts_with("</") {
                    if let Some(pos) = state
                        .open_tags
                        .iter()
                        .rposition(|(name, _)| name == &tag_name)
                    {
                        let (_, content) = state.open_tags.remove(pos);

                        let already_emitted = state.emitted_tag_lens.remove(&tag_name).unwrap_or(0);

                        if let Some(tag_handle) = self.tag_map.get(&tag_name) {
                            if content.len() > already_emitted {
                                let remaining_content = &content[already_emitted..];
                                // Strip leading newlines if this is the first content emission
                                let mut content_to_emit = if already_emitted == 0 {
                                    remaining_content.trim_start_matches('\n')
                                } else {
                                    remaining_content
                                };

                                // Also strip trailing newlines from the final emission
                                let trimmed = content_to_emit.trim_end_matches('\n');
                                if !trimmed.is_empty() {
                                    let mut final_str = trimmed.to_string();
                                    final_str.push('\n'); // ensure exactly one trailing newline
                                    events.push(Event::tagged_internal(
                                        tag_handle.clone(),
                                        final_str,
                                    ));
                                }
                            }
                            events.push(Event::end(tag_handle.clone()));
                        }
                    }
                } else if !tag_content.ends_with("/>") {
                    if state.open_tags.is_empty() && !state.content_buffer.is_empty() {
                        let content = &state.content_buffer[state.emitted_top_len..];

                        // Normalize whitespace before tag start
                        let mut slice = content;
                        if state.emitted_top_len == 0 {
                            slice = slice.trim_start_matches('\n');
                        }
                        let content_to_emit = if slice.trim().is_empty() {
                            String::new()
                        } else {
                            slice.to_string()
                        };

                        state.emitted_top_len = state.content_buffer.len();
                        if !content_to_emit.is_empty() {
                            events.push(Event::content(content_to_emit));
                        }
                    }

                    state.open_tags.push((tag_name.clone(), String::new()));

                    if let Some(tag_handle) = self.tag_map.get(&tag_name) {
                        events.push(Event::start(tag_handle.clone()));
                    }
                }
            } else {
                if state.open_tags.is_empty() {
                    state.content_buffer.push_str(tag_content);
                } else if let Some((_, ref mut content)) = state.open_tags.last_mut() {
                    content.push_str(tag_content);
                }
            }
        } else {
            if state.open_tags.is_empty() {
                state.content_buffer.push_str(tag_content);
            } else if let Some((_, ref mut content)) = state.open_tags.last_mut() {
                content.push_str(tag_content);
            }
        }

        events
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
        if state.content_buffer.len() > state.emitted_top_len {
            let remaining = &state.content_buffer[state.emitted_top_len..];

            // Always trim leading and trailing newlines from final content
            let slice = remaining.trim_start_matches('\n').trim_end_matches('\n');

            let content_to_emit = if slice.trim().is_empty() {
                String::new()
            } else {
                slice.to_string()
            };

            if !content_to_emit.is_empty() {
                events.push(Event::content(content_to_emit));
            }
        }
        state.content_buffer.clear();
        state.emitted_top_len = 0;

        // Emit any unclosed tags as content (malformed XML). We must take ownership of the
        // drained tags first to avoid double-borrowing `state`.
        let drained: Vec<_> = state.open_tags.drain(..).collect();
        for (tag_name, content) in drained {
            let already_emitted = state.emitted_tag_lens.remove(&tag_name).unwrap_or(0);

            if let Some(tag_handle) = self.tag_map.get(&tag_name) {
                if content.len() > already_emitted {
                    let remaining_content = &content[already_emitted..];
                    // Strip leading newlines if this is the first content emission
                    let mut content_to_emit = if already_emitted == 0 {
                        remaining_content.trim_start_matches('\n')
                    } else {
                        remaining_content
                    };

                    // Also strip trailing newlines from the final emission
                    let trimmed = content_to_emit.trim_end_matches('\n');
                    if !trimmed.is_empty() {
                        let mut final_str = trimmed.to_string();
                        final_str.push('\n'); // ensure exactly one trailing newline
                        events.push(Event::tagged_internal(tag_handle.clone(), final_str));
                    }
                }
                events.push(Event::end(tag_handle.clone()));
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

        assert_eq!(events.len(), 4);
        assert_eq!(events[0].part(), TagParts::Start);
        assert_eq!(events[0].tag_handle(), Some(&think_tag));
        assert_eq!(events[1].part(), TagParts::Content);
        assert_eq!(events[1].get_content(), "Hello world");
        assert_eq!(events[2].part(), TagParts::End);
        assert_eq!(events[2].tag_handle(), Some(&think_tag));
        assert_eq!(events[3], Event::content("Regular content"));
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

        assert_eq!(all_events.len(), 6);
        assert_eq!(all_events[0].part(), TagParts::Start);
        assert_eq!(all_events[0].tag_handle(), Some(&think_tag));
        assert_eq!(all_events[1].part(), TagParts::Content);
        assert_eq!(all_events[1].get_content(), "Hello");
        assert_eq!(all_events[2].part(), TagParts::Content);
        assert_eq!(all_events[2].get_content(), " ");
        assert_eq!(all_events[3].part(), TagParts::Content);
        assert_eq!(all_events[3].get_content(), "world");
        assert_eq!(all_events[4].part(), TagParts::End);
        assert_eq!(all_events[4].tag_handle(), Some(&think_tag));
        assert_eq!(all_events[5], Event::content("Regular"));
    }

    #[test]
    fn test_multiple_tags() {
        let mut builder = XmlParserBuilder::new();
        let think_tag = builder.register_tag("think");
        let tool_response_tag = builder.register_tag("tool_response");
        let parser = builder.build();

        let text = "<think>Thinking</think>Content<tool_response>Response</tool_response>";
        let events = parser.parse_complete(text);

        assert_eq!(events.len(), 7);
        assert_eq!(events[0].tag_handle(), Some(&think_tag));
        assert_eq!(events[0].part(), TagParts::Start);
        assert_eq!(events[1].part(), TagParts::Content);
        assert_eq!(events[1].get_content(), "Thinking");
        assert_eq!(events[2].part(), TagParts::End);
        assert_eq!(events[2].tag_handle(), Some(&think_tag));
        assert_eq!(events[3], Event::content("Content"));
        assert_eq!(events[4].tag_handle(), Some(&tool_response_tag));
        assert_eq!(events[4].part(), TagParts::Start);
        assert_eq!(events[5].part(), TagParts::Content);
        assert_eq!(events[5].get_content(), "Response");
        assert_eq!(events[6].part(), TagParts::End);
        assert_eq!(events[6].tag_handle(), Some(&tool_response_tag));
    }

    #[test]
    fn test_unregistered_tags() {
        let mut builder = XmlParserBuilder::new();
        let think_tag = builder.register_tag("think");
        let parser = builder.build();

        let text = "<think>Registered</think><other>Not registered</other>";
        let events = parser.parse_complete(text);

        assert_eq!(events.len(), 4);
        assert_eq!(events[0].part(), TagParts::Start);
        assert_eq!(events[0].tag_handle(), Some(&think_tag));
        assert_eq!(events[1].part(), TagParts::Content);
        assert_eq!(events[1].get_content(), "Registered");
        assert_eq!(events[2].part(), TagParts::End);
        assert_eq!(events[2].tag_handle(), Some(&think_tag));
        assert_eq!(events[3], Event::content("<other>Not registered</other>"));
    }

    #[test]
    fn test_malformed_xml() {
        let mut builder = XmlParserBuilder::new();
        let think_tag = builder.register_tag("think");
        let parser = builder.build();

        let text = "<think>Unclosed tag content";
        let events = parser.parse_complete(text);

        assert_eq!(events.len(), 3);
        assert_eq!(events[0].part(), TagParts::Start);
        assert_eq!(events[0].tag_handle(), Some(&think_tag));
        assert_eq!(events[1].part(), TagParts::Content);
        assert_eq!(events[1].get_content(), "Unclosed tag content");
        assert_eq!(events[2].part(), TagParts::End);
        assert_eq!(events[2].tag_handle(), Some(&think_tag));
    }
}
