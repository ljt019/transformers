#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReasoningSupport {
    None,
    AlwaysOn,
    Toggleable,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelCapabilities {
    pub reasoning: ReasoningSupport,
    pub tool_calling: bool,
    pub streaming: bool,
}

impl ModelCapabilities {
    pub fn basic() -> Self {
        Self {
            reasoning: ReasoningSupport::None,
            tool_calling: false,
            streaming: false,
        }
    }

    pub fn with_reasoning(reasoning: ReasoningSupport) -> Self {
        Self {
            reasoning,
            tool_calling: false,
            streaming: false,
        }
    }

    pub fn with_tools() -> Self {
        Self {
            reasoning: ReasoningSupport::None,
            tool_calling: true,
            streaming: false,
        }
    }

    pub fn with_reasoning_and_tools(reasoning: ReasoningSupport) -> Self {
        Self {
            reasoning,
            tool_calling: true,
            streaming: false,
        }
    }
}
