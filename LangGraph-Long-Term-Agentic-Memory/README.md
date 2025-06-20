# Long-Term Agentic Memory

Building AI agents that remember and adapt to user preferences over time using LangGraph.

## AI Agent Personalization

![Agent Personalization](images/1.png)

Agents need memory to:
- Learn what to ignore from user feedback
- Remember meeting preferences and communication style
- Maintain conversation history and context

## Memory Types

![Memory Types](images/2.png)

| Type | Stores | Example |
|------|--------|---------|
| **Semantic** | Facts | User preferences, skills |
| **Episodic** | Experiences | Past conversations, actions |
| **Procedural** | Instructions | System prompts, behaviors |

### Examples

**Semantic Memory:**
- "John prefers 9 AM meetings"
- "Sarah is a Python developer"
- "Team uses Slack for communication"

**Episodic Memory:**
- "Discussed project deadline on Monday"
- "User canceled meeting last week"
- "Helped debug API issue yesterday"

**Procedural Memory:**
- "Always ask for confirmation before booking meetings"
- "Format code blocks with syntax highlighting"
- "Send follow-up emails within 24 hours"

## Memory Storage Approaches

![Memory Storage Mechanisms](images/3.png)

**Hot Path**: Memory updates happen during conversation (slower but immediate)
**Background**: Agent responds first, updates memory later (faster response)

# SUMMARY OF MEMORY TYPES DEMONSTRATED IN THIS PROJECT:

1. SEMANTIC MEMORY: Facts and general knowledge
   - Stores factual information about email handling
   - Vector embeddings for finding similar content

2. EPISODIC MEMORY: Examples and experiences
   - Specific email conversations and interactions
   - Few-shot examples from past situations
   - Learning from what worked and what didn't

3. PROCEDURAL MEMORY: Instructions and prompts
   - Behavioral rules that adapt over time
   - Prompt updates based on feedback
   - Learning how to perform tasks better

The system gets smarter with each interaction, learning from feedback
and improving its email handling capabilities continuously!