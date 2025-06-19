# EMAIL ASSISTANT WITH SEMANTIC AND EPISODIC MEMORY
# =================================================
# This is an AI email assistant that can:
# 1. Triage incoming emails (ignore/notify/respond)
# 2. Learn from examples using semantic similarity search
# 3. Remember past interactions and context
# 4. Take actions like writing emails, scheduling meetings, etc.

import os
import uuid
from dotenv import load_dotenv
from langgraph.store.memory import InMemoryStore
from pydantic import BaseModel, Field
from typing_extensions import TypedDict, Literal, Annotated
from langgraph.graph import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from typing import Literal
from langchain.chat_models import init_chat_model
from IPython.display import Image, display
from prompts import triage_user_prompt, triage_system_prompt, agent_system_prompt_memory
from langchain_core.tools import tool
from langmem import create_manage_memory_tool, create_search_memory_tool
from langgraph.prebuilt import create_react_agent

_ = load_dotenv()

# USER PROFILE CONFIGURATION
# ==========================
# Define the user this email assistant is working for
profile = {
    "name": "John",
    "full_name": "John Doe",
    "user_profile_background": "Senior software engineer leading a team of 5 developers",
}

# TRIAGE RULES AND INSTRUCTIONS
# =============================
# Define how emails should be categorized and what actions to take
prompt_instructions = {
    "triage_rules": {
        "ignore": "Marketing newsletters, spam emails, mass company announcements",
        "notify": "Team member out sick, build system notifications, project status updates",
        "respond": "Direct questions from team members, meeting requests, critical bug reports",
    },
    "agent_instructions": "Use these tools when appropriate to help manage John's tasks efficiently."
}

# SAMPLE EMAIL DATA
# =================
# Example email structure - this shows what format emails will be in
email = {
    "from": "Alice Smith <alice.smith@company.com>",
    "to": "John Doe <john.doe@company.com>",
    "subject": "Quick question about API documentation",
    "body": """
Hi John,

I was reviewing the API documentation for the new authentication service and noticed a few endpoints seem to be missing from the specs. Could you help clarify if this was intentional or if we should update the docs?

Specifically, I'm looking at:
- /auth/refresh
- /auth/validate

Thanks!
Alice""",
}

# MEMORY STORE SETUP
# ==================
# InMemoryStore with OpenAI embeddings for semantic similarity search
# This allows the agent to find similar past emails and learn from them
store = InMemoryStore(
    index={"embed": "openai:text-embedding-3-small"}
)

# TRAINING EXAMPLES FOR THE AGENT
# ================================
# We add example emails with their correct triage labels so the agent can learn

# Example 1: Email that needs a response (technical question from team member)
email = {
    "author": "Alice Smith <alice.smith@company.com>",
    "to": "John Doe <john.doe@company.com>",
    "subject": "Quick question about API documentation",
    "email_thread": """Hi John,

I was reviewing the API documentation for the new authentication service and noticed a few endpoints seem to be missing from the specs. Could you help clarify if this was intentional or if we should update the docs?

Specifically, I'm looking at:
- /auth/refresh
- /auth/validate

Thanks!
Alice""",
}

data = {
    "email": email,
    # This label teaches the agent that emails like this should get a "respond" classification
    "label": "respond"
}

# Store this example in the "lance" user's examples namespace
store.put(
    ("email_assistant", "lance", "examples"), 
    str(uuid.uuid4()), 
    data
)

# Example 2: Email that should be ignored (just informational update)
data = {
    "email": {
        "author": "Sarah Chen <sarah.chen@company.com>",
        "to": "John Doe <john.doe@company.com>",
        "subject": "Update: Backend API Changes Deployed to Staging",
        "email_thread": """Hi John,
    
    Just wanted to let you know that I've deployed the new authentication endpoints we discussed to the staging environment. Key changes include:
    
    - Implemented JWT refresh token rotation
    - Added rate limiting for login attempts
    - Updated API documentation with new endpoints
    
    All tests are passing and the changes are ready for review. You can test it out at staging-api.company.com/auth/*
    
    No immediate action needed from your side - just keeping you in the loop since this affects the systems you're working on.
    
    Best regards,
    Sarah
    """,
    },
    # This label teaches the agent that informational updates should be "ignored"
    "label": "ignore"
}

store.put(
    ("email_assistant", "lance", "examples"),
    str(uuid.uuid4()),
    data
)

# HELPER FUNCTIONS
# ================

# Template for formatting examples to show the agent
template = """Email Subject: {subject}
Email From: {from_email}
Email To: {to_email}
Email Content: 
```
{content}
```
> Triage Result: {result}"""

# Format retrieved examples into a readable format for the prompt
def format_few_shot_examples(examples):
    """Convert stored examples into formatted text for the LLM prompt"""
    strs = ["Here are some previous examples:"]
    for eg in examples:
        strs.append(
            template.format(
                subject=eg.value["email"]["subject"],
                to_email=eg.value["email"]["to"],
                from_email=eg.value["email"]["author"],
                content=eg.value["email"]["email_thread"][:400],  # Truncate long emails
                result=eg.value["label"],
            )
        )
    return "\n\n------------\n\n".join(strs)

# DEMONSTRATION OF SEMANTIC SEARCH
# =================================
# Show how the system finds similar emails from memory
email_data = {
        "author": "Sarah Chen <sarah.chen@company.com>",
        "to": "John Doe <john.doe@company.com>",
        "subject": "Update: Backend API Changes Deployed to Staging",
        "email_thread": """Hi John,
    
    Wanted to let you know that I've deployed the new authentication endpoints we discussed to the staging environment. Key changes include:
    
    - Implemented JWT refresh token rotation
    - Added rate limiting for login attempts
    - Updated API documentation with new endpoints
    
    All tests are passing and the changes are ready for review. You can test it out at staging-api.company.com/auth/*
    
    No immediate action needed from your side - just keeping you in the loop since this affects the systems you're working on.
    
    Best regards,
    Sarah
    """,
    }

# Search for similar emails in the store and print the results
results = store.search(
    ("email_assistant", "lance", "examples"),
    query=str({"email": email_data}),
    limit=1)
print(format_few_shot_examples(results))
# Here are some previous examples:
# ------------
# Email Subject: Update: Backend API Changes Deployed to Staging
# Email From: Sarah Chen <sarah.chen@company.com>
# Email To: John Doe <john.doe@company.com>
# Email Content: 
# ```
# Hi John,
    
#     Just wanted to let you know that I've deployed the new authentication endpoints we discussed to the staging environment. Key changes include:
    
#     - Implemented JWT refresh token rotation
#     - Added rate limiting for login attempts
#     - Updated API documentation with new endpoints
    
#     All tests are passing and the changes are ready for review. You can test it out at st
# ```
# > Triage Result: ignore

llm = init_chat_model("openai:gpt-4o-mini")

class Router(BaseModel):
    """Analyze the unread email and route it according to its content."""

    reasoning: str = Field(
        description="Step-by-step reasoning behind the classification."
    )
    classification: Literal["ignore", "respond", "notify"] = Field(
        description="The classification of an email: 'ignore' for irrelevant emails, "
        "'notify' for important information that doesn't need a response, "
        "'respond' for emails that need a reply",
    )

# Create LLM that outputs structured Router objects
llm_router = llm.with_structured_output(Router)

# AGENT TOOLS
# ===========
# Define the actions the agent can take when responding to emails

@tool
def write_email(to: str, subject: str, content: str) -> str:
    """Write and send an email."""
    # Placeholder response - in real app would send email
    return f"Email sent to {to} with subject '{subject}'"

@tool
def schedule_meeting(
    attendees: list[str], 
    subject: str, 
    duration_minutes: int, 
    preferred_day: str
) -> str:
    """Schedule a calendar meeting."""
    # Placeholder response - in real app would check calendar and schedule
    return f"Meeting '{subject}' scheduled for {preferred_day} with {len(attendees)} attendees"

@tool
def check_calendar_availability(day: str) -> str:
    """Check calendar availability for a given day."""
    # Placeholder response - in real app would check actual calendar
    return f"Available times on {day}: 9:00 AM, 2:00 PM, 4:00 PM"

class State(TypedDict):
    email_input: dict
    messages: Annotated[list, add_messages]

# MAIN TRIAGE LOGIC
# ==================
def triage_router(state: State, config, store) -> Command[
    Literal["response_agent", "__end__"]
]:
    """
    Main email triage function that:
    1. Extracts email details from state
    2. Searches for similar past examples
    3. Uses LLM to classify the email
    4. Routes to appropriate next step
    """
    # Extract email components
    author = state['email_input']['author']
    to = state['email_input']['to']
    subject = state['email_input']['subject']
    email_thread = state['email_input']['email_thread']

    # Build namespace for this user's examples
    namespace = (
        "email_assistant",
        config['configurable']['langgraph_user_id'],
        "examples"
    )
    
    # Search for similar emails in memory to use as examples
    examples = store.search(
        namespace, 
        query=str({"email": state['email_input']})
    ) 
    examples=format_few_shot_examples(examples)
    
    # Build the system prompt with user profile and examples
    system_prompt = triage_system_prompt.format(
        full_name=profile["full_name"],
        name=profile["name"],
        user_profile_background=profile["user_profile_background"],
        triage_no=prompt_instructions["triage_rules"]["ignore"],
        triage_notify=prompt_instructions["triage_rules"]["notify"],
        triage_email=prompt_instructions["triage_rules"]["respond"],
        examples=examples
    )
    
    # Build the user prompt with the current email
    user_prompt = triage_user_prompt.format(
        author=author, 
        to=to, 
        subject=subject, 
        email_thread=email_thread
    )
    
    # Get classification from LLM
    result = llm_router.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )
    
    # Route based on classification
    if result.classification == "respond":
        print("📧 Classification: RESPOND - This email requires a response")
        goto = "response_agent"
        update = {
            "messages": [
                {
                    "role": "user",
                    "content": f"Respond to the email {state['email_input']}",
                }
            ]
        }
    elif result.classification == "ignore":
        print("🚫 Classification: IGNORE - This email can be safely ignored")
        update = None
        goto = END
    elif result.classification == "notify":
        # In real life, this would send a notification somewhere
        print("🔔 Classification: NOTIFY - This email contains important information")
        update = None
        goto = END
    else:
        raise ValueError(f"Invalid classification: {result.classification}")
    
    return Command(goto=goto, update=update)

# MEMORY MANAGEMENT TOOLS
# =======================
# Create tools for managing long-term memory (storing and retrieving information)
manage_memory_tool = create_manage_memory_tool(
    namespace=(
        "email_assistant", 
        "{langgraph_user_id}",
        "collection"
    )
)
search_memory_tool = create_search_memory_tool(
    namespace=(
        "email_assistant",
        "{langgraph_user_id}",
        "collection"
    )
)

# RESPONSE AGENT SETUP
# ====================
def create_prompt(state):
    """Create the prompt for the response agent with system instructions and conversation history"""
    return [
        {
            "role": "system", 
            "content": agent_system_prompt_memory.format(
                instructions=prompt_instructions["agent_instructions"], 
                **profile
            )
        }
    ] + state['messages']

# All tools available to the response agent
tools= [
    write_email, 
    schedule_meeting,
    check_calendar_availability,
    manage_memory_tool,  # For storing information
    search_memory_tool   # For retrieving past information
]

# Create the response agent using LangGraph's ReAct pattern
response_agent = create_react_agent(
    "openai:gpt-4o",  # More powerful model for responses
    tools=tools,
    prompt=create_prompt,
    store=store  # Pass the store for memory access
)

# WORKFLOW GRAPH CONSTRUCTION
# ===========================
# Build the overall email processing workflow
config = {"configurable": {"langgraph_user_id": "lance"}}

email_agent = StateGraph(State)
# Add the triage router node
email_agent = email_agent.add_node(triage_router)
# Add the response agent node
email_agent = email_agent.add_node("response_agent", response_agent)
# Start with triage
email_agent = email_agent.add_edge(START, "triage_router")
# Compile the graph with memory store
email_agent = email_agent.compile(store=store)

# DEMONSTRATION: LEARNING FROM EXAMPLES
# =====================================
# This section shows how the agent learns and improves over time

print("=" * 70)
print("DEMONSTRATION: How the agent learns from examples")
print("=" * 70)

# First run: Without a specific example, the agent might classify this as "respond"
email_input = {
    "author": "Tom Jones <tome.jones@bar.com>",
    "to": "John Doe <john.doe@company.com>",
    "subject": "Quick question about API documentation",
    "email_thread": """Hi John - want to buy documentation?""",
}

print("BEFORE adding example:")
response = email_agent.invoke(
    {"email_input": email_input}, 
    config={"configurable": {"langgraph_user_id": "harrison"}}
)
print(response)
# 📧 Classification: RESPOND - This email requires a response

# Now we add an example showing this type of email should be ignored (it's spam/sales)
data = {
    "email": {
    "author": "Tom Jones <tome.jones@bar.com>",
    "to": "John Doe <john.doe@company.com>",
    "subject": "Quick question about API documentation",
    "email_thread": """Hi John - want to buy documentation?""",
},
    "label": "ignore"  # Teaching the agent this should be ignored
}

# Store the corrective example
store.put(
    ("email_assistant", "harrison", "examples"),
    str(uuid.uuid4()),
    data
)

print("\nAFTER adding example:")
# Run again with the same email - now it should be classified as "ignore"
email_input = {
    "author": "Tom Jones <tome.jones@bar.com>",
    "to": "John Doe <john.doe@company.com>",
    "subject": "Quick question about API documentation",
    "email_thread": """Hi John - want to buy documentation?""",
}

response = email_agent.invoke(
    {"email_input": email_input}, 
    config={"configurable": {"langgraph_user_id": "harrison"}}
)
print(response)
# 🚫 Classification: IGNORE - This email can be safely ignored