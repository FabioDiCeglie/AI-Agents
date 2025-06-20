import json
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
from langmem import create_multi_prompt_optimizer

_ = load_dotenv()

# ===============================================================================
# EMAIL ASSISTANT WITH THREE TYPES OF LONG-TERM MEMORY
# ===============================================================================
# This project demonstrates an intelligent email assistant that uses:
# 
# 1. SEMANTIC MEMORY: Stores general knowledge and facts about email handling
#    - Stores email examples with their correct classifications
#    - Uses vector embeddings to find similar past emails
#
# 2. EPISODIC MEMORY: Remembers specific events and experiences
#    - Stores individual email conversations and responses
#    - Remembers what worked and what didn't in past interactions
#    - Tracks feedback and corrections from the user
#    - Few-shot examples based on similarity
#    - Learns patterns from previous email interactions
#
# 3. PROCEDURAL MEMORY: Learns and adapts behavioral patterns
#    - Automatically updates prompts based on user feedback
#    - Learns new rules for email triage (ignore/notify/respond)
#    - Adapts response style based on corrections
# ===============================================================================

# USER PROFILE CONFIGURATION
# ==========================
# Define the user this email assistant is working for
# This acts as the agent's "identity" and context for all interactions
profile = {
    "name": "John",
    "full_name": "John Doe",
    "user_profile_background": "Senior software engineer leading a team of 5 developers",
}

# TRIAGE RULES AND INSTRUCTIONS
# =============================
# Define how emails should be categorized and what actions to take
# These are the INITIAL rules - they will be updated through procedural memory
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

# MEMORY STORE SETUP - SEMANTIC MEMORY FOUNDATION
# ================================================
# InMemoryStore with OpenAI embeddings for semantic similarity search
# This is the core of our SEMANTIC MEMORY system:
# - Stores email examples with vector embeddings
# - Allows finding similar emails based on content similarity
# - Enables learning from past email patterns and decisions
store = InMemoryStore(
    index={"embed": "openai:text-embedding-3-small"}  # Creates vector embeddings for semantic search
)

# TRAINING EXAMPLES FOR THE AGENT - SEMANTIC MEMORY EXAMPLES
# ===========================================================
# We add example emails with their correct triage labels so the agent can learn
# This demonstrates SEMANTIC MEMORY - storing knowledge about email patterns

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

# HELPER FUNCTIONS FOR SEMANTIC MEMORY
# ====================================

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
    """
    Convert stored examples into formatted text for the LLM prompt
    This function is crucial for SEMANTIC MEMORY:
    - Takes semantically similar emails found by vector search
    - Formats them as few-shot examples for the LLM
    - Helps the agent learn from similar past situations
    """
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

# Initialize the language model
llm = init_chat_model("openai:gpt-4o-mini")

class Router(BaseModel):
    """
    Structured output for email triage decisions.
    This enforces consistent classification format for better learning.
    """

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

# AGENT TOOLS - PROCEDURAL MEMORY ACTIONS
# ========================================
# Define the actions the agent can take when responding to emails
# These tools represent PROCEDURAL MEMORY - learned skills and actions

@tool
def write_email(to: str, subject: str, content: str) -> str:
    """
    Write and send an email.
    This tool gets better through PROCEDURAL MEMORY:
    - Learns email style preferences from feedback
    - Adapts tone and format based on past corrections
    """
    # Placeholder response - in real app would send email
    return f"Email sent to {to} with subject '{subject}'"

@tool
def schedule_meeting(
    attendees: list[str], 
    subject: str, 
    duration_minutes: int, 
    preferred_day: str
) -> str:
    """
    Schedule a calendar meeting.
    PROCEDURAL MEMORY aspect: Learns scheduling preferences over time
    """
    # Placeholder response - in real app would check calendar and schedule
    return f"Meeting '{subject}' scheduled for {preferred_day} with {len(attendees)} attendees"

@tool
def check_calendar_availability(day: str) -> str:
    """
    Check calendar availability for a given day.
    Can learn patterns about preferred meeting times through usage
    """
    # Placeholder response - in real app would check actual calendar
    return f"Available times on {day}: 9:00 AM, 2:00 PM, 4:00 PM"

class State(TypedDict):
    """
    Graph state that flows through the workflow.
    Contains both the input email and the conversation messages.
    """
    email_input: dict
    messages: Annotated[list, add_messages]

# MAIN TRIAGE LOGIC - INTEGRATING ALL THREE MEMORY TYPES
# =======================================================
def triage_router(state: State, config, store) -> Command[
    Literal["response_agent", "__end__"]
]:
    """
    Main triage function that demonstrates all three memory types:
    
    SEMANTIC MEMORY: Searches for similar past emails using vector similarity
    EPISODIC MEMORY: Retrieves specific past conversations and decisions
    PROCEDURAL MEMORY: Uses learned rules that adapt based on feedback
    """
    # Extract email details
    author = state['email_input']['author']
    to = state['email_input']['to']
    subject = state['email_input']['subject']
    email_thread = state['email_input']['email_thread']

    # SEMANTIC MEMORY: Find similar past emails using vector search
    namespace = (
        "email_assistant",
        config['configurable']['langgraph_user_id'],
        "examples"
    )
    examples = store.search(
        namespace, 
        query=str({"email": state['email_input']})  # Vector search for similar emails
    ) 
    examples=format_few_shot_examples(examples)  # Format for prompt inclusion

    # PROCEDURAL MEMORY: Retrieve learned triage rules (these adapt over time)
    langgraph_user_id = config['configurable']['langgraph_user_id']
    namespace = (langgraph_user_id, )

    # Get or initialize the "ignore" rule (updated through feedback)
    result = store.get(namespace, "triage_ignore")
    if result is None:
        store.put(
            namespace, 
            "triage_ignore", 
            {"prompt": prompt_instructions["triage_rules"]["ignore"]}
        )
        ignore_prompt = prompt_instructions["triage_rules"]["ignore"]
    else:
        ignore_prompt = result.value['prompt']  # Use learned rule

    # Get or initialize the "notify" rule
    result = store.get(namespace, "triage_notify")
    if result is None:
        store.put(
            namespace, 
            "triage_notify", 
            {"prompt": prompt_instructions["triage_rules"]["notify"]}
        )
        notify_prompt = prompt_instructions["triage_rules"]["notify"]
    else:
        notify_prompt = result.value['prompt']  # Use learned rule

    # Get or initialize the "respond" rule
    result = store.get(namespace, "triage_respond")
    if result is None:
        store.put(
            namespace, 
            "triage_respond", 
            {"prompt": prompt_instructions["triage_rules"]["respond"]}
        )
        respond_prompt = prompt_instructions["triage_rules"]["respond"]
    else:
        respond_prompt = result.value['prompt']  # Use learned rule
    
    # Build the system prompt with profile info, learned rules, and similar examples
    system_prompt = triage_system_prompt.format(
        full_name=profile["full_name"],
        name=profile["name"],
        user_profile_background=profile["user_profile_background"],
        triage_no=ignore_prompt,        # PROCEDURAL MEMORY
        triage_notify=notify_prompt,    # PROCEDURAL MEMORY
        triage_email=respond_prompt,    # PROCEDURAL MEMORY
        examples=examples               # EPISODIC MEMORY
    )
    
    # Create user prompt with current email details
    user_prompt = triage_user_prompt.format(
        author=author, 
        to=to, 
        subject=subject, 
        email_thread=email_thread
    )
    
    # Get classification from LLM using all memory types
    result = llm_router.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )
    
    # Route based on classification with helpful print statements for debugging
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
        # If real life, this would do something else
        print("🔔 Classification: NOTIFY - This email contains important information")
        update = None
        goto = END
    else:
        raise ValueError(f"Invalid classification: {result.classification}")
    return Command(goto=goto, update=update)

# MEMORY MANAGEMENT TOOLS - EPISODIC MEMORY SYSTEM
# =================================================
# Create tools for managing long-term memory (storing and retrieving information)
# These tools implement EPISODIC MEMORY - remembering specific events and experiences

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

# RESPONSE AGENT SETUP - INTEGRATING PROCEDURAL MEMORY
# =====================================================
def create_prompt(state, config, store):
    """
    Creates dynamic prompts that incorporate PROCEDURAL MEMORY
    The prompt adapts based on learned instructions and feedback
    """
    langgraph_user_id = config['configurable']['langgraph_user_id']
    namespace = (langgraph_user_id, )
    
    # Get learned agent instructions (these evolve through feedback)
    result = store.get(namespace, "agent_instructions")
    if result is None:
        # Initialize with default instructions
        store.put(
            namespace, 
            "agent_instructions", 
            {"prompt": prompt_instructions["agent_instructions"]}
        )
        prompt = prompt_instructions["agent_instructions"]
    else:
        prompt = result.value['prompt']  # Use learned/updated instructions
    
    return [
        {
            "role": "system", 
            "content": agent_system_prompt_memory.format(
                instructions=prompt,  # PROCEDURAL MEMORY - learned behavior
                **profile
            )
        }
    ] + state['messages']

# All tools available to the response agent
tools= [
    write_email, 
    schedule_meeting,
    check_calendar_availability,
    manage_memory_tool,  # For storing information (EPISODIC MEMORY)
    search_memory_tool   # For retrieving past information (EPISODIC MEMORY)
]

# Create the response agent using LangGraph's ReAct pattern
response_agent = create_react_agent(
    "openai:gpt-4o",  # More powerful model for responses
    tools=tools,
    prompt=create_prompt,  # Dynamic prompt with PROCEDURAL MEMORY
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

# ===============================================================================
# TESTING THE EMAIL ASSISTANT - DEMONSTRATING ALL MEMORY TYPES
# ===============================================================================

# Test email that should trigger a response
email_input = {
    "author": "Alice Jones <alice.jones@bar.com>",
    "to": "John Doe <john.doe@company.com>",
    "subject": "Quick question about API documentation",
    "email_thread": """Hi John,

Urgent issue - your service is down. Is there a reason why""",
}

config = {"configurable": {"langgraph_user_id": "lance"}}

# First invocation - agent responds normally
response = email_agent.invoke(
    {"email_input": email_input},
    config=config
)
# 📧 Classification: RESPOND - This email requires a response

# Print the conversation for debugging (keeping your useful prints!)
for m in response["messages"]:
    m.pretty_print()
# ================================ Human Message =================================
# Respond to the email {'author': 'Alice Jones <alice.jones@bar.com>', 'to': 'John Doe <john.doe@company.com>', 'subject': 'Quick question about API documentation', 'email_thread': 'Hi John,\n\nUrgent issue - your service is down. Is there a reason why'}
# ================================== Ai Message ==================================
# Tool Calls:
#   write_email (call_6cH3mDpUjSftw0OugqocfNlM)
#  Call ID: call_6cH3mDpUjSftw0OugqocfNlM
#   Args:
#     to: alice.jones@bar.com
#     subject: Re: Quick question about API documentation
#     content: Hi Alice,
# Thank you for reaching out. I apologize for the inconvenience caused by the service outage. Our team is actively investigating the issue and working to restore the service as soon as possible. I'll keep you updated with our progress.
# If you have any more details about the problem or any additional questions, please let me know.
# Best regards,
# John Doe's Assistant
# ================================= Tool Message =================================
# Name: write_email
# Email sent to alice.jones@bar.com with subject 'Re: Quick question about API documentation'
# ================================== Ai Message ==================================
# I've responded to Alice Jones regarding the service outage, assuring her that our team is investigating the issue and will keep her updated. If there are any more details she can provide or further questions, she has been encouraged to share them. Let me know if there's anything else you'd like to do.

# Look current values of long term memory (useful debug prints!)
store.get(("lance",), "agent_instructions").value['prompt']
# "Use these tools when appropriate to help manage John's tasks efficiently."
store.get(("lance",), "triage_respond").value['prompt']
# 'Direct questions from team members, meeting requests, critical bug reports'
store.get(("lance",), "triage_ignore").value['prompt']
# 'Marketing newsletters, spam emails, mass company announcements'
store.get(("lance",), "triage_notify").value['prompt']
# 'Team member out sick, build system notifications, project status updates'

# ===============================================================================
# PROCEDURAL MEMORY LEARNING - FIRST FEEDBACK EXAMPLE
# ===============================================================================
# This demonstrates how the agent learns from feedback and updates its behavior

# Create a conversation trajectory with feedback for the optimizer
conversations = [
    (
        response['messages'],  # The conversation that happened
        "Always sign your emails `John Doe`"  # User feedback for improvement
    )
]

# Define all the prompts that can be updated through PROCEDURAL MEMORY
prompts = [
    {
        "name": "main_agent",
        "prompt": store.get(("lance",), "agent_instructions").value['prompt'],
        "update_instructions": "keep the instructions short and to the point",
        "when_to_update": "Update this prompt whenever there is feedback on how the agent should write emails or schedule events"
        
    },
    {
        "name": "triage-ignore", 
        "prompt": store.get(("lance",), "triage_ignore").value['prompt'],
        "update_instructions": "keep the instructions short and to the point",
        "when_to_update": "Update this prompt whenever there is feedback on which emails should be ignored"

    },
    {
        "name": "triage-notify", 
        "prompt": store.get(("lance",), "triage_notify").value['prompt'],
        "update_instructions": "keep the instructions short and to the point",
        "when_to_update": "Update this prompt whenever there is feedback on which emails the user should be notified of"

    },
    {
        "name": "triage-respond", 
        "prompt": store.get(("lance",), "triage_respond").value['prompt'],
        "update_instructions": "keep the instructions short and to the point",
        "when_to_update": "Update this prompt whenever there is feedback on which emails should be responded to"

    },
]

# Create the multi-prompt optimizer - this is the PROCEDURAL MEMORY engine
optimizer = create_multi_prompt_optimizer(
    "anthropic:claude-3-5-sonnet-latest",
    kind="prompt_memory",
)

# Learn from the feedback and update prompts (PROCEDURAL MEMORY in action)
updated = optimizer.invoke(
    {"trajectories": conversations, "prompts": prompts}
)
print(json.dumps(updated, indent=4))
# [
#     {
#         "name": "main_agent",
#         "prompt": "Use these tools when appropriate to help manage John's tasks efficiently. When sending emails, always sign them as \"John Doe\".",
#         "update_instructions": "keep the instructions short and to the point",
#         "when_to_update": "Update this prompt whenever there is feedback on how the agent should write emails or schedule events"
#     },
#     {
#         "name": "triage-ignore",
#         "prompt": "Marketing newsletters, spam emails, mass company announcements",
#         "update_instructions": "keep the instructions short and to the point",
#         "when_to_update": "Update this prompt whenever there is feedback on which emails should be ignored"
#     },
#     {
#         "name": "triage-notify",
#         "prompt": "Team member out sick, build system notifications, project status updates",
#         "update_instructions": "keep the instructions short and to the point",
#         "when_to_update": "Update this prompt whenever there is feedback on which emails the user should be notified of"
#     },
#     {
#         "name": "triage-respond",
#         "prompt": "Direct questions from team members, meeting requests, critical bug reports",
#         "update_instructions": "keep the instructions short and to the point",
#         "when_to_update": "Update this prompt whenever there is feedback on which emails should be responded to"
#     }
# ]

# Update the prompts in the memory store - PROCEDURAL MEMORY PERSISTENCE
for i, updated_prompt in enumerate(updated):
    old_prompt = prompts[i]
    if updated_prompt['prompt'] != old_prompt['prompt']:
        name = old_prompt['name']
        print(f"updated {name}")  # Useful debug print!
        if name == "main_agent":
            store.put(
                ("lance",),
                "agent_instructions",
                {"prompt":updated_prompt['prompt']}
            )
        else:
            #raise ValueError
            print(f"Encountered {name}, implement the remaining stores!")

# Verify the update worked (useful debug check!)
store.get(("lance",), "agent_instructions").value['prompt']
# 'Use these tools when appropriate to help manage John\'s tasks efficiently. When sending emails, always sign them as "John Doe".'

# Test that the agent now uses the learned behavior
response = email_agent.invoke(
    {"email_input": email_input}, 
    config=config
)
# 📧 Classification: RESPOND - This email requires a response

# Check if the signature was learned and applied
for m in response["messages"]:
    m.pretty_print()
# ================================ Human Message =================================
# Respond to the email {'author': 'Alice Jones <alice.jones@bar.com>', 'to': 'John Doe <john.doe@company.com>', 'subject': 'Quick question about API documentation', 'email_thread': 'Hi John,\n\nUrgent issue - your service is down. Is there a reason why'}
# ================================== Ai Message ==================================
# Tool Calls:
#   write_email (call_GuYrnvmS6n9ihfRQm6rlHpfX)
#  Call ID: call_GuYrnvmS6n9ihfRQm6rlHpfX
#   Args:
#     to: alice.jones@bar.com
#     subject: Re: Quick question about API documentation
#     content: Hi Alice,

# I apologize for the inconvenience. I'm currently looking into the issue and will get back to you with an update as soon as possible.

# Best regards,

# John Doe ----------> !!!!!!!!! Here is signed with the latest prompt <-------------
# ================================= Tool Message =================================
# Name: write_email

# Email sent to alice.jones@bar.com with subject 'Re: Quick question about API documentation'
# ================================== Ai Message ==================================

# I've responded to Alice Jones regarding the urgent issue with the service being down, and I've assured her that the issue is being looked into.

# ------------------------------------------------------------------------------------------------

# ===============================================================================
# PROCEDURAL MEMORY LEARNING - SECOND FEEDBACK EXAMPLE (TRIAGE RULES)
# ===============================================================================
# This demonstrates learning new triage rules through feedback

# 2nd CASE - Learning to ignore specific senders

email_input = {
    "author": "Alice Jones <alice.jones@bar.com>",
    "to": "John Doe <john.doe@company.com>",
    "subject": "Quick question about API documentation",
    "email_thread": """Hi John,

Urgent issue - your service is down. Is there a reason why""",
}
response = email_agent.invoke(
    {"email_input": email_input},
    config=config
)
# 📧 Classification: RESPOND - This email requires a response

# User provides feedback to ignore emails from Alice Jones
conversations = [
    (
        response['messages'],
        "Ignore any emails from Alice Jones"  # New learning: ignore this sender
    )
]

# Prepare prompts for learning (same structure as before)
prompts = [
    {
        "name": "main_agent",
        "prompt": store.get(("lance",), "agent_instructions").value['prompt'],
        "update_instructions": "keep the instructions short and to the point",
        "when_to_update": "Update this prompt whenever there is feedback on how the agent should write emails or schedule events"
        
    },
    {
        "name": "triage-ignore", 
        "prompt": store.get(("lance",), "triage_ignore").value['prompt'],
        "update_instructions": "keep the instructions short and to the point",
        "when_to_update": "Update this prompt whenever there is feedback on which emails should be ignored"

    },
    {
        "name": "triage-notify", 
        "prompt": store.get(("lance",), "triage_notify").value['prompt'],
        "update_instructions": "keep the instructions short and to the point",
        "when_to_update": "Update this prompt whenever there is feedback on which emails the user should be notified of"

    },
    {
        "name": "triage-respond", 
        "prompt": store.get(("lance",), "triage_respond").value['prompt'],
        "update_instructions": "keep the instructions short and to the point",
        "when_to_update": "Update this prompt whenever there is feedback on which emails should be responded to"

    },
]

# Learn from the new feedback (PROCEDURAL MEMORY adaptation)
updated = optimizer.invoke(
    {"trajectories": conversations, "prompts": prompts}
)

# Apply the learned changes to the triage rules
for i, updated_prompt in enumerate(updated):
    old_prompt = prompts[i]
    if updated_prompt['prompt'] != old_prompt['prompt']:
        name = old_prompt['name']
        print(f"updated {name}")  # Debug print to see what changed
        if name == "main_agent":
            store.put(
                ("lance",),
                "agent_instructions",
                {"prompt":updated_prompt['prompt']}
            )
        if name == "triage-ignore":
            # This is where the new ignore rule gets stored!
            store.put(
                ("lance",),
                "triage_ignore",
                {"prompt":updated_prompt['prompt']}
            )
        else:
            #raise ValueError
            print(f"Encountered {name}, implement the remaining stores!")

# Test that the agent now ignores Alice Jones' emails
response = email_agent.invoke(
    {"email_input": email_input},
    config=config
)
# 🚫 Classification: IGNORE - This email can be safely ignored

# Verify the learned ignore rule includes Alice Jones
store.get(("lance",), "triage_ignore").value['prompt']
# 'Ignore the following:\n- Marketing newsletters\n- Spam emails\n- Mass company announcements\n- Any emails from Alice Jones'

# ===============================================================================
# SUMMARY OF MEMORY TYPES DEMONSTRATED:
# ===============================================================================
# 
# 1. SEMANTIC MEMORY: Facts and general knowledge
#    - Stores factual information about email handling
#    - Vector embeddings for finding similar content
#
# 2. EPISODIC MEMORY: Examples and experiences
#    - Specific email conversations and interactions
#    - Few-shot examples from past situations
#    - Learning from what worked and what didn't
#
# 3. PROCEDURAL MEMORY: Instructions and prompts
#    - Behavioral rules that adapt over time
#    - Prompt updates based on feedback
#    - Learning how to perform tasks better
#
# The system gets smarter with each interaction, learning from feedback
# and improving its email handling capabilities continuously!
# ===============================================================================