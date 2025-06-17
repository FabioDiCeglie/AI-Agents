import os
from dotenv import load_dotenv
from langchain_core.tools import tool
from langgraph.graph import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from typing import Literal
from prompts import triage_system_prompt, triage_user_prompt, agent_system_prompt
from typing_extensions import TypedDict, Literal, Annotated
from langchain.chat_models import init_chat_model
from langgraph.prebuilt import create_react_agent

# This script demonstrates a basic email assistant that uses LangGraph to triage
# and respond to emails. It uses a language model to classify incoming emails
# and, if a response is needed, delegates to a separate agent equipped with tools
# to write emails, schedule meetings, and check calendars.

_ = load_dotenv()
project_id = os.environ.get("GOOGLE_CLOUD_PROJECT")

# --- User and Prompt Configuration ---
# Defines the user's profile and instructions for the AI agents.
# This information provides context to the language models.
profile = {
    "name": "John",
    "full_name": "John Doe",
    "user_profile_background": "Senior software engineer leading a team of 5 developers",
}

prompt_instructions = {
    "triage_rules": {
        "ignore": "Marketing newsletters, spam emails, mass company announcements",
        "notify": "Team member out sick, build system notifications, project status updates",
        "respond": "Direct questions from team members, meeting requests, critical bug reports",
    },
    "agent_instructions": "Use these tools when appropriate to help manage John's tasks efficiently."
}

# Example incoming email that the agent will process.
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

# --- LLM and Tools Initialization ---

# Initialize the language model for the triage step.
# This model will classify the email based on the triage rules.
llm = init_chat_model(
    "gemini-1.5-pro", model_provider="google_vertexai", temperature=0, project=project_id
)

# --- Agent Tools ---
# Define tools that the response agent can use to interact with other systems.
# In a real application, these would integrate with actual email, calendar, etc.

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

# --- Response Agent Definition ---

# Create the agent that will handle emails requiring a response.
tools = [write_email, schedule_meeting, check_calendar_availability]

# Initialize a separate LLM for the response agent.
agent_llm = init_chat_model(
    "gemini-1.5-pro",
    model_provider="google_vertexai",
    temperature=0,
    project=project_id,
)

def create_prompt(state):
    return [
        {
            "role": "system", 
            "content": agent_system_prompt.format(
                instructions=prompt_instructions["agent_instructions"],
                **profile
                )
        }
    ] + state['messages']

# Create the response agent using LangGraph's prebuilt create_react_agent.
# This agent uses the ReAct (Reasoning and Acting) framework to decide which tool to use.
agent = create_react_agent(
    agent_llm,
    tools=tools,
    prompt=create_prompt,
)

# --- Graph State and Router ---

# Define the state for the LangGraph graph.
# It holds the initial email and the conversation history.
class State(TypedDict):
    email_input: dict
    messages: Annotated[list, add_messages]

def triage_router(state: State) -> Command[
    Literal["response_agent", "__end__"]
]:
    """
    This function acts as a router, classifying the incoming email and
    directing the graph to the appropriate next step.
    """
    author = state['email_input']['author']
    to = state['email_input']['to']
    subject = state['email_input']['subject']
    email_thread = state['email_input']['email_thread']

    # Format prompts for the triage LLM call
    system_prompt = triage_system_prompt.format(
        full_name=profile["full_name"],
        name=profile["name"],
        user_profile_background=profile["user_profile_background"],
        triage_no=prompt_instructions["triage_rules"]["ignore"],
        triage_notify=prompt_instructions["triage_rules"]["notify"],
        triage_email=prompt_instructions["triage_rules"]["respond"],
        examples=None
    )
    user_prompt = triage_user_prompt.format(
        author=author, 
        to=to, 
        subject=subject, 
        email_thread=email_thread
    )
    # Get the classification from the LLM
    result = llm.invoke(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
    )

    # Based on the classification, decide the next step in the graph
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

# --- Graph Construction and Execution ---

# Create the StateGraph, which defines the structure of the agent.
email_agent = StateGraph(State)

# Add the triage_router and response_agent as nodes in the graph.
email_agent = email_agent.add_node(triage_router)
email_agent = email_agent.add_node("response_agent", agent)

# Set the entry point of the graph.
email_agent = email_agent.add_edge(START, "triage_router")

# Compile the graph into a runnable object.
email_agent = email_agent.compile()

# Generate and save a visual representation of the graph.
# This helps in understanding the agent's flow.
# Show the agent
with open("agent_graph.png", "wb") as f:
    f.write(email_agent.get_graph(xray=True).draw_mermaid_png())

# --- Run the Agent ---

# Define the input for the agent.
email_input = {
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

# Invoke the agent with the sample email.
response = email_agent.invoke({"email_input": email_input})

# Print the final response and conversation history.
response.pretty_print()

for m in response["messages"]:
    m.pretty_print()