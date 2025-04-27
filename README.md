# LLM Agentic Workflow Examples

This repository contains Python examples demonstrating various agentic workflow patterns for Large Language Models (LLMs), inspired by the concepts discussed in Anthropic's article on [Building effective agents](https://www.anthropic.com/engineering/building-effective-agents).

The examples focus on a simple calendar scheduling task and utilize the OpenAI API with Pydantic for structured data handling.

## Principles

These examples aim to follow the principles of:

1.  **Simplicity:** Using direct LLM API calls and clear Python code rather than complex frameworks.
2.  **Composability:** Demonstrating how basic LLM calls can be chained, routed, or run in parallel to build more sophisticated logic.
3.  **Transparency:** The code is structured to make the flow of logic and LLM interactions easy to follow.

## Workflows Implemented

The `workflows/` directory contains implementations of the following patterns:

1.  **`prompt-chaining.py`**:
    *   **Pattern:** Prompt Chaining
    *   **Description:** Decomposes the task of scheduling a calendar event into a sequence of LLM calls:
        1.  **Extract:** Determine if the input is a calendar event request.
        2.  **Gate:** Check if confidence is high enough to proceed.
        3.  **Parse:** Extract event details (name, date, duration, participants).
        4.  **Confirm:** Generate a natural language confirmation message.
    *   **Reference:** [Anthropic: Prompt Chaining Workflow](https://www.anthropic.com/engineering/building-effective-agents#workflow-prompt-chaining)

2.  **`routing.py`**:
    *   **Pattern:** Routing
    *   **Description:** Classifies an incoming request related to calendar events and directs it to a specialized handler:
        1.  **Route:** An LLM call determines if the request is to `new_event`, `modify_event`, or `other`.
        2.  **Handle:** Based on the classification, calls a specific function (`handle_new_event` or `handle_modify_event`) to process the request using another LLM call tailored to that subtask.
    *   **Reference:** [Anthropic: Routing Workflow](https://www.anthropic.com/engineering/building-effective-agents#workflow-routing)

3.  **`parallization.py`** (Note: Filename typo):
    *   **Pattern:** Parallelization (Sectioning)
    *   **Description:** Runs multiple independent checks on the user input concurrently:
        1.  **Validate:** One LLM call checks if the input is a valid calendar request.
        2.  **Secure:** Another LLM call checks for potential security risks (e.g., prompt injection).
        3.  **Aggregate:** The results are combined to determine if the request is safe and valid to proceed. This pattern is useful for implementing guardrails or parallel evaluations.
    *   **Reference:** [Anthropic: Parallelization Workflow](https://www.anthropic.com/engineering/building-effective-agents#workflow-parallelization)

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd <repository-directory>
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **Set Environment Variable:**
    Ensure your OpenAI API key is set as an environment variable:
    ```bash
    export OPENAI_API_KEY='your-api-key'
    ```

## Usage

Navigate to the `workflows/` directory and run the Python scripts directly:

```bash
cd workflows
python prompt-chaining.py
python routing.py
python parallization.py
```

Each script contains example inputs and will print the results or log the processing steps to the console. You can modify the `user_input` variables within the scripts to test different scenarios. 