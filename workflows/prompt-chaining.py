from typing import Optional
from datetime import datetime
from pydantic import BaseModel, Field
from openai import OpenAI
import os
import logging

# Set up logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
model = "gpt-4o"

# --------------------------------------------------------------
# Step 1: Define the data models for each stage
# --------------------------------------------------------------


class EventExtraction(BaseModel):
    """First LLM call: Extract basic event information"""

    description: str = Field(description="Raw description of the event")
    is_calendar_event: bool = Field(
        description="Whether this text describes a calendar event"
    )
    confidence_score: float = Field(description="Confidence score between 0 and 1")


class EventDetails(BaseModel):
    """Second LLM call: Parse specific event details"""

    name: str = Field(description="Name of the event")
    date: str = Field(
        description="Date and time of the event. Use ISO 8601 to format this value."
    )
    duration_minutes: int = Field(description="Expected duration in minutes")
    participants: list[str] = Field(description="List of participants")


class EventConfirmation(BaseModel):
    """Third LLM call: Generate confirmation message"""

    confirmation_message: str = Field(
        description="Natural language confirmation message"
    )
    calendar_link: Optional[str] = Field(
        description="Generated calendar link if applicable"
    )

# --------------------------------------------------------------
# Step 2: Define the functions
# --------------------------------------------------------------

def extract_event_info(user_input: str) -> EventExtraction:
    """First LLM call to determine if input is a calendar event"""
    logger.info("Starting event extraction analysis")
    logger.debug(f"Input text: {user_input}")

    today = datetime.now()
    date_context = f"Today's date is {today.strftime('%Y-%m-%d')}"

    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": f"{date_context} Analyze if the text describes a calendar event.",},
            {"role": "user", "content": user_input},
        ],
        response_format=EventExtraction,
    )

    result = completion.choices[0].message.parsed
    logger.info(f"Extraction complete - Is calendar event: {result.is_calendar_event}, Confidence: {result.confidence_score:.2f}")
    return result

def parse_event_details(description: str) -> EventDetails:
    """Second LLM call to parse specific event details"""
    logger.info("Starting event details parsing")
    logger.debug(f"Input text: {description}")

    today = datetime.now()
    date_context = f"Today's date is {today.strftime('%Y-%m-%d')}"

    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": f"{date_context} Extract detailed event information. When dates reference 'next Tuesday' or similar relative dates, use this current date as reference."},
            {"role": "user", "content": description},
        ],
        response_format=EventDetails,
    )

    result = completion.choices[0].message.parsed
    logger.info(
        f"Parsed event details - Name: {result.name}, Date: {result.date}, Duration: {result.duration_minutes}min"
    )
    logger.debug(f"Participants: {', '.join(result.participants)}")
    return result

def generate_confirmation_message(event_details: EventDetails) -> EventConfirmation:
    """Third LLM call to generate confirmation message"""
    logger.info("Generating confirmation message")

    completion = client.beta.chat.completions.parse(
        model=model,
        messages=[
            {"role": "system", "content": "Generate a natural language confirmation message for the event. Sign of with your name; Susie"},
            {"role": "user", "content": str(event_details.model_dump())},
        ],
        response_format=EventConfirmation,
    )

    result = completion.choices[0].message.parsed
    logger.info(f"Confirmation message generated: {result.confirmation_message}")
    return result

# --------------------------------------------------------------
# Step 3: Chain the functions together
# --------------------------------------------------------------

def process_calendar_request(user_input: str) -> Optional[EventConfirmation]:
    """Main function to process calendar request"""
    logger.info("Processing calendar request")

    try:
        # Extract event information
        initial_extraction = extract_event_info(user_input)
            # Gate check: Verify if it's a calendar event with sufficient confidence
        if (
            not initial_extraction.is_calendar_event
            or initial_extraction.confidence_score < 0.7
        ):
            logger.warning(
                f"Gate check failed - is_calendar_event: {initial_extraction.is_calendar_event}, confidence: {initial_extraction.confidence_score:.2f}"
            )
            return None

        logger.info("Gate check passed, proceeding with event processing")

        # Parse event details
        event_details = parse_event_details(user_input)

        # Generate confirmation message
        confirmation = generate_confirmation_message(event_details)

        return confirmation
    except Exception as e:
        logger.error(f"Error processing calendar request: {e}")
        return None

# --------------------------------------------------------------
# Step 4: Test the chain with a valid input
# --------------------------------------------------------------

user_input = "Let's schedule a 1h team meeting next Tuesday at 2pm with Alice and Bob to discuss the project roadmap."

result = process_calendar_request(user_input)
if result:
    logger.info(f"Final confirmation message: {result.confirmation_message}")
    if result.calendar_link:
        logger.info(f"Calendar link: {result.calendar_link}")
else:
    print("This doesn't appear to be a calendar event request.")

# --------------------------------------------------------------
# Step 5: Test the chain with an invalid input
# --------------------------------------------------------------

user_input = "Can you send an email to Alice and Bob to discuss the project roadmap?"

result = process_calendar_request(user_input)
if result:
    print(f"Confirmation: {result.confirmation_message}")
    if result.calendar_link:
        print(f"Calendar Link: {result.calendar_link}")
else:
    print("This doesn't appear to be a calendar event request.")