from dotenv import load_dotenv

_ = load_dotenv()

from langgraph.graph import StateGraph, END
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults

tool = TavilySearchResults(max_results=2)

from langgraph.checkpoint.sqlite import SqliteSaver

memory = SqliteSaver.from_conn_string(":memory:")

class Agent:
    def __init__(self, model, tools, checkpointer, system=""):
        self.system = system
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_openai)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges("llm", self.exists_action, {True: "action", False: END})
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        # Compile the graph with the checkpointer memory
        self.graph = graph.compile(checkpointer=checkpointer)
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)

    def call_openai(self, state: AgentState):
        messages = state['messages']
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages
        message = self.model.invoke(messages)
        return {'messages': [message]}

    def exists_action(self, state: AgentState):
        result = state['messages'][-1]
        return len(result.tool_calls) > 0

    def take_action(self, state: AgentState):
        tool_calls = state['messages'][-1].tool_calls
        results = []
        for t in tool_calls:
            print(f"Calling: {t}")
            result = self.tools[t['name']].invoke(t['args'])
            results.append(ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result)))
        print("Back to the model!")
        return {'messages': results}

prompt = """You are a smart research assistant. Use the search engine to look up information. \
You are allowed to make multiple calls (either together or in sequence). \
Only look up information when you are sure of what you want. \
If you need to look up some information before asking a follow up question, you are allowed to do that!
"""
model = ChatOpenAI(model="gpt-4o")
abot = Agent(model, [tool], system=prompt, checkpointer=memory)

messages = [HumanMessage(content="What is the weather in sf?")]

## ------------------------------------------------------------------------------------------------

## PERSISTANCE

## We set a thread id to keep track of the conversation
thread = {"configurable": {"thread_id": "1"}}

## We stream the conversation and print the messages to understand the flow of the conversation
for event in abot.graph.stream({"messages": messages}, thread):
    for v in event.values():
        print(v['messages'])
## [AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_bmfLa92f6oAIKN9KvXtqbKDz', 'function': {'arguments': '{"query":"current weather in San Francisco"}', 'name': 'tavily_search_results_json'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 22, 'prompt_tokens': 151, 'total_tokens': 173, 'prompt_tokens_details': {'cached_tokens': 0, 'audio_tokens': 0}, 'completion_tokens_details': {'reasoning_tokens': 0, 'audio_tokens': 0, 'accepted_prediction_tokens': 0, 'rejected_prediction_tokens': 0}}, 'model_name': 'gpt-4o', 'system_fingerprint': 'fp_831e067d82', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-61b0d6e4-385b-4d1e-8946-d4302a784230-0', tool_calls=[{'name': 'tavily_search_results_json', 'args': {'query': 'current weather in San Francisco'}, 'id': 'call_bmfLa92f6oAIKN9KvXtqbKDz'}])]
## Calling: {'name': 'tavily_search_results_json', 'args': {'query': 'current weather in San Francisco'}, 'id': 'call_bmfLa92f6oAIKN9KvXtqbKDz'}
## Back to the model!
## [ToolMessage(content='[{\'url\': \'https://weathershogun.com/weather/usa/ca/san-francisco/480/may/2025-05-22\', \'content\': \'San Francisco, California Weather: Thursday, May 22, 2025. Cloudy weather, overcast skies with clouds. Day 70°. Night 57°.\'}, {\'url\': \'https://www.weatherbug.com/weather-forecast/now/san-francisco-ca-94103\', \'content\': "Today\'s Weather - San Francisco, CA. May 22, 2025 9:11 AM. Exploratorium. 58°. Feels Like 58°. Hi 64°F Lo 50°F. Mostly Sunny. Live Radar. Weather Details."}]', name='tavily_search_results_json', tool_call_id='call_bmfLa92f6oAIKN9KvXtqbKDz')]
## [AIMessage(content='The current weather in San Francisco is mostly sunny with a temperature of 58°F, though it feels like 58°F. The high is expected to reach 64°F and the low around 50°F.', response_metadata={'token_usage': {'completion_tokens': 43, 'prompt_tokens': 340, 'total_tokens': 383, 'prompt_tokens_details': {'cached_tokens': 0, 'audio_tokens': 0}, 'completion_tokens_details': {'reasoning_tokens': 0, 'audio_tokens': 0, 'accepted_prediction_tokens': 0, 'rejected_prediction_tokens': 0}}, 'model_name': 'gpt-4o', 'system_fingerprint': 'fp_76544d79cb', 'finish_reason': 'stop', 'logprobs': None}, id='run-7aa2facf-4dd0-4831-9657-83bbd0e0c7b5-0')]

## We set a thread id to keep track of the conversation and ask another question to the same agent
messages = [HumanMessage(content="What about in la?")]
thread = {"configurable": {"thread_id": "1"}}
for event in abot.graph.stream({"messages": messages}, thread):
    for v in event.values():
        print(v)
## [AIMessage(content='', additional_kwargs={'tool_calls': [{'id': 'call_yDECnctIXTDDBlbISblX3E2O', 'function': {'arguments': '{"query":"current weather in Los Angeles"}', 'name': 'tavily_search_results_json'}, 'type': 'function'}]}, response_metadata={'token_usage': {'completion_tokens': 22, 'prompt_tokens': 410, 'total_tokens': 432, 'prompt_tokens_details': {'cached_tokens': 0, 'audio_tokens': 0}, 'completion_tokens_details': {'reasoning_tokens': 0, 'audio_tokens': 0, 'accepted_prediction_tokens': 0, 'rejected_prediction_tokens': 0}}, 'model_name': 'gpt-4o', 'system_fingerprint': 'fp_76544d79cb', 'finish_reason': 'tool_calls', 'logprobs': None}, id='run-a25e434a-8a97-452e-952a-38ada57f6318-0', tool_calls=[{'name': 'tavily_search_results_json', 'args': {'query': 'current weather in Los Angeles'}, 'id': 'call_yDECnctIXTDDBlbISblX3E2O'}])]
## Calling: {'name': 'tavily_search_results_json', 'args': {'query': 'current weather in Los Angeles'}, 'id': 'call_yDECnctIXTDDBlbISblX3E2O'}
## Back to the model!
## [ToolMessage(content='[{\'url\': \'https://weathershogun.com/weather/usa/ca/los-angeles/451/may/2025-05-22\', \'content\': \'Los Angeles, California Weather: Thursday, May 22, 2025. Cloudy weather, overcast skies with clouds. Day 72°. Night 57°. Precipitation 25 %.\'}, {\'url\': \'https://world-weather.info/forecast/usa/los_angeles/may-2025/\', \'content\': "Weather in Los Angeles in May 2025\\n\\nLos Angeles Weather Forecast for May 2025 is based on long term prognosis and previous years\' statistical data.\\n\\nMay\\n\\n+57°\\n\\n+57°\\n\\n+57°\\n\\n+55°\\n\\n+55°\\n\\n+57°\\n\\n+57°\\n\\n+54°\\n\\n+57°\\n\\n+63°\\n\\n+63°\\n\\n+63°\\n\\n+54°\\n\\n+52°\\n\\n+54°\\n\\n+55°\\n\\n+59°\\n\\n+55°\\n\\n+55°\\n\\n+59°\\n\\n+61°\\n\\n+59°\\n\\n+63°\\n\\n+61°\\n\\n+59°\\n\\n+59°\\n\\n+61°\\n\\n+57°\\n\\n+57°\\n\\n+61°\\n\\n+64°\\n\\nExtended weather forecast in Los Angeles\\n\\nWeather in large and nearby cities\\n\\nWeather in Washington, D.C.+66°\\n\\nSacramento+82°\\n\\nMarina del Rey+63° [...] Monterey Park+81°\\n\\nNorth Glendale+82°\\n\\nNorwalk+77°\\n\\nPasadena+82°\\n\\nRosemead+82°\\n\\nSanta Monica+66°\\n\\nSouth El Monte+82°\\n\\nManhattan Beach+73°\\n\\nInglewood+72°\\n\\nBellflower+75°\\n\\nBeverly Hills+73°\\n\\nBurbank+84°\\n\\nCompton+75°\\n\\nStudio City+82°\\n\\nWarner Center+84°"}]', name='tavily_search_results_json', tool_call_id='call_yDECnctIXTDDBlbISblX3E2O')]
## [AIMessage(content='The current weather in Los Angeles, California, is cloudy with overcast skies. The daytime temperature is around 72°F, and the nighttime temperature is expected to be 57°F, with a 25% chance of precipitation.', response_metadata={'token_usage': {'completion_tokens': 47, 'prompt_tokens': 945, 'total_tokens': 992, 'prompt_tokens_details': {'cached_tokens': 0, 'audio_tokens': 0}, 'completion_tokens_details': {'reasoning_tokens': 0, 'audio_tokens': 0, 'accepted_prediction_tokens': 0, 'rejected_prediction_tokens': 0}}, 'model_name': 'gpt-4o', 'system_fingerprint': 'fp_76544d79cb', 'finish_reason': 'stop', 'logprobs': None}, id='run-130116f0-5864-4139-854e-721f3c8f2f39-0')]

## With the same thread id, we can ask another question to the same agent comparing the results of the previous question
messages = [HumanMessage(content="Which one is warmer?")]
thread = {"configurable": {"thread_id": "1"}}
for event in abot.graph.stream({"messages": messages}, thread):
    for v in event.values():
        print(v['messages'])
## [AIMessage(content="Los Angeles is currently slightly warmer than San Francisco. Los Angeles has a daytime temperature of 72°F, compared to San Francisco's 70°F. Both cities have the same nighttime temperature of 57°F.", response_metadata={'token_usage': {'completion_tokens': 43, 'prompt_tokens': 1004, 'total_tokens': 1047, 'prompt_tokens_details': {'cached_tokens': 0, 'audio_tokens': 0}, 'completion_tokens_details': {'reasoning_tokens': 0, 'audio_tokens': 0, 'accepted_prediction_tokens': 0, 'rejected_prediction_tokens': 0}}, 'model_name': 'gpt-4o', 'system_fingerprint': 'fp_76544d79cb', 'finish_reason': 'stop', 'logprobs': None}, id='run-0717f06e-a4e5-4558-95fe-ba9a06709481-0')]

## We show here that with a different thread id, the agent will not remember the previous conversation
messages = [HumanMessage(content="Which one is warmer?")]
thread = {"configurable": {"thread_id": "2"}}
for event in abot.graph.stream({"messages": messages}, thread):
    for v in event.values():
        print(v['messages'])
## [AIMessage(content="Could you please clarify what you're comparing to determine which is warmer? Are you comparing two specific locations, types of clothing, materials, or something else? Let me know so I can provide the appropriate information.", response_metadata={'token_usage': {'completion_tokens': 43, 'prompt_tokens': 149, 'total_tokens': 192, 'prompt_tokens_details': {'cached_tokens': 0, 'audio_tokens': 0}, 'completion_tokens_details': {'reasoning_tokens': 0, 'audio_tokens': 0, 'accepted_prediction_tokens': 0, 'rejected_prediction_tokens': 0}}, 'model_name': 'gpt-4o', 'system_fingerprint': 'fp_f9f4fb6dbf', 'finish_reason': 'stop', 'logprobs': None}, id='run-23762721-1634-400f-a3f4-9ee7e9f823a8-0')]

## ------------------------------------------------------------------------------------------------

## STREAMING

from langgraph.checkpoint.aiosqlite import AsyncSqliteSaver

memory = AsyncSqliteSaver.from_conn_string(":memory:")
abot = Agent(model, [tool], system=prompt, checkpointer=memory)

import asyncio

## Here we stream the conversation and print the messages to understand the flow of the conversation in an async way with streaming
async def stream_weather():
    messages = [HumanMessage(content="What is the weather in SF?")]
    thread = {"configurable": {"thread_id": "4"}}
    async for event in abot.graph.astream_events({"messages": messages}, thread, version="v1"):
        kind = event["event"]
        if kind == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            if content:
                # Empty content in the context of OpenAI means
                # that the model is asking for a tool to be invoked.
                # So we only print non-empty content
                print(content, end="|")
                ## Calling: {'name': 'tavily_search_results_json', 'args': {'query': 'current weather in San Francisco'}, 'id': 'call_SiDSmmJHxBStl6yiHfD9bx6f'}
                ## Back to the model!
                ## The| current| weather| in| San| Francisco| is| sunny| with| a| temperature| reaching| up| to| |71|°F| during| the| day| and| dropping| to| around| |57|°F| at| night|.|

asyncio.run(stream_weather())