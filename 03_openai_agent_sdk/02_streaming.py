import os
from openai import AsyncOpenAI
from agents import Agent, OpenAIChatCompletionsModel, Runner
from openai.types.responses import ResponseTextDeltaEvent
from dotenv import load_dotenv
import asyncio

load_dotenv()

gemini_api_key = os.getenv('GEMINI_API_KEY')

client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

async def main():
    agent=Agent(
        name="Assistant",
        instructions="A helpful assistant that can answer questions and provide information.",
        model=OpenAIChatCompletionsModel(model="gemini-2.0-flash", openai_client=client)
    )
    query = input("What would you like to ask?")
    result=Runner.run_streamed(agent,input=query)
    async for event in result.stream_events():
        if event.type == "raw_response_event" and isinstance(event.data, ResponseTextDeltaEvent):
            print(event.data.delta, end='', flush=True)
            
asyncio.run(main())