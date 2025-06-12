from openai import AsyncOpenAI
from agents import Agent, OpenAIChatCompletionsModel,Runner
import os
from dotenv import load_dotenv

load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")

client= AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

agent =Agent(
    name="Assistant",
    instructions="A helpful assistant that can answer questions and provide information.",
    model=OpenAIChatCompletionsModel(model="gemini-2.0-flash", openai_client=client),
)
query=input("What would you like to ask? ")
result=Runner.run_sync(agent, query)

print(result.final_output)