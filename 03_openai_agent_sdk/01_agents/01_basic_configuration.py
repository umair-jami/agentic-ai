from agents import Agent,OpenAIChatCompletionsModel, Runner, function_tool
from openai import AsyncOpenAI
import os
from dotenv import load_dotenv


load_dotenv()

gemini_api_key=os.getenv("GEMINI_API_KEY")

client=AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

@function_tool
def get_weather(city:str)->str:
    return f"The weaather in {city} is sunny"

agent=Agent(
    name="Assistant",
    instructions="You are a weather assistant...",
    model=OpenAIChatCompletionsModel(model="gemini-2.0-flash", openai_client=client),
    tools=[get_weather]
)

query=input("Enter the query: ")

result=Runner.run_sync(
    agent,
    query
)
print(result.final_output)