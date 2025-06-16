from openai import AsyncOpenAI
from agents import Agent, OpenAIChatCompletionsModel,Runner,function_tool
import os
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

gemini_api_key=os.getenv('GEMINI_API_KEY')

client=AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

@function_tool
async def fetch_weather(location)->str:
    """Fetch the weather for a given location.
    args:
        location:The location to fetch the weather for.
        
    returns:
        The weather for the given location.
    """
    print(f"Fetching weather for {location}...")
    return "sunny"

@function_tool
async def fetch_news(topic)->str:
    """Fetch the latest news for a given topic.
    args:
        topic:The topic to fetch the news for.
        
    returns:
        The latest news for the given topic.
    """
    print(f"Fetching news for {topic}...")
    return "Latest news on AI: OpenAI releases new model."

@function_tool
async def fetch_stock_price(symbol)->str:
    """Fetch the stock price for a given symbol.
    args:
        symbol:The stock symbol to fetch the price for.
        
    returns:
        The stock price for the given symbol.
    """
    print(f"Fetching stock price for {symbol}...")
    return "Stock price for AAPL: $150.00"

agent=Agent(
    name="Assistant",
    instructions="You are a helpful assistant. You can fetch weather, news, and stock prices.",
    model=OpenAIChatCompletionsModel(model="gemini-2.0-flash",openai_client=client),
    tools=[fetch_weather,fetch_news,fetch_stock_price]
)

query= input("What would you like to know? ")
result=Runner.run_sync(agent, query)
print("Final Output:", result.final_output)