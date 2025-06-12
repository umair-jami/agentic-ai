from openai import AsyncOpenAI
from agents import Agent, OpenAIChatCompletionsModel, Runner
import os
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List

load_dotenv()

gemini_api_key = os.getenv('GEMINI_API_KEY')

client = AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

class Quiz(BaseModel):
    question:str
    options:List[str]
    answer:str
    
agent=Agent(
    name="QuizMaster",
    instructions="Create a quiz with a question, multiple options, and the correct answer.",
    model=OpenAIChatCompletionsModel(model="gemini-2.0-flash", openai_client=client),
    output_type=Quiz
)
query = input("What topic would you like a quiz on? ")
result=Runner.run_sync(agent, input=query)

print(result.final_output)