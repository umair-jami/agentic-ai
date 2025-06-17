from openai import AsyncOpenAI
from agents import Agent, OpenAIChatCompletionsModel, Runner, function_tool
from pymongo import MongoClient
from dotenv import load_dotenv
from pydantic import BaseModel
import os
import uuid

# Load environment variables
load_dotenv()
mongo_uri = os.getenv("MONGO_URI")
gemini_api_key = os.getenv("GEMINI_API_KEY")

# MongoDB connection
client_mongo = MongoClient(mongo_uri)
db = client_mongo["todo_db"]
todos_collection = db["todos"]

# OpenAI client
client=AsyncOpenAI(
    api_key=gemini_api_key,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# -------------------------------
#       Tool: Add Todo
# -------------------------------
class AddTodoInput(BaseModel):
    title: str
    description: str

@function_tool
async def add_todo(input: AddTodoInput) -> str:
    todo = {
        "id": str(uuid.uuid4()),
        "title": input.title,
        "description": input.description,
        "done": False
    }
    todos_collection.insert_one(todo)
    return f"Todo added: {input.title}"

# -------------------------------
#       Tool: Get All Todos
# -------------------------------
@function_tool
async def get_todos() -> str:
    todos = list(todos_collection.find({}, {"_id": 0}))
    if not todos:
        return "No todos found."
    return "\n".join([f"{t['title']} - {'Done' if t['done'] else 'Pending'}" for t in todos])

# -------------------------------
#       Tool: Mark Todo Done
# -------------------------------
class MarkDoneInput(BaseModel):
    title: str

@function_tool
async def mark_todo_done(input: MarkDoneInput) -> str:
    result = todos_collection.update_one(
        {"title": input.title},
        {"$set": {"done": True}}
    )
    if result.modified_count == 0:
        return f"No todo found with title '{input.title}'."
    return f"Marked '{input.title}' as done."

# -------------------------------
#       Tool: Delete Todo
# -------------------------------
class DeleteTodoInput(BaseModel):
    title: str

@function_tool
async def delete_todo(input: DeleteTodoInput) -> str:
    result = todos_collection.delete_one({"title": input.title})
    if result.deleted_count == 0:
        return f"No todo found with title '{input.title}'."
    return f"Deleted todo '{input.title}'."

# -------------------------------
#        Agent Setup
# -------------------------------
agent = Agent(
    name="TodoAgent",
    instructions="You are a helpful agent that can manage a todo list. You can add, view, update, and delete todos.",
    model=OpenAIChatCompletionsModel(model="gemini-2.0-flash", openai_client=client),
    tools=[add_todo, get_todos, mark_todo_done, delete_todo]
)

# -------------------------------
#        Run Agent
# -------------------------------
if __name__ == "__main__":
    query = input("What would you like to do with your todos? ")
    result = Runner.run_sync(agent, query)
    print("Final Output:", result.final_output)
