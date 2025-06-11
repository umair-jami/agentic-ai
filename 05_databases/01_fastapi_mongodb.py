from fastapi import FastAPI
from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()
app=FastAPI()

def get_db_client():
    """Initialize and return a MongoDB client."""
    try:
        client = MongoClient(os.getenv("DB_URI"))
        print("Connected to MongoDB successfully")
        return client
    except Exception as e:
        print(f"Error connecting to MongoDB: {e}")
        return None

client=get_db_client()
db=client["fastapidb"]

@app.get("/")
def read_root():
    return {"status": "Server is running"}


@app.get("/todos")
def read_todos():
    try:
        todos=db.todos.find()
        listTodos=[]
        for todo in todos:
            listTodos.append({
                "id": str(todo["_id"]),
                "title": todo["title"],
                "description": todo["description"],
                "created_at": todo["Created_at"],
                "completed": todo["Completed"]
            })
        return {
            "data":listTodos,
            "message": "Todos retrieved successfully",
            "status": "success",
            "error": None
        }
    except Exception as e:
        return {
            "data": None,
            "message": "Error retrieving todos",
            "status": "error",
            "error": str(e)
        }
