from fastapi import FastAPI
from pymongo import MongoClient
from dotenv import load_dotenv
from bson.objectid import ObjectId
from pydantic import BaseModel
from datetime import datetime
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

class Todo(BaseModel):
    title: str
    description: str
    Created_at: str
    Completed: bool = False

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
                "created_at": todo["created_at"],
                "completed": todo["completed"]
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

# filter base on id
@app.get("/todos/{todo_id}")
def read_todo_by_id(todo_id:str):
    try:
        todo=db.todos.find_one({"_id":ObjectId(todo_id)})
        if todo is None:
            return {
                "data": None,
                "message": "Todo not found",
                "status": "error",
                "error": "Todo with the given ID does not exist"
            }
        return {
            "data": {
                "id": str(todo["_id"]),
                "title": todo["title"],
                "description": todo["description"],
                "created_at": todo["created_at"],
                "completed": todo["completed"]
            },
            "message": "Todo retrieved successfully",
            "status": "success",
            "error": None
        }
    except Exception as e:
        return {
            "data": None,
            "message": "Error retrieving todo",
            "status": "error",
            "error": str(e)
        }
        
@app.get("/todos/{todo_title}")
def read_todo_by_title(todo_title:str):
    try:
        todo=db.todos.find_one({"title":todo_title})
        if todo is None:
            return {
                "data": None,
                "message": "Todo not found",
                "status": "error",
                "error": "Todo with the given ID does not exist"
            }
        return {
            "data": {
                "id": str(todo["_id"]),
                "title": todo["title"],
                "description": todo["description"],
                "created_at": todo["created_at"],
                "completed": todo["completed"]
            },
            "message": "Todo retrieved successfully",
            "status": "success",
            "error": None
        }
    except Exception as e:
        return {
            "data": None,
            "message": "Error retrieving todo",
            "status": "error",
            "error": str(e)
        }
        
# Create a new todo
@app.post("/create_todo")
def create_todo(create_todo:Todo):
    try:
        result=db.todos.insert_one({
            "title": create_todo.title,
            "description": create_todo.description,
            "Created_at": str(datetime.now()),
            "Completed": create_todo.Completed
        })
        return {
            "data":{
                "id": str(result.inserted_id)
            },
            "message": "Todo created successfully",
            "status": "success",
            "error": None
        }
    except Exception as e:
        print(f"Error creating todo: {e}")
        return {
            "data": {},
            "error": "Error creating todo",
            "message": str(e),
            "status": "failed"
            }
        
        
@app.delete("/delete_todo/{todo_id}")
def delete_todo(todo_id:str):
    try:
        result=db.todos.delete_one({"_id":ObjectId(todo_id)})
        if result.deleted_count==0:
            return {
                "data": None,
                "message": "Todo not found",
                "status": "error",
                "error": "Todo with the given ID does not exist"
            }
        return {
            "data":{},
            "message": "Todo deleted successfully",
            "status":" success",
            "error":None
        }
    except Exception as e:
        return {
            "data": None,
            "message": "Error deleting todo",
            "status": "error",
            "error": str(e)
        }

@app.put("/update_todo/{id}")
def update_todo(id: str, todo: Todo):
    try:
        result = db.todos.update_one(
            {"_id": ObjectId(id)},
            {
                "$set": {
                    "title": todo.title,
                    "description": todo.description,
                    "status": todo.status
                }
            }
        )
        if result.modified_count == 0:
            return {
                "data": {},
                "error": "Todo not found",
                "message": "Todo not found",
                "status": "failed"
                }
        return {
            "data": {},
            "error": None,
            "message": "Todo updated successfully",
            "status": "success"
            }
    except Exception as e:
        print(f"Error updating todo: {e}")
        return {
            "data": {},
            "error": "Error updating todo",
            "message": str(e),
            "status": "failed"
            }