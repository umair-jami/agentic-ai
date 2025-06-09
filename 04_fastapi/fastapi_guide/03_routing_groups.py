from fastapi import FastAPI,APIRouter
from pydantic import BaseModel
from typing import Union,Optional
app=FastAPI()

# path parameters

@app.get("/users/{user_id}/{name}/{age}")
def get_user(user_id:int,name:str,age:int):
    try:
        return {
            "status":"success",
            "data":{
                "profile_url":"https://plus.unsplash.com/premium_photo-1734543932716-431337d9c3c4?q=80&w=2133&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
                "email": "abc@gmail.com",
                "name": name,
                "age": age,
                "address":["123 Main Street", "Apt 4", "New York, NY 10001"],
            }
        }
    except Exception as e:
        return{
            "message":str(e),
            "status":"error",
            "data":None
        }

# query parameters

@app.get('/users/')
def read_root2(id:str,name:str,age:int):
    try:
        return{
             "status": "success",
            "data": {
                "id": id,
                "profile_url":"https://plus.unsplash.com/premium_photo-1734543932716-431337d9c3c4?q=80&w=2133&auto=format&fit=crop&ixlib=rb-4.0.3&ixid=M3wxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8fA%3D%3D",
                "email": "abc@gmail.com",
                "name": name,
                "age": age,
                "address":["123 Main Street", "Apt 4", "New York, NY 10001"],
            }
        }
    except Exception as e:
        return {
            "message":str(e),
            "status":"error",
            "data":None
        }
        
# Request bodies & # validation in fastapi routes


class Item(BaseModel):
    name:str
    price:float
    description:str=None
    
@app.post("/items/{id}")
def create_item(id:str,item:Item,query:Optional[str]=None):
    try:
        return{
             "status": "success",
            "data":{
                "item":item,
                "id":id,
                "query":query
            }
        }
    except Exception as e:
        return {
            "message":str(e),
            "status":"error",
            "data":None
        }
        

# Grouping routes in Routers
postRouter=APIRouter()

@postRouter.get("/")
def read_posts():
    return{
        "status": "success",
        "data": {
            "posts": [{
              "id":1,
              "title":"Post 1",
              },
                {
              "id":2,
              "title":"Post 2",
              },
                { 
              "id":3,
              "title":"Post 3",
              }
                ]
        }
    }
    
@postRouter.post("/create")
def create_post(post_id:int,userId:str):
        try:
            if post_id:
                return {"message":"Post created successfully"}
            else:
                return {"message":"Post not created"}
        except Exception as e:
            return {"message": str(e)}
@postRouter.delete("/delete")
def delete_post(post_id:int,userId:str):
  # db post
    try:
      if post_id:
        return {"message":"Post deleted successfully"}
      else:
        return {"message":"Post not deleted"}
    except Exception as e:
      return {"message": str(e)}
  
@postRouter.get("/like")
def like_post(post_id:int,userId:str):
  # db post
    try:
      if post_id:
        return {"message":"Post liked successfully"}
      else:
        return {"message":"Post disliked successfully"}
    except Exception as e:
      return {"message": str(e)}
  
@app.get("/")
def read_root():
    return {"message":"Sever is running"}

app.include_router(postRouter,prefix="/posts")