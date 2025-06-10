from fastapi import FastAPI, File, UploadFile, Form
from fastapi.staticfiles import StaticFiles
import shutil
import os

UPLOAD_FOLDER="Uploads"

os.makedirs(UPLOAD_FOLDER,exist_ok=True)
app=FastAPI()

app.mount("/static", StaticFiles(directory=UPLOAD_FOLDER), name="static")

@app.get("/")
def read_root():
    return {"message": "Welcome to the File Upload API!"}

@app.post("/upload/")
async def upload_file(file:UploadFile = File(...)):
    file_location = os.path.join(UPLOAD_FOLDER,file.filename)

    with open(file_location, "wb") as buffer:
        shutil.copyfileobj(file.file,buffer)

        file_url=f"/static/{file.filename}"
        return {"filename": file.filename, "url":file_url}