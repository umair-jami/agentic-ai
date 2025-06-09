from fastapi import FastAPI, Query, Path
from pydantic import BaseModel, Field, EmailStr, conint
from typing import Optional

app=FastAPI()

# Request body model with validation

class AdmissionForm(BaseModel):
    full_name: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    age: conint(ge=17, le=30)  # Only allow ages between 17 and 30
    degree_program: str = Field(..., min_length=2)
    address: Optional[str] = Field(None, max_length=100)
    
@app.post("/submit_form/{student_id}")
def submit_form(
    student_id: int = Path(..., gt=0, description="Student ID must be a positive integer"),
    form: AdmissionForm = ...,
    apply_scholarship: Optional[bool] = Query(False, description="Apply for scholarship or not"),
):
    try:
        return{
            "status":"success",
            "message": "Form submitted successfully!",
            "data":{
                "student_id": student_id,
                "form_data": form,
                "apply_scholarship": apply_scholarship,
            }
        }
    except Exception as e:
        return{
            "message":str(e),
            "Status":"error",
            "data":None
        }