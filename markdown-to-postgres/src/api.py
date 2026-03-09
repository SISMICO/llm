from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Optional

# --- Pydantic Models ---

class NumbersInput(BaseModel):
    """
    Represents the input for arithmetic operations.
    """
    number1: float
    number2: float

class OperationResult(BaseModel):
    """
    Represents the result of an arithmetic operation.
    """
    result: float

class Person(BaseModel):
    """
    Represents a Person's information.
    """
    id: int
    name: str
    age: int
    email: Optional[str] = None
    occupation: Optional[str] = None

# --- In-memory "Database" for Persons ---
# In a real application, this would be a connection to a database.
fake_person_db: Dict[int, Person] = {
    1: Person(id=1, name="Alice Wonderland", age=30, email="alice.wonderland@example.com", occupation="Dreamer"),
    2: Person(id=2, name="Bob The Builder", age=45, email="bob.builder@example.com", occupation="Constructor"),
    3: Person(id=3, name="Charlie Chaplin", age=55, occupation="Comedian"),
}

# --- FastAPI Application ---
app = FastAPI(
    title="LLM Helper API",
    description="API endpoints for an LLM to perform calculations and retrieve data.",
    version="1.0.0",
)

# --- API Endpoints ---

@app.post("/sum", response_model=OperationResult, tags=["Calculations"])
async def sum_numbers(numbers: NumbersInput):
    """
    Receives two numbers and returns their sum.
    This endpoint is designed to be called by an LLM.
    """
    result = numbers.number1 + numbers.number2
    return OperationResult(result=result)

@app.post("/multiply", response_model=OperationResult, tags=["Calculations"])
async def multiply_numbers(numbers: NumbersInput):
    """
    Receives two numbers and returns their product.
    This endpoint is designed to be called by an LLM.
    """
    result = numbers.number1 * numbers.number2
    return OperationResult(result=result)

@app.get("/person/{person_id}", response_model=Person, tags=["Data Retrieval"])
async def get_person_info(person_id: int):
    """
    Receives an ID and returns the Person's information.
    If the person is not found, it returns a 404 error.
    This endpoint is designed to be called by an LLM.
    """
    person = fake_person_db.get(person_id)
    if not person:
        raise HTTPException(status_code=404, detail=f"Person with ID {person_id} not found.")
    return person

# --- How to run (for development) ---
# Save this code as main.py
# Run in your terminal: uvicorn api:app --app-dir src --reload
#
# You can then access the interactive API documentation (Swagger UI) at:
# http://127.0.0.1:8000/docs
#
# Or ReDoc at:
# http://127.0.0.1:8000/redoc
#
# Example cURL requests:
#
# Sum:
# curl -X POST "http://127.0.0.1:8000/sum" \
# -H "Content-Type: application/json" \
# -d '{"number1": 5, "number2": 3}'
# Expected: {"result":8.0}
#
# Multiply:
# curl -X POST "http://127.0.0.1:8000/multiply" \
# -H "Content-Type: application/json" \
# -d '{"number1": 5, "number2": 3}'
# Expected: {"result":15.0}
#
# Get Person (exists):
# curl -X GET "http://127.0.0.1:8000/person/1"
# Expected: {"id":1,"name":"Alice Wonderland","age":30,"email":"alice.wonderland@example.com","occupation":"Dreamer"}
#
# Get Person (does not exist):
# curl -X GET "http://127.0.0.1:8000/person/99"
# Expected: {"detail":"Person with ID 99 not found."} (with a 404 status code)
