from fastapi import FastAPI
from pydantic import BaseModel

# 1. Initialize the FastAPI app
app = FastAPI()

# 2. Use Pydantic to validate the incoming data
class AnswerPayload(BaseModel):
    student_answer: str
    reference_answer: str

# 3. Create the API Endpoint
@app.post("/grade")
async def grade_answer(payload: AnswerPayload):
    # This is where your PyTorch model would do the scoring
    # score = my_deberta_model.predict(payload.student_answer, payload.reference_answer)
    
    # Simulating the model's output for the API
    simulated_score = 0.75 
    
    return {
        "status": "success",
        "score": simulated_score,
        "message": "Inference complete"
    }
