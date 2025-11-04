# langgraph_models.py
from typing import TypedDict, List, Annotated
from pydantic import BaseModel, Field
from pydantic import BaseModel as V2BaseModel

# --- Pydantic Models (Auth) ---
# Moved here to be used by main.py
class UserCreate(BaseModel):
    username: str
    password: str
    role: str

class Token(BaseModel):
    access_token: str
    token_type: str

class UserInDB(BaseModel):
    id: int
    username: str
    role: str
    class Config:
        from_attributes = True

# --- Pydantic Models (Interview) ---
# Moved here to be used by main.py
class ScheduleRequest(BaseModel):
    associate_id: int
    topics: list[str] = ["Incident Management", "Problem Management", "Change Management"]

class PracticeRequest(BaseModel):
    topics: list[str] = ["Incident Management", "Problem Management", "Change Management"]

class AnswerRequest(BaseModel):
    interview_id: str
    answer: str
    
class EndRequest(BaseModel):
    interview_id: str

# --- LANGGRAPH MODELS ---
class QuestionAndAnswer(V2BaseModel):
    question: str = Field(description="A defination or facts or scenerio based interview question.")
    expected_answer: str = Field(description="A detailed, correct answer to the question, derived from the context (at least 4 lines).")

class Evaluation(V2BaseModel):
    feedback: str = Field(description="Constructive feedback for the user's answer. max 1 line")
    status: str = Field(description="Either 'complete' or 'incomplete'.")
    score: int = Field(description="A score from 0 to 10 for this specific answer.if answer if above 80% correct,give 10")

class TopicScore(V2BaseModel):
    topic: str = Field(description="The main topic being scored.")
    score: int = Field(description="The final score for this topic, from 0 to 10.")
    summary: str = Field(description="A brief summary of the performance on this topic. max 3 sentences")

class TopicSummary(V2BaseModel):
    summary: str = Field(description="A brief summary of the performance on this topic, based on the provided transcript and score. max 3 sentences")

class FollowUpQuestionAndAnswer(V2BaseModel):
    question: str = Field(description="A targeted, 1-line follow-up question about the missing information.")
    expected_answer: str = Field(description="1 line correct answer to the generated follow-up question.")


# --- LANGGRAPH STATE DEFINITION ---
class InterviewState(TypedDict):
    session_id: str # This will be the interview_id
    topics_to_cover: List[str]
    current_topic: str
    current_context: str
    current_question: str
    current_expected_answer: str
    current_question_id: int | None 
    history: Annotated[list, lambda x, y: x + y]
    latest_evaluation: Evaluation
    follow_up_count: int
    topic_scores: List[TopicScore]
    final_summary: str
    api_call_count: Annotated[int, lambda x, y: x + y]
