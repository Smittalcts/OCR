# main2.py
import os
import uuid
import logging
from typing import TypedDict, List, Annotated
from dotenv import load_dotenv
from pydantic import BaseModel, Field

from fastapi import FastAPI, Request, HTTPException, Depends, status
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security import OAuth2PasswordRequestForm

from langchain_core.messages import BaseMessage, HumanMessage
from langchain_core.globals import set_debug

import database
import auth
# Import new modular components
from langgraph_graph import build_graph
from langgraph_models import (
    UserCreate, Token, UserInDB, ScheduleRequest,
    PracticeRequest, AnswerRequest, EndRequest
)

# Removed the large commented-out block of old code

set_debug(True)

# --- ENHANCED LOGGING SETUP ---
logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s')
stream_handler = logging.StreamHandler()
stream_handler.setFormatter(log_formatter)
logger.addHandler(stream_handler)
logger.propagate = False


# --- INITIALIZATION ---
load_dotenv()
app = FastAPI()

@app.on_event("startup")
async def startup_event():
    database.init_db() 

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# This in-memory dict still holds *active* LangGraph states
interview_sessions = {}

# --- BUILD THE GRAPH ---
# The graph is now built and compiled from an external module
app_graph = build_graph()


# --- AUTHENTICATION API ENDPOINTS ---

@app.get("/", response_class=HTMLResponse)
async def get_login_page(request: Request):
    """Serves the login/signup page."""
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/signup", response_class=JSONResponse)
async def signup_user(request: UserCreate):
    """User registration endpoint."""
    logger.info(f"Attempting to create user: {request.username}")
    existing_user = database.get_user_by_username(request.username)
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already registered")
    
    hashed_password = auth.get_password_hash(request.password)
    user = database.create_user(
        username=request.username,
        hashed_password=hashed_password,
        role=request.role
    )
    if not user:
        raise HTTPException(status_code=500, detail="Could not create user")
    
    logger.info(f"User {request.username} created successfully.")
    return {"message": "User created successfully. Please log in."}

@app.post("/token", response_class=JSONResponse)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """Login endpoint, returns a JWT."""
    user = database.get_user_by_username(form_data.username)
    if not user or not auth.verify_password(form_data.password, user['hashed_password']):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token = auth.create_access_token(data={"sub": user['username']})
    return {"access_token": access_token, "token_type": "bearer"}

# --- APPLICATION API ENDPOINTS ---

@app.get("/app", response_class=HTMLResponse)
async def get_root(request: Request):
    """Serves the main application dashboard (formerly index.html)."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/api/users/me", response_model=UserInDB)
async def read_users_me(current_user: dict = Depends(auth.get_current_user)):
    """Returns the current logged-in user's details."""
    return current_user

# --- Manager Endpoints ---
@app.get("/api/manager/dashboard", response_class=JSONResponse)
async def get_manager_dashboard(current_user: dict = Depends(auth.get_current_manager)):
    """Gets dashboard data for a manager: associates and their interview results."""
    associates = database.get_all_associates()
    interview_results = database.get_manager_dashboard_data(current_user['id'])
    return {"associates": associates, "interview_results": interview_results}

@app.post("/api/manager/schedule", response_class=JSONResponse)
async def schedule_interview(request: ScheduleRequest, current_user: dict = Depends(auth.get_current_manager)):
    """Schedules a new interview for an associate."""
    interview_id = str(uuid.uuid4())
    logger.info(f"Manager {current_user['id']} scheduling interview for associate {request.associate_id}")
    interview = database.schedule_interview(
        interview_id=interview_id,
        associate_id=request.associate_id,
        manager_id=current_user['id'],
        topics=request.topics
    )
    return {"message": "Interview scheduled", "interview": interview}

@app.get("/api/manager/interview_transcript/{interview_id}", response_class=JSONResponse)
async def get_interview_transcript(interview_id: str, current_user: dict = Depends(auth.get_current_manager)):
    """NEW: Gets the full Q&A transcript for a specific interview."""
    # TODO: Add check to ensure this manager_id is allowed to view this interview_id
    transcript = database.get_full_interview_transcript(interview_id)
    if not transcript:
        raise HTTPException(status_code=404, detail="Transcript not found.")
    return transcript

# --- Associate Endpoints ---
@app.get("/api/associate/dashboard", response_class=JSONResponse)
async def get_associate_dashboard(current_user: dict = Depends(auth.get_current_user)):
    """Gets dashboard data for an associate: pending and completed interviews."""
    interviews = database.get_associate_interviews(current_user['id'])
    return interviews

@app.post("/api/associate/start_practice", response_class=JSONResponse)
async def start_practice_interview(request: PracticeRequest, current_user: dict = Depends(auth.get_current_user)):
    """NEW: Starts an instant practice interview for the associate."""
    interview_id = str(uuid.uuid4())
    logger.info(f"User {current_user['id']} starting practice interview {interview_id}")
    
    # Schedule it as a "practice" interview, managed by the user themselves
    database.schedule_interview(
        interview_id=interview_id,
        associate_id=current_user['id'],
        manager_id=current_user['id'], # Self-managed
        topics=request.topics
    )
    
    database.set_interview_status(interview_id, 'in_progress')
    
    # Use the initial state structure from the models file
    from langgraph_models import InterviewState
    initial_state = InterviewState(
        session_id=interview_id,
        topics_to_cover=request.topics,
        history=[],
        current_context="",
        current_question="",
        current_expected_answer="",
        current_question_id=None,
        latest_evaluation=None,
        follow_up_count=0, 
        topic_scores=[],
        final_summary="",
        api_call_count=0
    )
    
    final_state = app_graph.invoke(initial_state)
    interview_sessions[interview_id] = final_state
    
    return JSONResponse(content={"session_id": interview_id, "question": final_state.get('current_question')})

# --- INTERVIEW FLOW API ENDPOINTS (Protected) ---
@app.post("/api/interview/start/{interview_id}", response_class=JSONResponse)
async def start_interview(interview_id: str, current_user: dict = Depends(auth.get_current_user)):
    """Starts a specific, scheduled interview."""
    logger.info(f"Attempting to start interview: {interview_id} for user {current_user['id']}")
    
    interview_details = database.get_interview_details(interview_id, current_user['id'])
    
    if not interview_details:
        raise HTTPException(status_code=404, detail="Interview not found or not assigned to this user.")
    
    if interview_details['status'] == 'completed':
        raise HTTPException(status_code=400, detail="This interview has already been completed.")
    
    # Set status to in_progress
    database.set_interview_status(interview_id, 'in_progress')
    
    from langgraph_models import InterviewState
    initial_state = InterviewState(
        session_id=interview_id, 
        topics_to_cover=interview_details['topics'],
        history=[],
        current_context="",
        current_question="",
        current_expected_answer="",
        current_question_id=None,
        latest_evaluation=None,
        follow_up_count=0, 
        topic_scores=[],
        final_summary="",
        api_call_count=0
    )
    
    final_state = app_graph.invoke(initial_state)
    interview_sessions[interview_id] = final_state
    
    return JSONResponse(content={"session_id": interview_id, "question": final_state.get('current_question')})

@app.post("/api/interview/submit", response_class=JSONResponse)
async def submit_answer(request: AnswerRequest, current_user: dict = Depends(auth.get_current_user)):
    """Submits an answer to the currently active interview."""
    interview_id = request.interview_id
    if interview_id not in interview_sessions:
        raise HTTPException(status_code=404, detail="Invalid session ID. The interview may have timed out or not been started.")
    
    if interview_sessions[interview_id].get('session_id') != interview_id:
        raise HTTPException(status_code=403, detail="Session mismatch.")

    current_state = interview_sessions[interview_id]
    current_state["history"].append(HumanMessage(content=request.answer))
    
    final_state = app_graph.invoke(current_state, config={"recursion_limit": 50})
    interview_sessions[interview_id] = final_state
    
    newly_scored_topic = final_state.get("topic_scores")[-1] if final_state.get("topic_scores") and len(final_state.get("topic_scores")) > len(current_state.get("topic_scores")) else None
    
    response_data = {
        "evaluation": final_state.get("latest_evaluation").dict() if final_state.get("latest_evaluation") else None,
        "topic_score": newly_scored_topic.dict() if newly_scored_topic else None,
        "next_question": final_state.get("current_question"),
        "interview_finished": bool(final_state.get("final_summary"))
    }
    return JSONResponse(content=response_data)

@app.post("/api/interview/end", response_class=JSONResponse)
async def end_interview(request: EndRequest, current_user: dict = Depends(auth.get_current_user)):
    """Ends the interview and calculates the final score."""
    session_id = request.interview_id
    if session_id not in interview_sessions:
        raise HTTPException(status_code=404, detail="Invalid session ID.")
        
    current_state = interview_sessions[session_id]
    
    if not current_state.get('final_summary'):
        current_state['topics_to_cover'] = [] 
        final_state = app_graph.invoke(current_state, config={"recursion_limit": 50})
        interview_sessions[session_id] = final_state

    all_topic_scores_from_db = database.get_all_topic_results_for_session(session_id)
    
    if not all_topic_scores_from_db:
        return JSONResponse(content={"final_summary": "Interview ended, but no scores were recorded.", "all_topic_scores": []})

    total_score = sum(ts['final_score'] for ts in all_topic_scores_from_db)
    max_score = len(all_topic_scores_from_db) * 10
    summary = (f"The interview is complete. "
               f"Final Score: {total_score} out of {max_score}.")

    database.update_interview_on_finish(session_id, total_score, max_score, summary)
    
    if session_id in interview_sessions:
        del interview_sessions[session_id]

    return JSONResponse(content={"final_summary": summary, "all_topic_scores": all_topic_scores_from_db})

