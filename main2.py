# import os
# import uuid
# import logging
# from typing import TypedDict, List, Annotated
# from dotenv import load_dotenv
# from pydantic import BaseModel, Field

# from fastapi import FastAPI, Request, HTTPException
# from fastapi.responses import HTMLResponse, JSONResponse
# from fastapi.staticfiles import StaticFiles
# from fastapi.templating import Jinja2Templates

# from langgraph.graph import StateGraph, END, START
# from langchain_core.messages import BaseMessage, HumanMessage
# from langchain_openai import AzureChatOpenAI
# from langchain_core.prompts import ChatPromptTemplate
# from pydantic import BaseModel as V2BaseModel
# from langchain_core.globals import set_debug

# from vector_store import VectorStoreManager
# import new_db 
# from new_db import get_scores_for_topic
# set_debug(True)

# # --- ENHANCED LOGGING SETUP ---
# logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)
# log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s')
# stream_handler = logging.StreamHandler()
# stream_handler.setFormatter(log_formatter)
# logger.addHandler(stream_handler)
# logger.propagate = False


# # --- INITIALIZATION ---
# load_dotenv()
# app = FastAPI()

# @app.on_event("startup")
# async def startup_event():
#     new_db.init_db()

# app.mount("/static", StaticFiles(directory="static"), name="static")
# templates = Jinja2Templates(directory="templates")

# interview_sessions = {}

# # --- LLM and Vector Store Setup ---
# try:
#     llm = AzureChatOpenAI(
#         azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
#         api_version="2024-08-01-preview",
#         temperature=0.7,
#         max_tokens=1000,
#     )
#     vector_store_manager = VectorStoreManager(
#         persist_directory="./chroma_db",
#         collection_name="induction_docs"
#     )
#     logger.info("LLM and VectorStoreManager initialized successfully.")
# except Exception as e:
#     logger.error(f"FATAL: Could not initialize core components. Error: {e}", exc_info=True)
#     llm = None
#     vector_store_manager = None

# # --- Pydantic Models ---
# class StartRequest(BaseModel):
#     topics: list[str] = ["Incident Management", "Problem Management", "Change Management"]

# class AnswerRequest(BaseModel):
#     session_id: str
#     answer: str
    
# class EndRequest(BaseModel):
#     session_id: str

# class QuestionAndAnswer(V2BaseModel):
#     question: str = Field(description="A defination or facts or scenerio based interview question.")
#     expected_answer: str = Field(description="A detailed, correct answer to the question, derived from the context (at least 4 lines).")

# class Evaluation(V2BaseModel):
#     feedback: str = Field(description="Constructive feedback for the user's answer. max 1 line")
#     status: str = Field(description="Either 'complete' or 'incomplete'.")
#     score: int = Field(description="A score from 0 to 10 for this specific answer.if answer if above 80% correct,give 10")

# class TopicScore(V2BaseModel):
#     topic: str = Field(description="The main topic being scored.")
#     score: int = Field(description="The final score for this topic, from 0 to 10.")
#     summary: str = Field(description="A brief summary of the performance on this topic. max 3 sentences")

# class TopicSummary(V2BaseModel):
#     summary: str = Field(description="A brief summary of the performance on this topic, based on the provided transcript and score. max 3 sentences")

# class FollowUpQuestionAndAnswer(V2BaseModel):
#     question: str = Field(description="A targeted, 1-line follow-up question about the missing information.")
#     expected_answer: str = Field(description="1 line correct answer to the generated follow-up question.")

# # --- LANGGRAPH STATE DEFINITION ---
# class InterviewState(TypedDict):
#     session_id: str
#     topics_to_cover: List[str]
#     current_topic: str
#     current_context: str
#     current_question: str
#     current_expected_answer: str
#     history: Annotated[list, lambda x, y: x + y]
#     latest_evaluation: Evaluation
#     follow_up_count: int
#     topic_scores: List[TopicScore]
#     final_summary: str
#     api_call_count: Annotated[int, lambda x, y: x + y]

# # --- LANGGRAPH NODES ---

# def retrieve_context(state: InterviewState):
#     logger.info(f"--- Entering NODE: retrieve_context ---")
#     if not state["topics_to_cover"]:
#         logger.warning("No more topics to cover. Routing to finish interview.")
#         return {"final_summary": "Interview is ready to be concluded."}
    
#     topic = state["topics_to_cover"][0]
#     remaining_topics = state["topics_to_cover"][1:]
#     logger.info(f"Retrieving context for topic: '{topic}'")
    
#     search_results = vector_store_manager.advanced_hybrid_search(topic, k=8)
#     docs = search_results if isinstance(search_results, list) else search_results.get("documents", [])
    
#     if not docs:
#         context = f"No specific context found for {topic}."
#         logger.error(f"No context found for topic: '{topic}'. Using fallback.")
#     else:
#         context = "\n\n---\n\n".join([doc.page_content for doc in docs])
#         logger.info(f"Successfully retrieved {len(docs)} chunks, totaling {len(context)} characters.")
    
#     return {
#         "current_context": context,
#         "topics_to_cover": remaining_topics,
#         "current_topic": topic,
#         "follow_up_count": 0
#     }

# def generate_question(state: InterviewState):
#     logger.info(f"--- Entering NODE: generate_question ---")
#     prompt = ChatPromptTemplate.from_messages([
#         ("system", "You are a technical lead. Based ONLY on the context, create one SCENERIO or FACT or DEFINATION based question AND a detailed, correct answer for it (at least 4 lines). Respond ONLY in the requested JSON format."),
#         ("human", "Context:\n{context}")
#     ])

   
    
   
#     # invoke_input = {"context": state["current_context"]}
#     # final_prompt_value = prompt.format_prompt(**invoke_input)
#     # logger.info(f"QUESTION PROMPT:\n{final_prompt_value.to_string()}")
    
   
#     structured_llm = llm.with_structured_output(QuestionAndAnswer)
#     chain = prompt | structured_llm
    
#     qa_pair = chain.invoke({"context": state["current_context"]})
    
#     new_db.add_question(
#         session_id=state["session_id"],
#         topic=state["current_topic"],
#         question=qa_pair.question,
#         expected_answer=qa_pair.expected_answer
#     )
    
#     return {
#         "current_question": qa_pair.question, 
#         "current_expected_answer": qa_pair.expected_answer,
#         "history": state["history"] + [HumanMessage(content=f"AI_QUESTION: {qa_pair.question}")],
#         "api_call_count": 1
#     }

# def evaluate_answer(state: InterviewState):
#     logger.info(f"--- Entering NODE: evaluate_answer ---")
#     topic_history_str = new_db.get_topic_history_as_string(state["session_id"], state["current_topic"])
#     logger.info(f"Retrieved history for evaluation prompt:\n---\n{topic_history_str}\n---")
    
#     prompt = ChatPromptTemplate.from_messages([
#         ("system", """You are a technical lead. Evaluate the engineer's answer against the expected answer.

# 1.  **Provide a score from 0 to 10.**
#     -   Score 10 if the answer is 80% or more correct.
#     -   Score 0 for irrelevant or completely wrong answers.
# 2.  **Set the 'status' field.**
#     -   If the score is 9 or 10, the status MUST be 'complete'.
#     -   If the score is less than 9, the status MUST be 'incomplete'.
# 3.  **Provide 1 line feedback.**

# Respond ONLY in the requested JSON format."""),
#         ("human", "Full Conversation History on '{topic}':\n\n\nExpected Answer for the LAST question: \"{expected_answer}\",\n\nUSER_ANSWER=\"{user_answer}\"")
#     ])

    


#     structured_llm = llm.with_structured_output(Evaluation)
#     eval_chain = prompt | structured_llm
    
#     user_answer = state["history"][-1].content
    
#     # 2. Create ONE dictionary with ALL the keys the prompt needs
#     invoke_input = {
#         "topic": state["current_topic"],
#         # "history": topic_history_str,
#         "expected_answer": state["current_expected_answer"],
#         "user_answer": user_answer
#     }

#     # 3. Use this complete dictionary for logging the prompt
#     # final_prompt_value = prompt.format_prompt(**invoke_input)
#     # logger.info(f"Complete evaluation prompt:\n{final_prompt_value.to_string()}")
    
#     feedback = eval_chain.invoke(invoke_input)
    
#     new_db.add_answer_and_evaluation(
#         session_id=state['session_id'],
#         topic=state['current_topic'],
#         user_answer=user_answer,
#         feedback=feedback.feedback,
#         score=feedback.score,
#         status=feedback.status
#     )
    
#     return {"latest_evaluation": feedback, "api_call_count": 1}

# def generate_follow_up(state: InterviewState):
#     logger.info(f"--- Entering NODE: generate_follow_up ---")
#     topic_history_str = new_db.get_topic_history_as_string(state["session_id"], state["current_topic"])
#     logger.info(f"Retrieved history for follow-up prompt:\n---\n{topic_history_str}\n---")
    
#     prompt = ChatPromptTemplate.from_messages([
#         ("system", "You are a technical lead. The engineer's previous answers were incomplete. Based on the entire conversation history, generate a targeted 1 line follow-up question to probe for missing information, AND provide the specific, concise 1 line answer to that follow-up question. Respond ONLY in the requested JSON format."),
#         ("human", "Conversation History on '{topic}':\n{history}\n\nYour task is to generate the next logical follow-up question and its expected answer.")
#     ])

    
    
#     structured_llm = llm.with_structured_output(FollowUpQuestionAndAnswer)
#     chain = prompt | structured_llm

#      # 1. Create the dictionary with the keys required by THIS prompt
#     invoke_input = {
#         "topic": state["current_topic"],
#         "history": topic_history_str
#     }

#     # 2. Use this dictionary for logging
#     # final_prompt_value = prompt.format_prompt(**invoke_input)
#     # logger.info(f"Complete follow-up prompt:\n{final_prompt_value.to_string()}")

#     # 3. Use the SAME dictionary for the invocation
#     follow_up_qa_pair = chain.invoke(invoke_input)
    
#     new_db.add_question(
#         session_id=state['session_id'],
#         topic=state['current_topic'],
#         question=follow_up_qa_pair.question,
#         expected_answer=follow_up_qa_pair.expected_answer
#     )
    
#     return {
#         "current_question": follow_up_qa_pair.question,
#         "current_expected_answer": follow_up_qa_pair.expected_answer,
#         "history": state["history"] + [HumanMessage(content=f"AI_QUESTION: {follow_up_qa_pair.question}")],
#         "follow_up_count": state["follow_up_count"] + 1,
#         "api_call_count": 1
#     }

# def calculate_topic_score_and_summarize(state: InterviewState):
#     logger.info(f"--- Entering NODE: calculate_topic_score_and_summarize ---")
    
#     session_id = state["session_id"]
#     topic = state["current_topic"]

#     topic_history_str = new_db.get_topic_history_as_string(session_id, topic)
#     logger.info(f"Retrieved history for summary prompt:\n---\n{topic_history_str}\n---")
    
#     scores = get_scores_for_topic(session_id, topic)
#     logger.info(f"Retrieved scores for calculation: {scores}")
    
#     final_score = 0
#     if not scores:
#         final_score = 5 # Default score if something went wrong and no scores are found
#         logger.warning(f"No scores found for topic '{topic}', defaulting to 5.")
#     else:
#         # --- MODIFIED LOGIC BLOCK START ---

#         main_score = scores[0]
#         follow_up_scores = scores[1:]

#         # Condition 1: Perfect score on the first try, no follow-ups needed.
#         if main_score >= 9 and not follow_up_scores:
#             final_score = 10
        
#         # NEW Condition 2: The user scored >= 8 on ANY follow-up question, showing strong recovery.
#         elif follow_up_scores and any(s >= 8 for s in follow_up_scores):
#             final_score = 10
#             logger.info(f"High score on a follow-up detected. Overriding final score to 10.")

#         # Condition 3: User did very poorly on all follow-ups, so they get no credit for them.
#         elif follow_up_scores and all(s <= 2 for s in follow_up_scores):
#             final_score = main_score
        
#         # Condition 4: User had follow-ups with mixed results, calculate a weighted average.
#         elif follow_up_scores:
#             avg_follow_up = sum(follow_up_scores) / len(follow_up_scores)
#             # Weighted average: 70% for the initial answer, 30% for the follow-up average
#             final_score = round((main_score * 0.7) + (avg_follow_up * 0.3))
        
#         # Condition 5: Default case if there were no follow-ups (and score wasn't >= 9).
#         else:
#             final_score = main_score

#         # --- MODIFIED LOGIC BLOCK END ---

#     final_score = min(10, max(0, final_score))
#     logger.info(f"Final calculated score for topic '{topic}' is: {final_score}")

#     summary_prompt = ChatPromptTemplate.from_messages([
#         ("system", "You are a Technical Lead. Based on the provided transcript and the final calculated score, write a brief, objective summary of the user's performance on this topic.MAX 2 LINES. Respond ONLY in the requested JSON format."),
#         ("human", "Topic: {topic}\nFinal Calculated Score: {score}/10\n\nFull Transcript:\n{history}")
#     ])

#     print(f"SUMMARY PROMPT : ",summary_prompt)
    
#     structured_llm = llm.with_structured_output(TopicSummary)
#     summary_chain = summary_prompt | structured_llm
    
#     summary_obj = summary_chain.invoke({
#         "topic": topic,
#         "score": final_score,
#         "history": topic_history_str
#     })
    
#     final_topic_score = TopicScore(
#         topic=topic,
#         score=final_score,
#         summary=summary_obj.summary
#     )

#     new_db.add_topic_result(
#         session_id=session_id,
#         topic=final_topic_score.topic,
#         score=final_topic_score.score,
#         summary=final_topic_score.summary
#     )

#     return {"topic_scores": state["topic_scores"] + [final_topic_score], "api_call_count": 1}

# def finish_interview(state: InterviewState):
#     logger.info(f"--- Entering NODE: finish_interview ---")
    
#     final_topic_scores = new_db.get_all_topic_results_for_session(state["session_id"])
    
#     if not final_topic_scores:
#         logger.warning(f"No topic results found in DB for session {state['session_id']}. Final score is 0.")
#         total_score = 0
#         max_score = 0
#     else:
#         total_score = sum(topic['final_score'] for topic in final_topic_scores)
#         max_score = len(final_topic_scores) * 10

#     summary = f"Interview complete. Final Score: {total_score} out of {max_score}."
#     logger.info(summary)
#     logger.info(f"Total API calls made during this session: {state.get('api_call_count', 0)}")
#     return {"final_summary": summary}

# # --- LANGGRAPH ROUTERS AND EDGES ---

# def entry_point_condition(state: InterviewState):
#     logger.info(f"--- Router: entry_point_condition ---")
#     if state["history"] and not state["history"][-1].content.startswith("AI_QUESTION:"):
#         return "evaluate_answer"
#     else:
#         return "retrieve_context"

# def after_evaluation_condition(state: InterviewState):
#     logger.info(f"--- Router: after_evaluation_condition ---")
#     evaluation = state["latest_evaluation"]
    
#     # If the score is high enough (pass), always summarize.
#     if evaluation.score >= 9:
#         logger.info(f"--> Decision: Score is {evaluation.score}/10. Sufficiently high. Route to summarize topic.")
#         return "calculate_topic_score_and_summarize"
        
#     # If the score is not high enough AND we haven't exhausted our follow-ups, ask another question.
#     if state["follow_up_count"] < 2:
#         logger.info(f"--> Decision: Score is low ({evaluation.score}/10) and follow-up count ({state['follow_up_count']}) < 2. Route to generate follow-up.")
#         return "generate_follow_up"
        
#     # Otherwise, we must summarize because we're out of follow-up attempts.
#     else:
#         logger.info(f"--> Decision: Score is low ({evaluation.score}/10) but follow-up limit reached ({state['follow_up_count']}). Route to summarize topic.")
#         return "calculate_topic_score_and_summarize"

# def after_summary_condition(state: InterviewState):
#     logger.info(f"--- Router: after_summary_condition ---")
#     if not state["topics_to_cover"]:
#         return "finish_interview"
#     else:
#         return "retrieve_context"

# # --- BUILD THE GRAPH ---
# workflow = StateGraph(InterviewState)

# workflow.add_node("retrieve_context", retrieve_context)
# workflow.add_node("generate_question", generate_question)
# workflow.add_node("evaluate_answer", evaluate_answer)
# workflow.add_node("generate_follow_up", generate_follow_up)
# workflow.add_node("calculate_topic_score_and_summarize", calculate_topic_score_and_summarize)
# workflow.add_node("finish_interview", finish_interview)

# workflow.add_conditional_edges(START, entry_point_condition)
# workflow.add_edge("retrieve_context", "generate_question")
# workflow.add_edge("generate_question", END)
# workflow.add_edge("generate_follow_up", END)
# workflow.add_edge("finish_interview", END)
# workflow.add_conditional_edges("evaluate_answer", after_evaluation_condition)
# workflow.add_conditional_edges("calculate_topic_score_and_summarize", after_summary_condition)

# app_graph = workflow.compile()

# # --- API ENDPOINTS ---
# @app.get("/", response_class=HTMLResponse)
# async def get_root(request: Request):
#     return templates.TemplateResponse("index.html", {"request": request})

# @app.post("/start_interview")
# async def start_interview(request: StartRequest):
#     session_id = str(uuid.uuid4())
#     logger.info(f"Starting new interview session: {session_id}")
    
#     initial_state = InterviewState(
#         session_id=session_id,
#         topics_to_cover=request.topics, history=[], current_context="", current_question="",
#         current_expected_answer="", latest_evaluation=None, follow_up_count=0, 
#         topic_scores=[], final_summary="", api_call_count=0
#     )
#     final_state = app_graph.invoke(initial_state)
#     interview_sessions[session_id] = final_state
#     return JSONResponse(content={"session_id": session_id, "question": final_state.get('current_question')})

# @app.post("/submit_answer")
# async def submit_answer(request: AnswerRequest):
#     session_id = request.session_id
#     if session_id not in interview_sessions:
#         raise HTTPException(status_code=404, detail="Invalid session ID.")

#     current_state = interview_sessions[session_id]
#     current_state["history"].append(HumanMessage(content=request.answer))
    
#     final_state = app_graph.invoke(current_state, config={"recursion_limit": 50})
#     interview_sessions[session_id] = final_state
    
#     newly_scored_topic = final_state.get("topic_scores")[-1] if final_state.get("topic_scores") and len(final_state.get("topic_scores")) > len(current_state.get("topic_scores")) else None
    
#     response_data = {
#         "evaluation": final_state.get("latest_evaluation").dict() if final_state.get("latest_evaluation") else None,
#         "topic_score": newly_scored_topic.dict() if newly_scored_topic else None,
#         "next_question": final_state.get("current_question"),
#         "interview_finished": bool(final_state.get("final_summary"))
#     }
#     return JSONResponse(content=response_data)

# @app.post("/end_interview")
# async def end_interview(request: EndRequest):
#     session_id = request.session_id
#     if session_id not in interview_sessions:
#         raise HTTPException(status_code=404, detail="Invalid session ID.")
        
#     current_state = interview_sessions[session_id]
    
#     if not current_state.get('final_summary'):
#         current_state['topics_to_cover'] = [] 
#         final_state = app_graph.invoke(current_state, config={"recursion_limit": 50})
#         interview_sessions[session_id] = final_state

#     all_topic_scores_from_db = new_db.get_all_topic_results_for_session(session_id)
    
#     total_score = sum(ts['final_score'] for ts in all_topic_scores_from_db)
#     max_score = len(all_topic_scores_from_db) * 10
#     summary = (f"The interview is complete. "
#                f"Final Score: {total_score} out of {max_score}.")

#     return JSONResponse(content={"final_summary": summary, "all_topic_scores": all_topic_scores_from_db})










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

from langgraph.graph import StateGraph, END, START
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel as V2BaseModel
from langchain_core.globals import set_debug

from vector_store import VectorStoreManager
import database  # <-- Import new database module
import auth      # <-- Import new auth module

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
    database.init_db() # <-- Use new database init

app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# This in-memory dict still holds *active* LangGraph states
interview_sessions = {}

# --- LLM and Vector Store Setup ---
try:
    llm = AzureChatOpenAI(
        azure_deployment=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
        api_version="2024-08-01-preview",
        temperature=0.7,
        max_tokens=1000,
    )
    vector_store_manager = VectorStoreManager(
        persist_directory="./chroma_db",
        collection_name="induction_docs"
    )
    logger.info("LLM and VectorStoreManager initialized successfully.")
except Exception as e:
    logger.error(f"FATAL: Could not initialize core components. Error: {e}", exc_info=True)
    llm = None
    vector_store_manager = None

# --- Pydantic Models (Auth) ---
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
class ScheduleRequest(BaseModel):
    associate_id: int
    topics: list[str] = ["Incident Management", "Problem Management", "Change Management"]

class AnswerRequest(BaseModel):
    interview_id: str
    answer: str
    
class EndRequest(BaseModel):
    interview_id: str

# --- LANGGRAPH MODELS (Unchanged from original) ---
class QuestionAndAnswer(V2BaseModel):
    question: str = Field(description="A defination or facts or scenerio based interview question.")
    expected_answer: str = Field(description="A detailed, correct answer to the question, derived from the context (at least 4 lines).")
# ... (Evaluation, TopicScore, TopicSummary, FollowUpQuestionAndAnswer models are identical to your original)
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
    history: Annotated[list, lambda x, y: x + y]
    latest_evaluation: Evaluation
    follow_up_count: int
    topic_scores: List[TopicScore]
    final_summary: str
    api_call_count: Annotated[int, lambda x, y: x + y]

# --- LANGGRAPH NODES (Updated to use `database` module) ---

def retrieve_context(state: InterviewState):
    logger.info(f"--- Entering NODE: retrieve_context ---")
    if not state["topics_to_cover"]:
        logger.warning("No more topics to cover. Routing to finish interview.")
        return {"final_summary": "Interview is ready to be concluded."}
    
    topic = state["topics_to_cover"][0]
    remaining_topics = state["topics_to_cover"][1:]
    logger.info(f"Retrieving context for topic: '{topic}'")
    
    search_results = vector_store_manager.advanced_hybrid_search(topic, k=8)
    docs = search_results if isinstance(search_results, list) else search_results.get("documents", [])
    
    if not docs:
        context = f"No specific context found for {topic}."
        logger.error(f"No context found for topic: '{topic}'. Using fallback.")
    else:
        context = "\n\n---\n\n".join([doc.page_content for doc in docs])
        logger.info(f"Successfully retrieved {len(docs)} chunks, totaling {len(context)} characters.")
    
    return {
        "current_context": context,
        "topics_to_cover": remaining_topics,
        "current_topic": topic,
        "follow_up_count": 0
    }

def generate_question(state: InterviewState):
    logger.info(f"--- Entering NODE: generate_question ---")
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a technical lead. Based ONLY on the context, create one SCENERIO or FACT or DEFINATION based question AND a detailed, correct answer for it (at least 4 lines). Respond ONLY in the requested JSON format."),
        ("human", "Context:\n{context}")
    ])
    
    structured_llm = llm.with_structured_output(QuestionAndAnswer)
    chain = prompt | structured_llm
    
    qa_pair = chain.invoke({"context": state["current_context"]})
    
    database.add_question( # <-- Use database module
        session_id=state["session_id"],
        topic=state["current_topic"],
        question=qa_pair.question,
        expected_answer=qa_pair.expected_answer
    )
    
    return {
        "current_question": qa_pair.question, 
        "current_expected_answer": qa_pair.expected_answer,
        "history": state["history"] + [HumanMessage(content=f"AI_QUESTION: {qa_pair.question}")],
        "api_call_count": 1
    }

def evaluate_answer(state: InterviewState):
    logger.info(f"--- Entering NODE: evaluate_answer ---")
    topic_history_str = database.get_topic_history_as_string(state["session_id"], state["current_topic"]) # <-- Use database
    logger.info(f"Retrieved history for evaluation prompt:\n---\n{topic_history_str}\n---")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a technical lead. Evaluate the engineer's answer against the expected answer.

1.  **Provide a score from 0 to 10.**
    -   Score 10 if the answer is 80% or more correct.
    -   Score 0 for irrelevant or completely wrong answers.
2.  **Set the 'status' field.**
    -   If the score is 9 or 10, the status MUST be 'complete'.
    -   If the score is less than 9, the status MUST be 'incomplete'.
3.  **Provide 1 line feedback.**

Respond ONLY in the requested JSON format."""),
        ("human", "Full Conversation History on '{topic}':\n\n\nExpected Answer for the LAST question: \"{expected_answer}\",\n\nUSER_ANSWER=\"{user_answer}\"")
    ])

    structured_llm = llm.with_structured_output(Evaluation)
    eval_chain = prompt | structured_llm
    
    user_answer = state["history"][-1].content
    
    invoke_input = {
        "topic": state["current_topic"],
        "expected_answer": state["current_expected_answer"],
        "user_answer": user_answer
    }
    
    feedback = eval_chain.invoke(invoke_input)
    
    database.add_answer_and_evaluation( # <-- Use database
        session_id=state['session_id'],
        topic=state['current_topic'],
        user_answer=user_answer,
        feedback=feedback.feedback,
        score=feedback.score,
        status=feedback.status
    )
    
    return {"latest_evaluation": feedback, "api_call_count": 1}

def generate_follow_up(state: InterviewState):
    logger.info(f"--- Entering NODE: generate_follow_up ---")
    topic_history_str = database.get_topic_history_as_string(state["session_id"], state["current_topic"]) # <-- Use database
    logger.info(f"Retrieved history for follow-up prompt:\n---\n{topic_history_str}\n---")
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a technical lead. The engineer's previous answers were incomplete. Based on the entire conversation history, generate a targeted 1 line follow-up question to probe for missing information, AND provide the specific, concise 1 line answer to that follow-up question. Respond ONLY in the requested JSON format."),
        ("human", "Conversation History on '{topic}':\n{history}\n\nYour task is to generate the next logical follow-up question and its expected answer.")
    ])
    
    structured_llm = llm.with_structured_output(FollowUpQuestionAndAnswer)
    chain = prompt | structured_llm

    invoke_input = {
        "topic": state["current_topic"],
        "history": topic_history_str
    }

    follow_up_qa_pair = chain.invoke(invoke_input)
    
    database.add_question( # <-- Use database
        session_id=state['session_id'],
        topic=state['current_topic'],
        question=follow_up_qa_pair.question,
        expected_answer=follow_up_qa_pair.expected_answer
    )
    
    return {
        "current_question": follow_up_qa_pair.question,
        "current_expected_answer": follow_up_qa_pair.expected_answer,
        "history": state["history"] + [HumanMessage(content=f"AI_QUESTION: {follow_up_qa_pair.question}")],
        "follow_up_count": state["follow_up_count"] + 1,
        "api_call_count": 1
    }

def calculate_topic_score_and_summarize(state: InterviewState):
    logger.info(f"--- Entering NODE: calculate_topic_score_and_summarize ---")
    
    session_id = state["session_id"]
    topic = state["current_topic"]

    topic_history_str = database.get_topic_history_as_string(session_id, topic) # <-- Use database
    logger.info(f"Retrieved history for summary prompt:\n---\n{topic_history_str}\n---")
    
    scores = database.get_scores_for_topic(session_id, topic) # <-- Use database
    logger.info(f"Retrieved scores for calculation: {scores}")
    
    final_score = 0
    if not scores:
        final_score = 5 # Default score if something went wrong
        logger.warning(f"No scores found for topic '{topic}', defaulting to 5.")
    else:
        # --- MODIFIED LOGIC BLOCK START ---
        main_score = scores[0]
        follow_up_scores = scores[1:]
        if main_score >= 9 and not follow_up_scores:
            final_score = 10
        elif follow_up_scores and any(s >= 8 for s in follow_up_scores):
            final_score = 10
            logger.info(f"High score on a follow-up detected. Overriding final score to 10.")
        elif follow_up_scores and all(s <= 2 for s in follow_up_scores):
            final_score = main_score
        elif follow_up_scores:
            avg_follow_up = sum(follow_up_scores) / len(follow_up_scores)
            final_score = round((main_score * 0.7) + (avg_follow_up * 0.3))
        else:
            final_score = main_score
        # --- MODIFIED LOGIC BLOCK END ---

    final_score = min(10, max(0, final_score))
    logger.info(f"Final calculated score for topic '{topic}' is: {final_score}")

    summary_prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a Technical Lead. Based on the provided transcript and the final calculated score, write a brief, objective summary of the user's performance on this topic.MAX 2 LINES. Respond ONLY in the requested JSON format."),
        ("human", "Topic: {topic}\nFinal Calculated Score: {score}/10\n\nFull Transcript:\n{history}")
    ])
    
    structured_llm = llm.with_structured_output(TopicSummary)
    summary_chain = summary_prompt | structured_llm
    
    summary_obj = summary_chain.invoke({
        "topic": topic,
        "score": final_score,
        "history": topic_history_str
    })
    
    final_topic_score = TopicScore(
        topic=topic,
        score=final_score,
        summary=summary_obj.summary
    )

    database.add_topic_result( # <-- Use database
        session_id=session_id,
        topic=final_topic_score.topic,
        score=final_topic_score.score,
        summary=final_topic_score.summary
    )

    return {"topic_scores": state["topic_scores"] + [final_topic_score], "api_call_count": 1}

def finish_interview(state: InterviewState):
    logger.info(f"--- Entering NODE: finish_interview ---")
    
    session_id = state['session_id']
    final_topic_scores = database.get_all_topic_results_for_session(session_id) # <-- Use database
    
    if not final_topic_scores:
        logger.warning(f"No topic results found in DB for session {session_id}. Final score is 0.")
        total_score = 0
        max_score = 0
    else:
        total_score = sum(topic['final_score'] for topic in final_topic_scores)
        max_score = len(final_topic_scores) * 10

    summary = f"Interview complete. Final Score: {total_score} out of {max_score}."
    
    # --- IMPORTANT: Persist final score to the interviews table ---
    database.update_interview_on_finish( # <-- New call to save final score
        interview_id=session_id,
        total_score=total_score,
        max_score=max_score,
        summary=summary
    )
    
    logger.info(summary)
    logger.info(f"Total API calls made during this session: {state.get('api_call_count', 0)}")
    return {"final_summary": summary}

# --- LANGGRAPH ROUTERS AND EDGES (Unchanged from original) ---

def entry_point_condition(state: InterviewState):
    logger.info(f"--- Router: entry_point_condition ---")
    if state["history"] and not state["history"][-1].content.startswith("AI_QUESTION:"):
        return "evaluate_answer"
    else:
        return "retrieve_context"
def after_evaluation_condition(state: InterviewState):
    logger.info(f"--- Router: after_evaluation_condition ---")
    evaluation = state["latest_evaluation"]
    if evaluation.score >= 9:
        logger.info(f"--> Decision: Score is {evaluation.score}/10. Sufficiently high. Route to summarize topic.")
        return "calculate_topic_score_and_summarize"
    if state["follow_up_count"] < 2:
        logger.info(f"--> Decision: Score is low ({evaluation.score}/10) and follow-up count ({state['follow_up_count']}) < 2. Route to generate follow-up.")
        return "generate_follow_up"
    else:
        logger.info(f"--> Decision: Score is low ({evaluation.score}/10) but follow-up limit reached ({state['follow_up_count']}). Route to summarize topic.")
        return "calculate_topic_score_and_summarize"
def after_summary_condition(state: InterviewState):
    logger.info(f"--- Router: after_summary_condition ---")
    if not state["topics_to_cover"]:
        return "finish_interview"
    else:
        return "retrieve_context"

# --- BUILD THE GRAPH (Unchanged from original) ---
workflow = StateGraph(InterviewState)
workflow.add_node("retrieve_context", retrieve_context)
workflow.add_node("generate_question", generate_question)
workflow.add_node("evaluate_answer", evaluate_answer)
workflow.add_node("generate_follow_up", generate_follow_up)
workflow.add_node("calculate_topic_score_and_summarize", calculate_topic_score_and_summarize)
workflow.add_node("finish_interview", finish_interview)
workflow.add_conditional_edges(START, entry_point_condition)
workflow.add_edge("retrieve_context", "generate_question")
workflow.add_edge("generate_question", END)
workflow.add_edge("generate_follow_up", END)
workflow.add_edge("finish_interview", END)
workflow.add_conditional_edges("evaluate_answer", after_evaluation_condition)
workflow.add_conditional_edges("calculate_topic_score_and_summarize", after_summary_condition)
app_graph = workflow.compile()

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

# --- Associate Endpoints ---
@app.get("/api/associate/dashboard", response_class=JSONResponse)
async def get_associate_dashboard(current_user: dict = Depends(auth.get_current_user)):
    """Gets dashboard data for an associate: pending and completed interviews."""
    interviews = database.get_associate_interviews(current_user['id'])
    return interviews

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
    
    initial_state = InterviewState(
        session_id=interview_id, # Use the scheduled interview_id as the session_id
        topics_to_cover=interview_details['topics'],
        history=[],
        current_context="",
        current_question="",
        current_expected_answer="",
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
    
    # Simple check to ensure the session belongs to the user
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
    
    # If not already finished, force finish
    if not current_state.get('final_summary'):
        current_state['topics_to_cover'] = [] # Clear topics to force summary
        final_state = app_graph.invoke(current_state, config={"recursion_limit": 50})
        interview_sessions[session_id] = final_state

    all_topic_scores_from_db = database.get_all_topic_results_for_session(session_id)
    
    if not all_topic_scores_from_db:
        return JSONResponse(content={"final_summary": "Interview ended, but no scores were recorded.", "all_topic_scores": []})

    total_score = sum(ts['final_score'] for ts in all_topic_scores_from_db)
    max_score = len(all_topic_scores_from_db) * 10
    summary = (f"The interview is complete. "
               f"Final Score: {total_score} out of {max_score}.")

    # The `finish_interview` node already saved this, but this is a good fallback.
    database.update_interview_on_finish(session_id, total_score, max_score, summary)
    
    # Clean up the in-memory session
    if session_id in interview_sessions:
        del interview_sessions[session_id]

    return JSONResponse(content={"final_summary": summary, "all_topic_scores": all_topic_scores_from_db})