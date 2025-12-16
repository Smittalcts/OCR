import os
import uvicorn
import logging
import sys
import json
import operator
import bcrypt 
from datetime import datetime, timedelta
import httpx
from typing import List, Dict, Any, Optional, Annotated, Literal
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, Field
from jose import JWTError, jwt

# --- LangChain & LangGraph Imports ---
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict
from langchain_core.globals import set_debug
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.outputs import ChatResult, ChatGeneration
from urllib.parse import urlparse, urlunparse
# --- Database Import ---
import simple_db

# --- Prompts Import ---
import prompts

# --- Environment Setup ---
load_dotenv()

# Configure standard logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    stream=sys.stdout,
)
sys_logger = logging.getLogger("InterviewLogger")
set_debug(True)

def check_env_vars():
    required_vars = ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "OPENAI_API_VERSION"]
    if any(not os.environ.get(var) for var in required_vars):
        sys_logger.error("FATAL ERROR: Missing Azure environment variables.")
        exit(1)
check_env_vars()

# ==============================================================================
# AUTH CONFIGURATION
# ==============================================================================
SECRET_KEY = "YOUR_SUPER_SECRET_KEY_CHANGE_THIS" 
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

class Token(BaseModel):
    access_token: str
    token_type: str
    role: str

class UserLogin(BaseModel):
    username: str
    password: str

# ==============================================================================
# 1. HIGH-FIDELITY LOGGER
# ==============================================================================
class InterviewLogger:
    @staticmethod
    def _format_json(data: Any) -> str:
        try:
            return json.dumps(data, indent=2, default=str)
        except Exception:
            return str(data)

    @staticmethod
    def _format_messages(messages: List[BaseMessage]) -> str:
        logs = []
        for m in messages:
            content = m.content
            if len(content) > 500:
                content = content[:500] + "... [TRUNCATED]"
            logs.append(f"[{m.type.upper()}]: {content}")
        return "\n".join(logs)

    @staticmethod
    def log_node_exec(node_name: str, inputs: Dict[str, Any], prompt_data: Optional[Any] = None, output: Optional[Dict[str, Any]] = None):
        separator = "=" * 80
        timestamp = datetime.now().strftime("%H:%M:%S")
        
        log_buffer = [
            f"\n{separator}", 
            f"🔵 NODE: {node_name.upper()} | TIME: {timestamp}", 
            separator
        ]
        
        # Filter noise from input state
        keys_to_log = [
            "current_decision", "candidate_answer", 
            "retrieval_attempts", "is_context_valid", "current_search_query",
            "last_turn_feedback"
        ]
        filtered_input = {k: inputs.get(k) for k in keys_to_log if k in inputs and inputs[k] is not None}
        
        if filtered_input:
            log_buffer.append(f"\n📋 INPUT STATE:\n{InterviewLogger._format_json(filtered_input)}")
        
        if prompt_data:
            log_buffer.append(f"\n🧠 LLM PROMPT:\n")
            if isinstance(prompt_data, list):
                log_buffer.append(InterviewLogger._format_messages(prompt_data))
            else:
                log_buffer.append(str(prompt_data))

        if output:
            clean_output = {k: v for k, v in output.items() if k != "message_history"}
            log_buffer.append(f"\n✅ OUTPUT:\n{InterviewLogger._format_json(clean_output)}")
        
        log_buffer.append(separator + "\n")
        sys_logger.info("\n".join(log_buffer))

logger = InterviewLogger()

# ==============================================================================
# 2. IMPROVED DATA MODELS
# ==============================================================================

class TopicDecision(BaseModel):
    action: Literal["ask_question", "end_interview"]
    topic: Optional[str] = Field(None)
    sub_topic: Optional[str] = Field(None)
    difficulty: Literal["standard", "deep_dive", "fundamental"] = Field("standard")
    reasoning: str = Field(...)
    # Context Flags
    is_last_in_topic: bool = Field(False)
    is_new_main_topic: bool = Field(False)
    is_first_turn: bool = Field(False)

class GeneratedQuestion(BaseModel):
    conversational_entry: str = Field(..., description="The bridge to previous feeback and next question/topic")
    technical_question: str = Field(..., description="The actual interview question.")
    expected_answer: str
    question_source: Literal["sample_question", "generated_from_context"] = Field(..., description="Source of the question.")

class EvaluationResult(BaseModel):
    score: int
    evaluation: str
    missed_points: List[str]
    feedback_for_next_node: str
    decision: Literal["accepted", "needs_probe"] 

class FinalSummary(BaseModel):
    overall_rating: Literal["aced", "passed", "failed"]
    summary_text: str
    key_takeaways: List[str]

class RelevanceCheck(BaseModel):
    status: Literal["sufficient", "partial", "irrelevant"]
    reason: str = Field(description="MAX 1 Line")
    refined_context: Optional[str]
    search_query: Optional[str]

# ==============================================================================
# 3. STATE & SETUP
# ==============================================================================

class PerformanceTracker(TypedDict):
    score: int
    topic: str

class InterviewState(TypedDict):
    session_id: str
    topic_plan: Dict[str, str] 
    current_subtopic_index: int
    is_interview_over: bool
    message_history: List[BaseMessage]
    
    current_decision: Optional[TopicDecision]
    
    # Retrieval Loop
    retrieved_context: Optional[str]
    refined_context: Optional[str]
    current_search_query: Optional[str]
    retrieval_attempts: int
    is_context_valid: bool
    
    candidate_answer: Optional[str]
    
    last_question: Optional[str]
    last_expected_answer: Optional[str]
    last_turn_feedback: Optional[str] 
    
    graded_answers: Annotated[List[PerformanceTracker], operator.add]
    consecutive_low_scores: int
    
    agent_message: Optional[str]
    final_summary: Optional[Dict[str, Any]]

    is_in_probe_mode: bool           # Are we currently waiting for a follow-up?
    partial_answer_context: str      # The user's first incomplete answer
    original_question: str           # The question that started this loop
    original_expected_answer: str    # The full answer we are looking for
    last_missed_points: List[str]

# Load Topics
TOPIC_DATA = {}
try:
    with open("topics.json", 'r') as f:
        TOPIC_DATA = json.load(f)
except Exception as e:
    sys_logger.error(f"Could not load topics.json: {e}")
    # Fallback default topics if file missing
    TOPIC_DATA = {
        "Incident Management": {"sub_topics": ["Definition", "Process"], "sample_questions": ["What is an incident?"]},
    }

simple_db.init_db()

# LLM Setup
llm_client = AzureChatOpenAI(
    azure_deployment=os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
    api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"), 
    api_version="2024-08-01-preview", 
)

# Chroma Setup
PERSIST_DIRECTORY = "./chroma_db"
retriever = None
try:
    embeddings_model = AzureOpenAIEmbeddings(
        openai_api_version="2024-08-01-preview",
        azure_deployment=os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
        api_key=os.environ.get("EMBEDDING_API_KEY"),
        azure_endpoint=os.environ.get("EMBEDDING_END_POINT")
    )
    vector_store = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings_model)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 6})
    sys_logger.info(f"Loaded Chroma DB.")
except Exception as e:
    sys_logger.error(f"Error initializing Chroma: {e}")

async def retrieve_context_tool(query: str) -> str:
    if not retriever: return "Retriever unavailable."
    try:
        docs = await retriever.ainvoke(query)
        return "\n\n".join([d.page_content for d in docs]) or "No data found."
    except Exception as e:
        return f"Search Error: {e}"

# Define chains
llm_eval = llm_client.with_structured_output(EvaluationResult)
llm_gen = llm_client.with_structured_output(GeneratedQuestion)
llm_sum = llm_client.with_structured_output(FinalSummary)
llm_check = llm_client.with_structured_output(RelevanceCheck)

# ==============================================================================
# 5. NODES 
# ==============================================================================

async def node_evaluate(state: InterviewState) -> Dict[str, Any]:
    if state.get("is_in_probe_mode"):
        full_answer = f"{state['partial_answer_context']} . Follow-up clarification: {state['candidate_answer']}"
        question = state['original_question']
        expected = state['original_expected_answer']
        forced_decision = True 
    else:
        full_answer = state['candidate_answer']
        question = state.get('last_question', "Intro")
        expected = state.get('last_expected_answer', "N/A")
        forced_decision = False

    context = state.get('refined_context') or state.get('retrieved_context', "N/A")
    
    user_msg_content = prompts.EVALUATOR_USER_TEMPLATE.format(
        question=question, answer=full_answer, expected=expected, context=context
    )
    msgs = [prompts.EVALUATOR_SYSTEM, HumanMessage(content=user_msg_content)]
    eval_result = await llm_eval.ainvoke(msgs)
    
    output = {}
    
    if eval_result.decision == "needs_probe" and not forced_decision:
        output = {
            "is_in_probe_mode": True,
            "partial_answer_context": full_answer,
            "original_question": question,
            "original_expected_answer": expected,
            "last_missed_points": eval_result.missed_points, 
        }
        sys_logger.info(f"🧐 EVALUATOR: Partial answer detected. Triggering Probe.")
    else:
        simple_db.log_turn_data(
            session_id=state['session_id'],
            topic=state['current_decision'].topic,
            sub_topic=state['current_decision'].sub_topic,
            question=question,
            expected_answer=expected,
            user_answer=full_answer, 
            score=eval_result.score,
            evaluation_feedback=eval_result.evaluation,
            metadata=eval_result.model_dump()
        )
        
        output = {
            "is_in_probe_mode": False, 
            "graded_answers": [{"score": eval_result.score, "topic": state['current_decision'].topic}],
            "last_turn_feedback": eval_result.feedback_for_next_node,
            "message_history": state.get('message_history', []) + [HumanMessage(content=state['candidate_answer'])]
        }
        sys_logger.info(f"✅ EVALUATOR: Final Score {eval_result.score}")

    return output

async def node_ask_followup(state: InterviewState) -> Dict[str, Any]:
    missed = state.get("last_missed_points", [])
    original_q = state.get("original_question")
    original_expected = state.get("original_expected_answer")
    user_partial = state.get("partial_answer_context")

    prompt = f"""
    ROLE: OnCall Support Services Interviewer.
    GOAL: Ask a follow-up question to the partial answer given by user for the user to know what all points are expected in the expected answer other then the answer given.
    
    CONTEXT:
    - Question Asked: "{original_q}"
    - User's Partial Answer: "{user_partial}"
    - The Missing Truth (Expected): "{original_expected}"
    - Specific Missed Concepts: {missed}

    ||  ""THE MOST IMPORTANT REASON FOR A FOLLOW UP IS TO HELP USER COVER THE MISSED POINTS OF EXPECTED ANSWER THAT THEY UNKNOWNINGLY MISSED""  ||
    
    INSTRUCTIONS:
    - the follow up question should be MAX 1-2 line. question should be framed in a way that its answer should be 1 word pointing to the missed point or one liner or 2 liner answer expected at MAX. 
    - check for if main question particularly asked for this missed point or user missed it and acknowledge accordingly.
    - make sure the follow up question doesnot contain answer in itself 
    SITUATION 1:
    - user answered almost correctly and missed 1 major point,ask a brief follow up to test if user can answer a simple question related to that point.compare user's answer to expected answer.
    SITUATION 2:
    - user missed many major points,ask if he understood the main question? + ask a follow up to test whether user can answer these topics with a difficult situation based question.compare user answer to expected answer.
    SITUATION 3:
    - user gave completely irrelevant answer, ask if user understood the question + pick one simple point missed and ask a simple follow up question.
    """
    
    msg = await llm_client.ainvoke([SystemMessage(content=prompt)])
    new_hist = state.get('message_history', [])
    new_hist.append(AIMessage(content=msg.content))
    return {
        "agent_message": msg.content,
        "message_history": new_hist
    }

async def node_strategize(state: InterviewState) -> Dict[str, Any]:
    grades = state.get("graded_answers", [])
    plan = state['topic_plan']
    current_idx = state.get("current_subtopic_index", 0)
    
    output = {}
    
    if not grades:
        first_topic = next((t for t, s in plan.items() if s == "pending"), None)
        if not first_topic:
            output = {"current_decision": TopicDecision(action="end_interview", reasoning="None")}
        else:
            new_plan = plan.copy()
            new_plan[first_topic] = "active"
            t_info = TOPIC_DATA.get(first_topic, {})
            sub_topics = t_info.get("sub_topics", [])
            
            output = {
                "current_decision": TopicDecision(
                    action="ask_question", topic=first_topic, sub_topic=sub_topics[0], 
                    difficulty="standard", reasoning="Start",
                    is_first_turn=True, is_new_main_topic=True, is_last_in_topic=(len(sub_topics)==1)
                ),
                "topic_plan": new_plan,
                "current_subtopic_index": 0,
                "retrieval_attempts": 0,
                "current_search_query": None
            }
    else:
        last_score = grades[-1]['score']
        low_streak = state.get('consecutive_low_scores', 0)
        low_streak = (low_streak + 1) if last_score < 5 else 0
        
        active_topic = next((t for t, s in plan.items() if s == "active"), None)
        sub_topics = TOPIC_DATA.get(active_topic, {}).get("sub_topics", [])
        
        new_plan = plan.copy()
        decision = None
        new_idx = current_idx
        
        if low_streak >= 2: 
            new_plan[active_topic] = "failed"
            next_t = next((t for t, s in new_plan.items() if s == "pending"), None)
            if next_t:
                new_plan[next_t] = "active"
                decision = TopicDecision(
                    action="ask_question", topic=next_t, sub_topic=TOPIC_DATA[next_t]["sub_topics"][0], 
                    difficulty="fundamental", reasoning="Mercy Pivot", is_new_main_topic=True
                )
                new_idx = 0
                low_streak = 0
            else:
                decision = TopicDecision(action="end_interview", reasoning="Failed")
        
        elif new_idx + 1 < len(sub_topics): 
            new_idx += 1
            decision = TopicDecision(
                action="ask_question", topic=active_topic, sub_topic=sub_topics[new_idx], 
                difficulty="standard", reasoning="Next Subtopic",
                is_new_main_topic=False, is_last_in_topic=(new_idx + 1 == len(sub_topics))
            )
            
        else: 
            new_plan[active_topic] = "complete"
            next_t = next((t for t, s in new_plan.items() if s == "pending"), None)
            if next_t:
                new_plan[next_t] = "active"
                sub_next = TOPIC_DATA.get(next_t, {}).get("sub_topics", [])
                decision = TopicDecision(
                    action="ask_question", topic=next_t, sub_topic=sub_next[0], 
                    difficulty="standard", reasoning="Next Topic",
                    is_new_main_topic=True, is_last_in_topic=(len(sub_next)==1)
                )
                new_idx = 0
            else:
                decision = TopicDecision(action="end_interview", reasoning="Done")
                
        output = {
            "topic_plan": new_plan,
            "consecutive_low_scores": low_streak,
            "current_subtopic_index": new_idx,
            "current_decision": decision,
            "retrieval_attempts": 0,
            "current_search_query": None
        }

    logger.log_node_exec("strategize", state, prompt_data="Logic Block", output=output)
    return output

async def node_retrieve(state: InterviewState) -> Dict[str, Any]:
    decision = state['current_decision']
    query = state.get("current_search_query") or f"{decision.topic} {decision.sub_topic} definition steps process"
    
    context = await retrieve_context_tool(query)
    output = {
        "retrieved_context": context,
        "retrieval_attempts": state.get("retrieval_attempts", 0) + 1
    }
    logger.log_node_exec("retrieve", state, prompt_data=f"Query: {query}", output=output)
    return output

async def node_validate(state: InterviewState) -> Dict[str, Any]:
    decision = state['current_decision']
    context = state['retrieved_context']
    
    user_msg_content = prompts.VALIDATION_USER_TEMPLATE.format(
        topic=decision.topic,
        sub_topic=decision.sub_topic,
        context_snippet=context[:1500]
    )
    
    msgs = [prompts.VALIDATION_SYSTEM, HumanMessage(content=user_msg_content)]
    res = await llm_check.ainvoke(msgs)
    
    output = {
        "is_context_valid": (res.status == "sufficient"),
        "refined_context": res.refined_context,
        "current_search_query": res.search_query
    }
    logger.log_node_exec("validate", state, prompt_data=msgs, output=output)
    return output

async def node_generate(state: InterviewState) -> Dict[str, Any]:
    decision = state['current_decision']
    refined_context = state.get('refined_context')
    prev_feedback = state.get('last_turn_feedback', "Let's begin.")
    is_valid = state.get('is_context_valid', True)
    
    topic_info = TOPIC_DATA.get(decision.topic, {})
    sample_questions = topic_info.get("sample_questions", [])
    sample_qs_text = "\n".join([f"- {q}" for q in sample_questions])
    
    flags = {
        "is_first_turn": decision.is_first_turn,
        "is_new_main_topic": decision.is_new_main_topic,
        "is_last_in_topic": decision.is_last_in_topic,
        "is_context_valid": is_valid
    }
    
    prompt_str = prompts.get_qgen_system_prompt(decision, refined_context, prev_feedback, flags, sample_qs_text)
    msgs = [SystemMessage(content=prompt_str)]
    gen = await llm_gen.ainvoke(msgs)
    
    if gen.question_source == "sample_question":
        sys_logger.info(f"📌 Using Sample Question for {decision.sub_topic}: {gen.technical_question}")
    else:
        sys_logger.info(f"⚡ Generating New Question for {decision.sub_topic}")

    full_message = f"{gen.conversational_entry} {gen.technical_question}"
    new_hist = state.get('message_history', [])
    new_hist.append(AIMessage(content=full_message))
    
    output = {
        "last_question": gen.technical_question,
        "last_expected_answer": gen.expected_answer,
        "agent_message": full_message,
        "message_history": new_hist,
        "refined_context": None,
        "current_search_query": None
    }
    logger.log_node_exec("generate", state, prompt_data=msgs, output=output)
    return output

def calculate_interview_metrics(turns: List[Dict]) -> Dict[str, Any]:
    """Calculates quantitative metrics from interview turns."""
    if not turns:
        return None

    by_topic = {}
    total_score = 0
    total_max_score = 0
    
    for turn in turns:
        topic = turn.get("topic", "General")
        score = turn.get("score", 0)
        
        if topic not in by_topic:
            by_topic[topic] = {"score_sum": 0, "count": 0}
        
        by_topic[topic]["score_sum"] += score
        by_topic[topic]["count"] += 1
        
        total_score += score
        total_max_score += 10 

    section_breakdown = []
    for topic, data in by_topic.items():
        avg = data["score_sum"] / data["count"] if data["count"] > 0 else 0
        section_breakdown.append({
            "topic": topic,
            "questions_asked": data["count"],
            "total_score": data["score_sum"],
            "average_score": round(avg, 1),
            "max_possible": data["count"] * 10
        })

    overall_percentage = (total_score / total_max_score * 100) if total_max_score > 0 else 0

    return {
        "total_questions": len(turns),
        "total_score": total_score,
        "max_possible_score": total_max_score,
        "overall_percentage": round(overall_percentage, 1),
        "section_breakdown": section_breakdown
    }

async def node_summarize(state: InterviewState) -> Dict[str, Any]:
    all_turns = simple_db.get_turn_data(state['session_id'])
    
    metrics = calculate_interview_metrics(all_turns)
    
    log_str = str([{ "topic": t['topic'], "score": t['score'], "feedback": t['evaluation_feedback']} for t in all_turns])
    stats_summary = f"\nSTATS: Total Score: {metrics['total_score']}/{metrics['max_possible_score']} ({metrics['overall_percentage']}%)"
    
    msgs = [
        prompts.SUMMARIZER_SYSTEM, 
        HumanMessage(content=f"{prompts.SUMMARIZER_USER_PREFIX}{log_str} {stats_summary}")
    ]
    
    summary = await llm_sum.ainvoke(msgs)
    simple_db.save_summary(state['session_id'], summary.overall_rating, summary.summary_text)
    
    final_output = summary.model_dump()
    final_output["metrics"] = metrics 
    
    output = {"final_summary": final_output, "is_interview_over": True}
    logger.log_node_exec("summarize", state, prompt_data=msgs, output=output)
    return output

async def node_end(state):
    return {"agent_message": "Interview Complete."}

# ==============================================================================
# 6. GRAPH CONSTRUCTION
# ==============================================================================

workflow = StateGraph(InterviewState)
workflow.add_node("evaluate", node_evaluate)
workflow.add_node("strategize", node_strategize)
workflow.add_node("retrieve", node_retrieve)
workflow.add_node("validate", node_validate)
workflow.add_node("generate", node_generate)
workflow.add_node("summarize", node_summarize)
workflow.add_node("end", node_end)
workflow.add_node("ask_followup", node_ask_followup)

def route_start(state):
    if not state.get("session_id"): return "strategize"
    if state.get("candidate_answer"): return "evaluate"
    return "strategize"

def route_strategy(state):
    if state['current_decision'].action == "end_interview": return "summarize"
    return "retrieve"

def route_validate(state):
    if state["is_context_valid"]:
        return "generate"
    elif state.get("retrieval_attempts", 0) < 2:
        return "retrieve"
    else:
        return "generate" 
    
def route_evaluate(state):
    if state.get("is_in_probe_mode"):
        return "ask_followup"
    return "strategize"    

workflow.add_conditional_edges(START, route_start, {"evaluate": "evaluate", "strategize": "strategize"})
workflow.add_conditional_edges("strategize", route_strategy, {"summarize": "summarize", "retrieve": "retrieve"})
workflow.add_edge("retrieve", "validate")
workflow.add_conditional_edges("validate", route_validate, {"generate": "generate", "retrieve": "retrieve"})
workflow.add_edge("generate", END)
workflow.add_edge("summarize", "end")
workflow.add_edge("end", END)
workflow.add_conditional_edges("evaluate", route_evaluate, {
    "ask_followup": "ask_followup",
    "strategize": "strategize"
})
workflow.add_edge("ask_followup", END)

memory = MemorySaver()

# ==============================================================================
# 7. API AUTH HELPERS
# ==============================================================================

def verify_password(plain_password, hashed_password):
    return bcrypt.checkpw(plain_password.encode('utf-8'), hashed_password.encode('utf-8'))

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        role: str = payload.get("role")
        if username is None:
            raise credentials_exception
        return {"username": username, "role": role}
    except JWTError:
        raise credentials_exception

# ==============================================================================
# 8. API ENDPOINTS
# ==============================================================================

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

class NextRequest(BaseModel):
    candidate_id: str
    answer: str

@app.get("/")
async def get_ui(): return FileResponse("index.html")

@app.get("/login.html")
async def get_login(): return FileResponse("login.html")

@app.get("/admin.html")
async def get_admin_ui(): return FileResponse("admin.html")

@app.post("/register")
async def register(user: UserLogin):
    success = simple_db.create_user(user.username, user.password, role="candidate")
    if not success:
        raise HTTPException(status_code=400, detail="Username already registered")
    return {"message": "User created successfully"}

@app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = simple_db.get_user(form_data.username)
    if not user or not verify_password(form_data.password, user['password_hash']):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token = create_access_token(
        data={"sub": user['username'], "role": user['role']}
    )
    return {"access_token": access_token, "token_type": "bearer", "role": user['role']}

@app.post("/interview/start")
async def start_interview(current_user: dict = Depends(get_current_user)):
    if current_user['role'] == 'manager':
         return {"agent_message": "ACCESS DENIED: Managers cannot take interviews. Please log in as a candidate."}

    session_id = str(uuid4())
    simple_db.log_session_start(session_id, current_user['username'])

    init_state = {
        "session_id": session_id,
        "topic_plan": {k: "pending" for k in TOPIC_DATA.keys()},
        "graded_answers": [],
        "message_history": [],
        "consecutive_low_scores": 0,
        "current_subtopic_index": 0,
        "is_interview_over": False,
        "candidate_answer": None,
        "retrieval_attempts": 0
    }
    config = {"configurable": {"thread_id": session_id}}
    app_graph = workflow.compile(checkpointer=memory)
    output = await app_graph.ainvoke(init_state, config=config)
    return {"candidate_id": session_id, "agent_message": output.get("agent_message")}

@app.post("/interview/next")
async def next_step(req: NextRequest, current_user: dict = Depends(get_current_user)):
    config = {"configurable": {"thread_id": req.candidate_id}}
    app_graph = workflow.compile(checkpointer=memory)
    output = await app_graph.ainvoke({"candidate_answer": req.answer}, config=config)
    return {
        "candidate_id": req.candidate_id,
        "agent_message": output.get("agent_message"),
        "is_interview_over": output.get("is_interview_over", False)
    }

@app.get("/interview/state/{candidate_id}")
async def get_interview_state(candidate_id: str, current_user: dict = Depends(get_current_user)):
    turns = simple_db.get_turn_data(candidate_id)
    config = {"configurable": {"thread_id": candidate_id}}
    app_graph = workflow.compile(checkpointer=memory)
    
    final_summary_data = None
    try:
        snap = await app_graph.aget_state(config)
        if snap.values.get("final_summary"):
            final_summary_data = snap.values.get("final_summary")
            if "metrics" not in final_summary_data:
                final_summary_data["metrics"] = calculate_interview_metrics(turns)
    except:
        pass
        
    return {"candidate_id": candidate_id, "interview_turns": turns, "final_summary": final_summary_data}

# --- Admin Endpoints ---

@app.get("/admin/sessions")
async def get_all_sessions(current_user: dict = Depends(get_current_user)):
    if current_user['role'] != 'manager':
        raise HTTPException(status_code=403, detail="Admin privileges required")
    return simple_db.get_all_sessions_for_admin()

@app.get("/admin/session/{session_id}")
async def get_session_details(session_id: str, current_user: dict = Depends(get_current_user)):
    if current_user['role'] != 'manager':
        raise HTTPException(status_code=403, detail="Admin privileges required")
    
    turns = simple_db.get_turn_data(session_id)
    summary_row = simple_db.get_summary(session_id)
    
    # Calculate metrics on the fly (Handles abruptly stopped sessions)
    metrics = calculate_interview_metrics(turns)
    
    return {
        "session_id": session_id, 
        "turns": turns, 
        "metrics": metrics,
        "summary": summary_row
    }

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)