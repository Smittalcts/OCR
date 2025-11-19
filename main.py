import os
import uvicorn
import logging
import sys
import json
import asyncio
import operator
from typing import List, Dict, Any, Optional, Annotated, Literal
from uuid import uuid4
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# --- LangChain & LangGraph Imports ---
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver  # Changed from AsyncSqliteSaver
from typing_extensions import TypedDict
from langchain_core.globals import set_debug

# --- Custom Imports ---
import simple_db  # New DB module
import prompts    # New Prompts module

# --- Environment Setup ---
load_dotenv()

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)
set_debug(False)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("chromadb").setLevel(logging.WARNING)

# --- 1. Check Environment ---
def check_env_vars():
    required_vars = [
        "AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "OPENAI_API_VERSION",
        "AZURE_OPENAI_CHAT_DEPLOYMENT_NAME", "AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"
    ]
    if any(not os.environ.get(var) for var in required_vars):
        logger.error("FATAL ERROR: Missing required Azure environment variables.")
        exit(1)
check_env_vars()

# --- 2. Load Topic Structure ---
TOPIC_DATA = {}
try:
    with open("topics.json", 'r') as f:
        TOPIC_DATA = json.load(f)
    logger.info(f"Loaded {len(TOPIC_DATA)} topics.")
except Exception as e:
    logger.error(f"FATAL ERROR: Could not load topics.json: {e}", exc_info=True)
    exit(1)

# --- 3. Initialize DBs ---
simple_db.init_db()

# --- 4. LLM & Vector Store ---
llm_client = AzureChatOpenAI(
    model='gpt-4o-2024-11-20',
    azure_deployment=os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
    api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
    api_version=os.environ.get("OPENAI_API_VERSION"),
)

PERSIST_DIRECTORY = "./chroma_db"
retriever = None
try:
    embeddings_model = AzureOpenAIEmbeddings(
        openai_api_version=os.environ.get("OPENAI_API_VERSION"),
        azure_deployment=os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
    )
    vector_store = Chroma(
        persist_directory=PERSIST_DIRECTORY,
        embedding_function=embeddings_model
    )
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    logger.info(f"Loaded Chroma from '{PERSIST_DIRECTORY}'.")
except Exception as e:
    logger.error(f"Error initializing Chroma: {e}")

async def retrieve_context_tool(query: str, k: int = 3) -> str:
    if not retriever: return "Error: Retriever not initialized."
    try:
        retriever.search_kwargs["k"] = k
        docs = await retriever.ainvoke(query)
        return "\n\n".join([d.page_content for d in docs]) or "No data found."
    except Exception as e:
        return f"Search Error: {e}"

# --- 5. Models ---
class TopicDecision(BaseModel):
    action: Literal["ask_question", "end_interview"]
    topic: Optional[str] = Field(None)
    sub_topic: Optional[str] = Field(None)
    sample_question: Optional[str] = Field(None)
    difficulty: Literal["standard", "deep_dive", "fundamental"] = Field("standard")
    reasoning: str = Field(...)

class EvaluationResult(BaseModel):
    score: int = Field(...)
    evaluation: str = Field(...)
    missed_points: List[str] = Field(...)

class GeneratedQuestion(BaseModel):
    question: str = Field(..., description="The conversational question to ask.")
    expected_answer: str = Field(..., description="A 3-4 line technical answer expected from the candidate.")

class FinalSummary(BaseModel):
    overall_rating: Literal["Strong Hire", "Hire", "No Hire"]
    summary_text: str = Field(...)
    key_takeaways: List[str] = Field(...)

class RelevanceCheck(BaseModel):
    is_relevant: bool = Field(..., description="Is the retrieved context relevant?")
    reason: str = Field(...)

# --- 6. LangGraph State ---
class InterviewState(TypedDict):
    session_id: str
    topic_plan: Dict[str, str]
    current_subtopic_index: int
    graded_answers: Annotated[List[Dict[str, Any]], operator.add]
    message_history: Annotated[List[BaseMessage], operator.add]
    consecutive_low_scores: int 
    current_decision: Optional[TopicDecision]
    last_question: Optional[str]
    last_expected_answer: Optional[str]
    candidate_answer: Optional[str]
    retrieved_context: Optional[str]
    retrieval_attempts: int 
    agent_message: Optional[str]
    final_summary: Optional[Dict[str, Any]]
    is_interview_over: bool

# --- 7. Models & Chains ---
llm_eval = llm_client.with_structured_output(EvaluationResult)
llm_gen = llm_client.with_structured_output(GeneratedQuestion)
llm_sum = llm_client.with_structured_output(FinalSummary)
llm_check_relevance = llm_client.with_structured_output(RelevanceCheck)

def debug_print(node_name: str, data: Dict[str, Any]):
    print(f"\n\033[96m--- [{node_name.upper()}] OUTPUT ---\033[0m")
    try:
        print(json.dumps(data, indent=2, default=str))
    except Exception:
        print(data)
    print("\033[96m--------------------------------\033[0m\n")

# ==============================================================================
# SUBGRAPH 1: TOPIC MANAGER
# ==============================================================================

async def node_strategize_topic(state: InterviewState) -> Dict[str, Any]:
    grades = state.get("graded_answers", [])
    plan = state['topic_plan']
    current_idx = state.get("current_subtopic_index", 0)
    
    # 1. Start
    if not grades:
        first_topic = next((t for t, s in plan.items() if s == "pending"), None)
        if not first_topic:
            return {"current_decision": TopicDecision(action="end_interview", reasoning="No topics.")}
        
        new_plan = plan.copy()
        new_plan[first_topic] = "active"
        topic_info = TOPIC_DATA.get(first_topic, {})
        sub_topics = topic_info.get("sub_topics", ["General"])
        sample_qs = topic_info.get("sample_questions", [])
        
        res = {
            "topic_plan": new_plan,
            "consecutive_low_scores": 0,
            "current_subtopic_index": 0,
            "current_decision": TopicDecision(
                action="ask_question", topic=first_topic, sub_topic=sub_topics[0],
                sample_question=sample_qs[0] if sample_qs else None,
                difficulty="standard", reasoning="Starting interview."
            )
        }
        debug_print("strategize", res)
        return res

    # 2. Analyze
    last_grade = grades[-1]
    current_score = last_grade['score']
    low_streak = state.get('consecutive_low_scores', 0)
    low_streak = (low_streak + 1) if current_score < 5 else 0
    active_topic = next((t for t, s in plan.items() if s == "active"), None)
    
    # 3. Decide
    new_plan = plan.copy()
    decision = None
    new_idx = current_idx

    # Mercy Pivot
    if low_streak >= 2:
        if active_topic: new_plan[active_topic] = "failed"
        next_topic = next((t for t, s in new_plan.items() if s == "pending"), None)
        if next_topic:
            new_plan[next_topic] = "active"
            new_idx = 0
            t_info = TOPIC_DATA.get(next_topic, {})
            decision = TopicDecision(
                action="ask_question", topic=next_topic, sub_topic=t_info["sub_topics"][0],
                difficulty="fundamental", reasoning="Pivoting due to struggle."
            )
            low_streak = 0 
        else:
            decision = TopicDecision(action="end_interview", reasoning="Failed and no topics left.")
    else:
        # Check current topic progress
        last_decision = state.get('current_decision')
        if last_decision and last_decision.topic == active_topic:
             new_idx += 1 
        
        topic_info = TOPIC_DATA.get(active_topic, {})
        sub_topics = topic_info.get("sub_topics", [])
        sample_qs = topic_info.get("sample_questions", [])
        
        if new_idx < len(sub_topics):
            decision = TopicDecision(
                action="ask_question", topic=active_topic, sub_topic=sub_topics[new_idx],
                sample_question=sample_qs[new_idx] if new_idx < len(sample_qs) else None,
                difficulty="standard", reasoning=f"Next subtopic {new_idx+1}"
            )
        else:
            # Topic Done
            new_plan[active_topic] = "complete"
            next_topic = next((t for t, s in new_plan.items() if s == "pending"), None)
            if next_topic:
                new_plan[next_topic] = "active"
                new_idx = 0
                nt_info = TOPIC_DATA.get(next_topic, {})
                decision = TopicDecision(
                    action="ask_question", topic=next_topic, sub_topic=nt_info["sub_topics"][0],
                    difficulty="standard", reasoning="Next Topic."
                )
            else:
                decision = TopicDecision(action="end_interview", reasoning="Complete.")

    res = {
        "topic_plan": new_plan,
        "consecutive_low_scores": low_streak,
        "current_subtopic_index": new_idx,
        "current_decision": decision
    }
    debug_print("strategize", res)
    return res

topic_builder = StateGraph(InterviewState)
topic_builder.add_node("strategize", node_strategize_topic)
topic_builder.add_edge(START, "strategize")
topic_builder.add_edge("strategize", END)
topic_subgraph = topic_builder.compile()

# ==============================================================================
# SUBGRAPH 2: QUESTION GENERATOR
# ==============================================================================

async def subnode_retrieve_knowledge(state: InterviewState) -> Dict[str, Any]:
    decision = state['current_decision']
    attempt = state.get('retrieval_attempts', 0)
    query = f"{decision.topic} {decision.sub_topic}"
    if decision.sample_question: query += f" {decision.sample_question}"
    if attempt > 0: query += " definition explanation"
    
    context = await retrieve_context_tool(query, k=3)
    res = {"retrieved_context": context, "retrieval_attempts": attempt + 1}
    debug_print("retrieve_knowledge", res)
    return res

async def subnode_validate_context(state: InterviewState) -> Dict[str, Any]:
    decision = state['current_decision']
    context = state['retrieved_context']
    
    if not context or "No data found" in context:
        return {"is_context_valid": False}

    # Use PROMPT file
    msg = prompts.VALIDATION_USER_TEMPLATE.format(
        topic=decision.topic, sub_topic=decision.sub_topic, context_snippet=context
    )
    messages = [prompts.VALIDATION_SYSTEM, HumanMessage(content=msg)]
    
    check = await llm_check_relevance.ainvoke(messages)
    res = {"is_context_valid": check.is_relevant}
    debug_print("validate_context", res)
    return res

async def subnode_draft_question(state: InterviewState) -> Dict[str, Any]:
    decision = state['current_decision']
    context = state['retrieved_context']
    
    sample_instr = ""
    if decision.sample_question:
        sample_instr = f"Use reference: '{decision.sample_question}'."

    # Use PROMPT file
    system_content = prompts.get_qgen_system_prompt(decision, context, sample_instr)
    
    messages = [SystemMessage(content=system_content)]
    if state['message_history']:
        messages.extend(state['message_history'][-10:])
    
    gen = await llm_gen.ainvoke(messages)
    
    # --- LOGGING ---
    simple_db.log_message(state['session_id'], "assistant", gen.question)
    
    res = {
        "last_question": gen.question,
        "last_expected_answer": gen.expected_answer,
        "agent_message": gen.question,
        "message_history": [AIMessage(content=gen.question)],
        "retrieval_attempts": 0
    }
    debug_print("draft_question", res)
    return res

def route_validation(state: InterviewState):
    if state.get("is_context_valid", True): return "draft_question"
    if state.get("retrieval_attempts", 0) >= 3: return "draft_question"
    return "retrieve_knowledge"

qgen_builder = StateGraph(InterviewState)
qgen_builder.add_node("retrieve_knowledge", subnode_retrieve_knowledge)
qgen_builder.add_node("validate_context", subnode_validate_context)
qgen_builder.add_node("draft_question", subnode_draft_question)
qgen_builder.add_edge(START, "retrieve_knowledge")
qgen_builder.add_edge("retrieve_knowledge", "validate_context")
qgen_builder.add_conditional_edges("validate_context", route_validation, {
    "draft_question": "draft_question", "retrieve_knowledge": "retrieve_knowledge"
})
qgen_builder.add_edge("draft_question", END)
qgen_subgraph = qgen_builder.compile()

# ==============================================================================
# MAIN GRAPH NODES
# ==============================================================================

async def node_evaluate(state: InterviewState) -> Dict[str, Any]:
    question = state['last_question']
    answer = state['candidate_answer']
    expected = state.get('last_expected_answer', "N/A")
    
    # Use PROMPT file
    msg = prompts.EVALUATOR_USER_TEMPLATE.format(
        question=question, answer=answer, expected=expected
    )
    messages = [prompts.EVALUATOR_SYSTEM, HumanMessage(content=msg)]
    
    eval_result = await llm_eval.ainvoke(messages)
    
    # --- LOGGING ---
    simple_db.log_message(state['session_id'], "user", answer)
    simple_db.log_message(
        state['session_id'], 
        "system_eval", 
        f"Score: {eval_result.score}/10. {eval_result.evaluation}", 
        metadata=eval_result.model_dump()
    )
    
    record = {
        "question": question, "answer": answer,
        "score": eval_result.score, "details": eval_result.model_dump()
    }
    
    res = {
        "graded_answers": [record],
        "message_history": [HumanMessage(content=answer)],
        "candidate_answer": None
    }
    debug_print("evaluate", res)
    return res

async def node_summarize(state: InterviewState) -> Dict[str, Any]:
    grades_json = json.dumps(state['graded_answers'], default=str)
    
    # Use PROMPT file
    msgs = [
        prompts.SUMMARIZER_SYSTEM,
        HumanMessage(content=f"{prompts.SUMMARIZER_USER_PREFIX} {grades_json}"),
        HumanMessage(content="Transcript:")
    ] + state['message_history']
    
    summary = await llm_sum.ainvoke(msgs)
    
    # --- LOGGING ---
    simple_db.save_summary(state['session_id'], summary.overall_rating, summary.summary_text)
    
    res = {"final_summary": summary.model_dump(), "is_interview_over": True}
    debug_print("summarize", res)
    return res

async def node_end(state: InterviewState) -> Dict[str, Any]:
    return {"agent_message": "Thank you, the interview is complete."}

# ==============================================================================
# MAIN GRAPH & APP
# ==============================================================================

workflow = StateGraph(InterviewState)
workflow.add_node("evaluate", node_evaluate)
workflow.add_node("topic_manager", topic_subgraph)
workflow.add_node("question_generator", qgen_subgraph)
workflow.add_node("summarize", node_summarize)
workflow.add_node("end", node_end)

def route_start(state: InterviewState):
    if state.get("candidate_answer"): return "evaluate"
    return "topic_manager"

def route_decision(state: InterviewState):
    if state['current_decision'].action == "ask_question": return "question_generator"
    return "summarize"

workflow.add_conditional_edges(START, route_start, {"evaluate": "evaluate", "topic_manager": "topic_manager"})
workflow.add_edge("evaluate", "topic_manager")
workflow.add_conditional_edges("topic_manager", route_decision, {"question_generator": "question_generator", "summarize": "summarize"})
workflow.add_edge("question_generator", END)
workflow.add_edge("summarize", "end")
workflow.add_edge("end", END)

# --- IN-MEMORY CHECKPOINTER (Fixes Blob Error) ---
memory_checkpointer = MemorySaver()

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.get("/")
async def get_ui():
    return FileResponse("index.html")

class NextRequest(BaseModel):
    candidate_id: str
    answer: str

@app.post("/interview/start")
async def start_interview():
    session_id = str(uuid4())
    initial_state = {
        "session_id": session_id,
        "topic_plan": {k: "pending" for k in TOPIC_DATA.keys()},
        "graded_answers": [],
        "message_history": [],
        "consecutive_low_scores": 0,
        "current_subtopic_index": 0,
        "retrieval_attempts": 0,
        "is_interview_over": False,
        "last_expected_answer": None,
        "candidate_answer": None
    }
    config = {"configurable": {"thread_id": session_id}}
    
    app_graph = workflow.compile(checkpointer=memory_checkpointer)
    output = await app_graph.ainvoke(initial_state, config=config)
    
    # Log initial greeting
    simple_db.log_message(session_id, "system", "Interview Started")
    simple_db.log_message(session_id, "assistant", output.get("agent_message", ""))
    
    return {"candidate_id": session_id, "agent_message": output.get("agent_message")}

@app.post("/interview/next")
async def next_step(req: NextRequest):
    config = {"configurable": {"thread_id": req.candidate_id}}
    app_graph = workflow.compile(checkpointer=memory_checkpointer)
    
    output = await app_graph.ainvoke({"candidate_answer": req.answer}, config=config)
    
    return {
        "candidate_id": req.candidate_id,
        "agent_message": output.get("agent_message"),
        "is_interview_over": output.get("is_interview_over", False)
    }

@app.get("/interview/state/{candidate_id}")
async def get_interview_state(candidate_id: str):
    config = {"configurable": {"thread_id": candidate_id}}
    app_graph = workflow.compile(checkpointer=memory_checkpointer)
    
    # Try to get active state from RAM
    try:
        snapshot = await app_graph.aget_state(config)
        state = snapshot.values
    except Exception:
        state = {}

    # Fetch persistent logs from Simple DB
    transcript = simple_db.get_transcript(candidate_id)
    
    return {
        "candidate_id": candidate_id,
        "is_over": state.get("is_interview_over", False),
        "transcript": transcript, # Readable history for UI/Manager
        "final_summary": state.get("final_summary")
    }

if __name__ == "__main__":
    print("Server running at http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)