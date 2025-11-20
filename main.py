import os
import uvicorn
import logging
import sys
import json
import operator
from datetime import datetime
from typing import List, Dict, Any, Optional, Annotated, Literal
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

# --- LangChain & LangGraph Imports ---
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from typing_extensions import TypedDict
from langchain_core.globals import set_debug

# --- Database Import (Assumes simple_db.py exists in same folder) ---
import simple_db

# --- Environment Setup ---
load_dotenv()

# Configure standard logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    stream=sys.stdout,
)
sys_logger = logging.getLogger("InterviewLogger")
set_debug(False)

def check_env_vars():
    required_vars = ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", "OPENAI_API_VERSION"]
    if any(not os.environ.get(var) for var in required_vars):
        sys_logger.error("FATAL ERROR: Missing Azure environment variables.")
        exit(1)
check_env_vars()

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
            "current_decision", "candidate_answer", "user_intent", 
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
            # Remove heavy objects from output log
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

class UserIntent(BaseModel):
    intent: Literal["answer", "clarification", "off_topic"]

# --- UPDATED: Split Fields for Natural Flow ---
class GeneratedQuestion(BaseModel):
    conversational_entry: str = Field(..., description="The social bridge (e.g., 'That's correct. Now...')")
    technical_question: str = Field(..., description="The actual interview question.")
    expected_answer: str

class ClarificationResponse(BaseModel):
    explanation_part: str = Field(..., description="Direct answer to the user's confusion.")
    follow_up_question: str = Field(..., description="The original or simplified interview question to get back on track.")

class EvaluationResult(BaseModel):
    score: int
    evaluation: str
    missed_points: List[str]
    feedback_for_next_node: str 

class FinalSummary(BaseModel):
    overall_rating: Literal["Strong Hire", "Hire", "No Hire"]
    summary_text: str
    key_takeaways: List[str]

class RelevanceCheck(BaseModel):
    status: Literal["sufficient", "partial", "irrelevant"]
    reason: str
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
    message_history: Annotated[List[BaseMessage], operator.add] 
    
    current_decision: Optional[TopicDecision]
    
    # Retrieval Loop
    retrieved_context: Optional[str]
    refined_context: Optional[str]
    current_search_query: Optional[str]
    retrieval_attempts: int
    is_context_valid: bool
    
    candidate_answer: Optional[str]
    user_intent: Optional[str]
    
    last_question: Optional[str]
    last_expected_answer: Optional[str]
    last_turn_feedback: Optional[str] 
    
    graded_answers: Annotated[List[PerformanceTracker], operator.add]
    consecutive_low_scores: int
    
    agent_message: Optional[str]
    final_summary: Optional[Dict[str, Any]]

# Load Topics
TOPIC_DATA = {}
try:
    with open("topics.json", 'r') as f:
        TOPIC_DATA = json.load(f)
except Exception as e:
    sys_logger.error(f"Could not load topics.json: {e}")
    exit(1)

simple_db.init_db()

# LLM Setup
llm_client = AzureChatOpenAI(
    model='gpt-4o-2024-11-20',
    azure_deployment=os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"),
    api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"),
    api_version=os.environ.get("OPENAI_API_VERSION"),
)

# Chroma Setup
PERSIST_DIRECTORY = "./chroma_db"
retriever = None
try:
    embeddings_model = AzureOpenAIEmbeddings(
        openai_api_version=os.environ.get("OPENAI_API_VERSION"),
        azure_deployment=os.environ.get("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME"),
    )
    vector_store = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings_model)
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
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

# Structured Chains
llm_intent = llm_client.with_structured_output(UserIntent)
llm_eval = llm_client.with_structured_output(EvaluationResult)
llm_gen = llm_client.with_structured_output(GeneratedQuestion)
llm_clarify = llm_client.with_structured_output(ClarificationResponse)
llm_sum = llm_client.with_structured_output(FinalSummary)
llm_check = llm_client.with_structured_output(RelevanceCheck)

# ==============================================================================
# 4. PROMPTS (INLINED FOR STABILITY)
# ==============================================================================

INTENT_SYSTEM = SystemMessage(content="""
ROLE: Dialogue Intent Classifier.
CATEGORIES:
1. "answer": Candidate is attempting to answer the question.
2. "clarification": Candidate is asking for definition, help, or context.
3. "off_topic": Completely irrelevant.
""")

VALIDATION_SYSTEM = SystemMessage(content="""
ROLE: Content Auditor.
TASK: Check if retrieved text is sufficient for a Senior Interview.
OUTPUT: JSON with status ("sufficient"/"partial"/"irrelevant") and "refined_context" (Clean Cheat Sheet).
If partial/irrelevant, provide a better "search_query".
""")

def get_qgen_prompt(decision, refined_context, prev_feedback, flags):
    # Dynamic Transition Construction
    if flags.get("is_first_turn"):
        trans_instr = f"OPENING: Greet professionally. State topic: {decision.topic}."
    elif flags.get("is_new_main_topic"):
        trans_instr = f"TRANSITION: 'That covers the previous section. Moving on to {decision.topic}...'"
    elif flags.get("is_last_in_topic"):
        trans_instr = f"TRANSITION: Acknowledge answer using: '{prev_feedback}'. Mention this is the last question on this topic."
    else:
        trans_instr = f"TRANSITION: Acknowledge answer using: '{prev_feedback}'."

    return f"""
    ROLE: Senior On-Call SRE Interviewer.
    GOAL: Natural, human-like flow.
    
    CONTEXT:
    - Topic: {decision.topic} -> {decision.sub_topic}
    - Cheat Sheet: {refined_context}
    
    INSTRUCTIONS:
    1. {trans_instr}
    2. ASK: Formulate a specific question based on the Cheat Sheet.
    
    IMPORTANT: Split your response into 'conversational_entry' (The bridge/greeting) and 'technical_question' (The actual question).
    """

CLARIFICATION_SYSTEM = """
ROLE: Helpful Interviewer.
USER SITUATION: Candidate asked for clarification on the previous question.
INSTRUCTIONS:
1. 'explanation_part': Explain the concept simply using the Context.
2. 'follow_up_question': Re-phrase the original question to be clearer.
"""

EVALUATOR_SYSTEM = SystemMessage(content="""
ROLE: Technical Grader.
INSTRUCTIONS:
1. Score 0-10 based on RAG Context.
2. Generate 'feedback_for_next_node': A conversational sentence summarizing how they did (e.g., "Great point on X, but you missed Y.").
""")

# ==============================================================================
# 5. NODES
# ==============================================================================

# --- NODE: INTENT ROUTER ---
async def node_intent_router(state: InterviewState) -> Dict[str, Any]:
    if not state.get("candidate_answer"):
        return {"user_intent": "answer"}
    
    msgs = [INTENT_SYSTEM, HumanMessage(content=state["candidate_answer"])]
    res = await llm_intent.ainvoke(msgs)
    
    logger.log_node_exec("intent_router", state, prompt_data=msgs, output={"intent": res.intent})
    return {"user_intent": res.intent}

# --- NODE: EVALUATE ---
async def node_evaluate(state: InterviewState) -> Dict[str, Any]:
    question = state.get('last_question', "Intro")
    answer = state['candidate_answer']
    expected = state.get('last_expected_answer', "N/A")
    context = state.get('refined_context') or state.get('retrieved_context', "N/A")
    
    user_msg = f"QUESTION: {question}\nANSWER: {answer}\nEXPECTED: {expected}\nCONTEXT: {context}"
    msgs = [EVALUATOR_SYSTEM, HumanMessage(content=user_msg)]
    
    eval_result = await llm_eval.ainvoke(msgs)
    
    # Log to DB
    simple_db.log_turn_data(
        session_id=state['session_id'],
        topic=state['current_decision'].topic,
        sub_topic=state['current_decision'].sub_topic,
        question=question,
        expected_answer=expected,
        user_answer=answer,
        score=eval_result.score,
        evaluation_feedback=eval_result.evaluation,
        metadata=eval_result.model_dump()
    )
    
    # Prune History (Keep last 6 items only)
    curr_hist = state.get('message_history', [])
    curr_hist.append(HumanMessage(content=answer))
    if len(curr_hist) > 6: curr_hist = curr_hist[-6:]
    
    output = {
        "graded_answers": [{"score": eval_result.score, "topic": state['current_decision'].topic}],
        "last_turn_feedback": eval_result.feedback_for_next_node,
        "message_history": curr_hist,
        "candidate_answer": None
    }
    logger.log_node_exec("evaluate", state, prompt_data=msgs, output=output)
    return output

# --- NODE: CLARIFICATION ---
async def node_handle_clarification(state: InterviewState) -> Dict[str, Any]:
    last_q = state.get("last_question", "")
    context = state.get("refined_context") or state.get("retrieved_context", "")
    
    user_msg = f"USER ASKED: {state['candidate_answer']}\nORIGINAL QUESTION: {last_q}\nCONTEXT: {context}"
    msgs = [SystemMessage(content=CLARIFICATION_SYSTEM), HumanMessage(content=user_msg)]
    
    res = await llm_clarify.ainvoke(msgs)
    
    # Natural concatenation: Explanation + Follow-up
    full_response = f"{res.explanation_part} {res.follow_up_question}"
    
    curr_hist = state.get('message_history', [])
    curr_hist.append(HumanMessage(content=state['candidate_answer']))
    curr_hist.append(AIMessage(content=full_response))
    
    output = {
        "agent_message": full_response,
        "message_history": curr_hist[-6:], # Keep history light
        "candidate_answer": None
    }
    logger.log_node_exec("handle_clarification", state, prompt_data=msgs, output=output)
    return output

# --- NODE: STRATEGIST ---
async def node_strategize(state: InterviewState) -> Dict[str, Any]:
    grades = state.get("graded_answers", [])
    plan = state['topic_plan']
    current_idx = state.get("current_subtopic_index", 0)
    
    output = {}
    
    if not grades:
        # First Turn Logic
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
        # Progression Logic
        last_score = grades[-1]['score']
        low_streak = state.get('consecutive_low_scores', 0)
        low_streak = (low_streak + 1) if last_score < 5 else 0
        
        active_topic = next((t for t, s in plan.items() if s == "active"), None)
        sub_topics = TOPIC_DATA.get(active_topic, {}).get("sub_topics", [])
        
        new_plan = plan.copy()
        decision = None
        new_idx = current_idx
        
        if low_streak >= 2: # Mercy Pivot
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
        
        elif new_idx + 1 < len(sub_topics): # Next Subtopic
            new_idx += 1
            decision = TopicDecision(
                action="ask_question", topic=active_topic, sub_topic=sub_topics[new_idx], 
                difficulty="standard", reasoning="Next Subtopic",
                is_new_main_topic=False, is_last_in_topic=(new_idx + 1 == len(sub_topics))
            )
            
        else: # Next Main Topic
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

# --- NODE: RETRIEVE ---
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

# --- NODE: VALIDATE ---
async def node_validate(state: InterviewState) -> Dict[str, Any]:
    decision = state['current_decision']
    context = state['retrieved_context']
    
    user_msg = f"TOPIC: {decision.topic}\nSUBTOPIC: {decision.sub_topic}\nRAW TEXT: {context[:1500]}"
    msgs = [VALIDATION_SYSTEM, HumanMessage(content=user_msg)]
    
    res = await llm_check.ainvoke(msgs)
    
    output = {
        "is_context_valid": (res.status == "sufficient"),
        "refined_context": res.refined_context,
        "current_search_query": res.search_query
    }
    logger.log_node_exec("validate", state, prompt_data=msgs, output=output)
    return output

# --- NODE: GENERATE ---
async def node_generate(state: InterviewState) -> Dict[str, Any]:
    decision = state['current_decision']
    refined_context = state.get('refined_context')
    prev_feedback = state.get('last_turn_feedback', "Let's begin.")
    is_valid = state.get('is_context_valid', True)
    
    flags = {
        "is_first_turn": decision.is_first_turn,
        "is_new_main_topic": decision.is_new_main_topic,
        "is_last_in_topic": decision.is_last_in_topic,
        "is_context_valid": is_valid
    }
    
    prompt_str = get_qgen_prompt(decision, refined_context, prev_feedback, flags)
    
    # OPTIMIZATION: Clean context window, only system prompt (with cheat sheet) is sent
    msgs = [SystemMessage(content=prompt_str)]
    
    gen = await llm_gen.ainvoke(msgs)
    
    # Combine conversational entry + question for natural flow
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

# --- NODE: SUMMARIZE ---
async def node_summarize(state: InterviewState) -> Dict[str, Any]:
    all_turns = simple_db.get_turn_data(state['session_id'])
    log_str = str([{ "topic": t['topic'], "score": t['score'], "feedback": t['evaluation_feedback']} for t in all_turns])
    
    summ_sys = SystemMessage(content="ROLE: Hiring Manager. Summarize interview performance.")
    msgs = [summ_sys, HumanMessage(content=f"Interview Logs: {log_str}")]
    
    summary = await llm_sum.ainvoke(msgs)
    simple_db.save_summary(state['session_id'], summary.overall_rating, summary.summary_text)
    
    output = {"final_summary": summary.model_dump(), "is_interview_over": True}
    logger.log_node_exec("summarize", state, prompt_data=msgs, output=output)
    return output

async def node_end(state):
    return {"agent_message": "Interview Complete."}

# ==============================================================================
# 6. GRAPH CONSTRUCTION
# ==============================================================================

workflow = StateGraph(InterviewState)
workflow.add_node("intent_router", node_intent_router)
workflow.add_node("handle_clarification", node_handle_clarification)
workflow.add_node("evaluate", node_evaluate)
workflow.add_node("strategize", node_strategize)
workflow.add_node("retrieve", node_retrieve)
workflow.add_node("validate", node_validate)
workflow.add_node("generate", node_generate)
workflow.add_node("summarize", node_summarize)
workflow.add_node("end", node_end)

def route_start(state):
    if not state.get("session_id"): return "strategize"
    if state.get("candidate_answer"): return "intent_router"
    return "strategize"

def route_intent(state):
    intent = state.get("user_intent", "answer")
    if intent == "clarification": return "handle_clarification"
    return "evaluate"

def route_strategy(state):
    if state['current_decision'].action == "end_interview": return "summarize"
    return "retrieve"

def route_validate(state):
    if state["is_context_valid"]:
        return "generate"
    elif state.get("retrieval_attempts", 0) < 2:
        return "retrieve"
    else:
        return "generate" # Fallback

workflow.add_conditional_edges(START, route_start, {"intent_router": "intent_router", "strategize": "strategize"})
workflow.add_conditional_edges("intent_router", route_intent, {"handle_clarification": "handle_clarification", "evaluate": "evaluate"})
workflow.add_edge("handle_clarification", END)
workflow.add_edge("evaluate", "strategize")
workflow.add_conditional_edges("strategize", route_strategy, {"summarize": "summarize", "retrieve": "retrieve"})
workflow.add_edge("retrieve", "validate")
workflow.add_conditional_edges("validate", route_validate, {"generate": "generate", "retrieve": "retrieve"})
workflow.add_edge("generate", END)
workflow.add_edge("summarize", "end")
workflow.add_edge("end", END)

memory = MemorySaver()

# ==============================================================================
# 7. API
# ==============================================================================

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.get("/")
async def get_ui(): return FileResponse("index.html")

class NextRequest(BaseModel):
    candidate_id: str
    answer: str

@app.post("/interview/start")
async def start_interview():
    session_id = str(uuid4())
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
async def next_step(req: NextRequest):
    config = {"configurable": {"thread_id": req.candidate_id}}
    app_graph = workflow.compile(checkpointer=memory)
    output = await app_graph.ainvoke({"candidate_answer": req.answer}, config=config)
    return {
        "candidate_id": req.candidate_id,
        "agent_message": output.get("agent_message"),
        "is_interview_over": output.get("is_interview_over", False)
    }

@app.get("/interview/state/{candidate_id}")
async def get_interview_state(candidate_id: str):
    turns = simple_db.get_turn_data(candidate_id)
    config = {"configurable": {"thread_id": candidate_id}}
    app_graph = workflow.compile(checkpointer=memory)
    try:
        snap = await app_graph.aget_state(config)
        summ = snap.values.get("final_summary")
    except:
        summ = None
    return {"candidate_id": candidate_id, "interview_turns": turns, "final_summary": summ}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)