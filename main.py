import os
import uvicorn
import logging
import sys
import json
import operator
from datetime import datetime
import httpx
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
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
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

class GeneratedQuestion(BaseModel):
    conversational_entry: str = Field(..., description="The social bridge")
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

# class AzureResponsesChat(BaseChatModel):
#     """
#     Custom Adapter for Azure 'Responses API' (Preview).
#     Docs: https://learn.microsoft.com/en-us/azure/ai-services/openai/how-to/responses-api
#     """
#     endpoint: str
#     api_key: str
#     api_version: str = "2025-03-01-preview" 
#     deployment_name: str
    
#     def _generate(self, messages, stop=None, run_manager=None, **kwargs):
#         raise NotImplementedError("Use ainvoke() for async execution")

#     @property
#     def _llm_type(self):
#         return "azure-responses-custom"

#     async def _agenerate(self, messages: List[BaseMessage], stop=None, run_manager=None, **kwargs) -> ChatResult:
#         # 1. Format Input as List of Dictionaries
#         formatted_input = []
#         for m in messages:
#             role = "user"
#             if isinstance(m, AIMessage): role = "assistant"
#             elif isinstance(m, SystemMessage): role = "developer"
#             else: role = "user"
            
#             formatted_input.append({"role": role, "content": m.content})

#         # 2. Construct URL Robustly
#         # Extract strictly the base (https://resource.azure.com) to avoid path duplication
#         parsed_url = urlparse(self.endpoint)
#         base_url = urlunparse((parsed_url.scheme, parsed_url.netloc, "", "", "", ""))
        
#         # Construct the correct Responses API path
#         url = f"{base_url}/openai/v1/responses?api-version={self.api_version}"

#         # 3. Request Payload
#         payload = {
#             "model": self.deployment_name,
#             "input": formatted_input,
#             "stream": False
#         }

#         # 4. Execute Request
#         headers = {
#             "Content-Type": "application/json",
#             "api-key": self.api_key
#         }
        
#         async with httpx.AsyncClient() as client:
#             try:
#                 resp = await client.post(url, json=payload, headers=headers, timeout=60.0)
#                 resp.raise_for_status()
#                 data = resp.json()
                
#                 # 5. Parse Output
#                 content = ""
#                 if "output" in data and len(data["output"]) > 0:
#                     first_item = data["output"][0]
#                     if "content" in first_item and len(first_item["content"]) > 0:
#                         content = first_item["content"][0].get("text", "")
                
#                 if not content:
#                     sys_logger.error(f"Unexpected JSON structure: {data}")
#                     content = "Error: Empty response from model."

#                 return ChatResult(generations=[ChatGeneration(message=AIMessage(content=content))])
                
#             except httpx.HTTPStatusError as e:
#                 sys_logger.error(f"Azure API Error: {e.response.text}")
#                 raise ValueError(f"Azure API Error {e.response.status_code}: {e.response.text}")


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
# --- REPLACED CLIENT ---
llm_client = AzureChatOpenAI(
    azure_deployment=os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT_NAME"), # e.g. "gpt-5.1-chat"
    api_key=os.environ.get("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.environ.get("AZURE_OPENAI_ENDPOINT"), # Must be: https://YOUR-RESOURCE.cognitiveservices.azure.com/
    api_version="2024-08-01-preview", # Stable version for Global Standard
   
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
    

# Helper to inject instructions and parse manually
def create_robust_chain(llm, pydantic_cls):
    from langchain_core.output_parsers import PydanticOutputParser
    from langchain_core.runnables import RunnableLambda
    
    parser = PydanticOutputParser(pydantic_object=pydantic_cls)
    
    def _inject_instructions(inputs):
        msgs = list(inputs)
        # We append instructions to the last message content
        # This ensures the model sees the schema requirement
        instr = f"\n\nIMPORTANT: Return valid JSON matching this schema:\n{parser.get_format_instructions()}"
        
        if msgs and isinstance(msgs[-1], HumanMessage):
            # Modify last user message
            existing = msgs[-1].content
            msgs[-1] = HumanMessage(content=existing + instr)
        else:
            # Fallback
            msgs.append(HumanMessage(content=f"Instructions: {instr}"))
        return msgs

    return RunnableLambda(_inject_instructions) | llm | parser

# Define chains
llm_eval = llm_client.with_structured_output(EvaluationResult)
llm_gen = llm_client.with_structured_output(GeneratedQuestion)
llm_sum = llm_client.with_structured_output(FinalSummary)
llm_check = llm_client.with_structured_output(RelevanceCheck)

# ==============================================================================
# 5. NODES (Refactored)
# ==============================================================================

# --- NODE: EVALUATE ---
async def node_evaluate(state: InterviewState) -> Dict[str, Any]:
    # 1. Check if we are in Round 2 (Probe Mode)
    if state.get("is_in_probe_mode"):
        # MERGE ANSWERS: Context from Turn 1 + New Answer from Turn 2
        full_answer = f"{state['partial_answer_context']} . Follow-up clarification: {state['candidate_answer']}"
        question = state['original_question']
        expected = state['original_expected_answer']
        
        # Force decision to "accepted" because we don't want infinite loops
        forced_decision = True 
    else:
        # Round 1 (Standard)
        full_answer = state['candidate_answer']
        question = state.get('last_question', "Intro")
        expected = state.get('last_expected_answer', "N/A")
        forced_decision = False

    context = state.get('refined_context') or state.get('retrieved_context', "N/A")
    
    # Call LLM
    user_msg_content = prompts.EVALUATOR_USER_TEMPLATE.format(
        question=question, answer=full_answer, expected=expected, context=context
    )
    msgs = [prompts.EVALUATOR_SYSTEM, HumanMessage(content=user_msg_content)]
    eval_result = await llm_eval.ainvoke(msgs)
    
    # OUTPUT LOGIC
    output = {}
    
    # CASE A: Needs Probe (And not already forced)
    if eval_result.decision == "needs_probe" and not forced_decision:
        output = {
            "is_in_probe_mode": True,
            "partial_answer_context": full_answer,
            "original_question": question,
            "original_expected_answer": expected,
            "last_missed_points": eval_result.missed_points, # Pass to FollowUp Node
            # We do NOT log to DB yet. We wait for the final score.
        }
        sys_logger.info(f"🧐 EVALUATOR: Partial answer detected. Triggering Probe.")
        
    # CASE B: Accepted (Or Round 2 Complete)
    else:
        # Now we Log to DB (It's the final verdict)
        simple_db.log_turn_data(
            session_id=state['session_id'],
            topic=state['current_decision'].topic,
            sub_topic=state['current_decision'].sub_topic,
            question=question,
            expected_answer=expected,
            user_answer=full_answer, # Log the MERGED answer
            score=eval_result.score,
            evaluation_feedback=eval_result.evaluation,
            metadata=eval_result.model_dump()
        )
        
        output = {
            "is_in_probe_mode": False, # Reset Flag
            "graded_answers": [{"score": eval_result.score, "topic": state['current_decision'].topic}],
            "last_turn_feedback": eval_result.feedback_for_next_node,
            "message_history": state.get('message_history', []) + [HumanMessage(content=state['candidate_answer'])]
        }
        sys_logger.info(f"✅ EVALUATOR: Final Score {eval_result.score}")

    return output

# ---NODE: FOLLOW_UP ---
# In main.py

async def node_ask_followup(state: InterviewState) -> Dict[str, Any]:
    # 1. Retrieve Data from State
    missed = state.get("last_missed_points", [])
    original_q = state.get("original_question")
    original_expected = state.get("original_expected_answer") # <--- NOW USING THIS
    user_partial = state.get("partial_answer_context")

    # 2. Build Prompt
    # We give the LLM the "Truth" (Expected) so it knows what to hint at without giving it away.
    prompt = f"""
    ROLE: Senior Interviewer.
    GOAL: Ask a follow-up to a partial answer.
    
    CONTEXT:
    - Question Asked: "{original_q}"
    - User's Partial Answer: "{user_partial}"
    - The Missing Truth (Expected): "{original_expected}"
    - Specific Missed Concepts: {missed}
    
    INSTRUCTIONS:
    1. Acknowledge the part they got right (briefly).
    2. Ask a targeted follow-up question that nudges them toward the "Missing Truth".
    3. CONSTRAINT: Do NOT reveal the answer. Just ask for clarification.
    
    EXAMPLE:
    "You correctly identified X, but how does Y fit into this process?"
    """
    
    # 3. Generate Question
    msg = await llm_client.ainvoke([SystemMessage(content=prompt)])
    
    # 4. Update History (So the user sees the question)
    new_hist = state.get('message_history', [])
    new_hist.append(AIMessage(content=msg.content))
    
    # 5. Return (This stops the graph and sends 'agent_message' to UI)
    return {
        "agent_message": msg.content,
        "message_history": new_hist
    }

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
    
    # Use template from prompts.py
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

# --- NODE: GENERATE ---
async def node_generate(state: InterviewState) -> Dict[str, Any]:
    decision = state['current_decision']
    refined_context = state.get('refined_context')
    prev_feedback = state.get('last_turn_feedback', "Let's begin.")
    is_valid = state.get('is_context_valid', True)
    
    # Fetch Sample Questions from TOPIC_DATA
    topic_info = TOPIC_DATA.get(decision.topic, {})
    sample_questions = topic_info.get("sample_questions", [])
    sample_qs_text = "\n".join([f"- {q}" for q in sample_questions])
    
    flags = {
        "is_first_turn": decision.is_first_turn,
        "is_new_main_topic": decision.is_new_main_topic,
        "is_last_in_topic": decision.is_last_in_topic,
        "is_context_valid": is_valid
    }
    
    # Call the logic from prompts.py with sample questions
    prompt_str = prompts.get_qgen_system_prompt(decision, refined_context, prev_feedback, flags, sample_qs_text)
    
    msgs = [SystemMessage(content=prompt_str)]
    
    gen = await llm_gen.ainvoke(msgs)
    
    # Log usage of sample questions
    if gen.question_source == "sample_question":
        sys_logger.info(f"📌 Using Sample Question for {decision.sub_topic}: {gen.technical_question}")
    else:
        sys_logger.info(f"⚡ Generating New Question for {decision.sub_topic}")

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

# ... existing imports ...

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
        total_max_score += 10 # Assuming 10 is max per question

    # Calculate averages per topic
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

# ... rest of code ...

# --- NODE: SUMMARIZE ---
async def node_summarize(state: InterviewState) -> Dict[str, Any]:
    all_turns = simple_db.get_turn_data(state['session_id'])
    
    # 1. Calculate Metrics Pythonically
    metrics = calculate_interview_metrics(all_turns)
    
    # 2. Prepare Context for LLM
    log_str = str([{ "topic": t['topic'], "score": t['score'], "feedback": t['evaluation_feedback']} for t in all_turns])
    stats_summary = f"\nSTATS: Total Score: {metrics['total_score']}/{metrics['max_possible_score']} ({metrics['overall_percentage']}%)"
    
    # 3. Invoke LLM
    msgs = [
        prompts.SUMMARIZER_SYSTEM, 
        HumanMessage(content=f"{prompts.SUMMARIZER_USER_PREFIX}{log_str} {stats_summary}")
    ]
    
    summary = await llm_sum.ainvoke(msgs)
    simple_db.save_summary(state['session_id'], summary.overall_rating, summary.summary_text)
    
    # 4. Return combined data (LLM Summary + Calculated Metrics)
    final_output = summary.model_dump()
    final_output["metrics"] = metrics # Inject metrics here
    
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
        return "generate" # Fallback
    
def route_evaluate(state):
    if state.get("is_in_probe_mode"):
        return "ask_followup"
    return "strategize"    

# Simplified Edges (No Intent Router)
workflow.add_conditional_edges(START, route_start, {"evaluate": "evaluate", "strategize": "strategize"})
# workflow.add_edge("evaluate", "strategize")
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
    
    final_summary_data = None
    try:
        snap = await app_graph.aget_state(config)
        # Check if final_summary exists in state
        if snap.values.get("final_summary"):
            final_summary_data = snap.values.get("final_summary")
            
            # If metrics weren't saved in state (e.g. older sessions), calculate them now
            if "metrics" not in final_summary_data:
                final_summary_data["metrics"] = calculate_interview_metrics(turns)
    except:
        pass
        
    return {"candidate_id": candidate_id, "interview_turns": turns, "final_summary": final_summary_data}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)