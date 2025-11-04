# langgraph_graph.py
import logging
from langgraph.graph import StateGraph, END, START
from langgraph_models import InterviewState
from langgraph_nodes import (
    retrieve_context,
    generate_question,
    evaluate_answer,
    generate_follow_up,
    calculate_topic_score_and_summarize,
    finish_interview
)

logger = logging.getLogger(__name__)

# --- LANGGRAPH ROUTERS AND EDGES ---

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

# --- BUILD THE GRAPH ---
def build_graph():
    """Builds and compiles the LangGraph."""
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
    logger.info("LangGraph workflow compiled successfully.")
    return app_graph
