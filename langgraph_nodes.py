# langgraph_nodes.py
import logging
from langchain_core.prompts import ChatPromptTemplate

# Import services and models
from service import llm, vector_store_manager
import database
from langgraph_models import (
    InterviewState, QuestionAndAnswer, Evaluation, 
    FollowUpQuestionAndAnswer, TopicSummary, TopicScore
)

logger = logging.getLogger(__name__)

# --- LANGGRAPH NODES ---

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
    
    question_id = database.add_question(
        session_id=state["session_id"],
        topic=state["current_topic"],
        question=qa_pair.question,
        expected_answer=qa_pair.expected_answer,
        question_type='main'
    )
    
    return {
        "current_question": qa_pair.question, 
        "current_expected_answer": qa_pair.expected_answer,
        "current_question_id": question_id,
        "history": state["history"] + [f"AI_QUESTION: {qa_pair.question}"],
        "api_call_count": 1
    }

def evaluate_answer(state: InterviewState):
    logger.info(f"--- Entering NODE: evaluate_answer ---")
    topic_history_str = database.get_topic_history_as_string(state["session_id"], state["current_topic"])
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
    
    database.add_answer_and_evaluation(
        session_id=state['session_id'],
        question_id=state['current_question_id'],
        topic=state['current_topic'],
        user_answer=user_answer,
        feedback=feedback.feedback,
        score=feedback.score,
        status=feedback.status
    )
    
    return {"latest_evaluation": feedback, "api_call_count": 1}

def generate_follow_up(state: InterviewState):
    logger.info(f"--- Entering NODE: generate_follow_up ---")
    topic_history_str = database.get_topic_history_as_string(state["session_id"], state["current_topic"])
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
    
    follow_up_num = state["follow_up_count"] + 1
    question_type = f"followup_{follow_up_num}"
    
    question_id = database.add_question(
        session_id=state['session_id'],
        topic=state['current_topic'],
        question=follow_up_qa_pair.question,
        expected_answer=follow_up_qa_pair.expected_answer,
        question_type=question_type
    )
    
    return {
        "current_question": follow_up_qa_pair.question,
        "current_expected_answer": follow_up_qa_pair.expected_answer,
        "current_question_id": question_id,
        "history": state["history"] + [f"AI_QUESTION: {follow_up_qa_pair.question}"],
        "follow_up_count": follow_up_num,
        "api_call_count": 1
    }

def calculate_topic_score_and_summarize(state: InterviewState):
    logger.info(f"--- Entering NODE: calculate_topic_score_and_summarize ---")
    
    session_id = state["session_id"]
    topic = state["current_topic"]

    topic_history_str = database.get_topic_history_as_string(session_id, topic)
    logger.info(f"Retrieved history for summary prompt:\n---\n{topic_history_str}\n---")
    
    scores = database.get_scores_for_topic(session_id, topic)
    logger.info(f"Retrieved scores for calculation: {scores}")
    
    final_score = 0
    if not scores:
        final_score = 5
        logger.warning(f"No scores found for topic '{topic}', defaulting to 5.")
    else:
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

    database.add_topic_result(
        session_id=session_id,
        topic=final_topic_score.topic,
        score=final_topic_score.score,
        summary=final_topic_score.summary
    )

    return {"topic_scores": state["topic_scores"] + [final_topic_score], "api_call_count": 1}

def finish_interview(state: InterviewState):
    logger.info(f"--- Entering NODE: finish_interview ---")
    
    session_id = state['session_id']
    final_topic_scores = database.get_all_topic_results_for_session(session_id)
    
    if not final_topic_scores:
        logger.warning(f"No topic results found in DB for session {session_id}. Final score is 0.")
        total_score = 0
        max_score = 0
    else:
        total_score = sum(topic['final_score'] for topic in final_topic_scores)
        max_score = len(final_topic_scores) * 10

    summary = f"Interview complete. Final Score: {total_score} out of {max_score}."
    
    database.update_interview_on_finish(
        interview_id=session_id,
        total_score=total_score,
        max_score=max_score,
        summary=summary
    )
    
    logger.info(summary)
    logger.info(f"Total API calls made during this session: {state.get('api_call_count', 0)}")
    return {"final_summary": summary}
