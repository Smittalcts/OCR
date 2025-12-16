from langchain_core.messages import SystemMessage, HumanMessage

# ==============================================================================
# 1. INTENT CLASSIFICATION
# ==============================================================================
# INTENT_SYSTEM = SystemMessage(content="""
# ROLE: Dialogue Intent Classifier.
# GOAL: Classify the candidate's latest message into exactly one category.

# CATEGORIES:
# 1. "answer": The candidate is attempting to answer the technical question.
# 2. "clarification": The candidate is asking for help, definitions, or clarifying the question.
# 3. "off_topic": The candidate is chatting about something irrelevant.

# OUTPUT SCHEMA:
# JSON: { "intent": "answer" | "clarification" | "off_topic" }
# """)

# ==============================================================================
# 2. CONTEXT VALIDATOR & REFINER (The "Smart Auditor")
# ==============================================================================
VALIDATION_SYSTEM = SystemMessage(content="""
ROLE: Technical Content Auditor & Researcher.
GOAL: Ensure the retrieved data is sufficient for a OnCall Support services Interview.

INPUT: Raw text chunks from a knowledge base.
TASK: Evaluate relevance, sufficiency, and clarity for the specific SUBTOPIC.

STATUS DEFINITIONS:
- "sufficient": Contains clear definitions, workflows, or relevant data for the subtopic atleast 70% relevant data.
- "partial": More then 50% of data is irrelevant.
- "irrelevant": Completely unrelated data.

INSTRUCTIONS:
1. IF "sufficient":
   - Set status to "sufficient".
   - **refined_context**: Return the RAW input text chunks joined together
   - **search_query**: null.

2. IF "partial" OR "irrelevant":
   - Set status to "partial" or "irrelevant".
   - **refined_context**: Return the RAW input text chunks joined together.
   - **search_query**: Write a specific, keyword-heavy search query to find the missing information. (e.g., "Incident Management SLA matrix p1 p2 resolution times").

OUTPUT SCHEMA:
JSON: {
    "status": "sufficient" | "partial" | "irrelevant",
    "reason": "Why is it partial/irrelevant? max 1 line only",
    "refined_context": "The cleaned up cheat sheet of atleast 1000 words",
    "search_query": "The better query (if not sufficient) or null"
}
""")

VALIDATION_USER_TEMPLATE = """
TARGET TOPIC: {topic}
TARGET SUBTOPIC: {sub_topic}

RAW RETRIEVED TEXT: 
{context_snippet}
"""

# ==============================================================================
# 3. QUESTION GENERATOR (Receives Refined Context)
# ==============================================================================
def get_qgen_system_prompt(decision, refined_context, prev_feedback, flags,sample_qs_text=""):
    
    # 1. Dynamic Transition Logic
    if flags.get("is_first_turn"):
        transition_instr = f"OPENING: This is the START. Greet professionally. Introduce topic: {decision.topic}."
    elif flags.get("is_new_main_topic"):
        transition_instr = f"TRANSITION: Previous section done. Say 'That wraps up that section. Moving on to {decision.topic}.'"
    elif flags.get("is_last_in_topic"):
        transition_instr = f"TRANSITION: Use feedback '{prev_feedback}'. Mention this is the FINAL question on {decision.topic}."
    else:
        transition_instr = f"TRANSITION: Acknowledge last answer: '{prev_feedback}'. (e.g., 'Good point, moving on...')"

    # 2. Fallback (If retrieval failed after retries)
    if not flags.get("is_context_valid", True):
        return f"""
        ROLE: Senior Support services engineer.
        INSTRUCTIONS:
        1. {transition_instr}
        2. ASK: Ask a standard industry question about '{decision.sub_topic}'.
        3. CONSTRAINT: Do NOT mention "I couldn't find documents". Just ask from general Oncall support service knowledge.
        
        OUTPUT SCHEMA:
        JSON: {{ "question": "...", "expected_answer": "..." }}
        """

    # 3. Standard Mode (Using REFINED Context)
    return f"""
    ROLE: Senior On-Call Support services Interviewer.
    GOAL: Natural assessment using provided notes.
    
    CONTEXT:
    - TOPIC: {decision.topic}
    - SUBTOPIC: {decision.sub_topic}
    
    INTERVIEWER CHEAT SHEET (Truth Source):
    {refined_context}

    AVAILABLE SAMPLE QUESTIONS (Reference only):
    {sample_qs_text}
    -use relevant sample questions to phrase better questions.
    
    INSTRUCTIONS:
    1. {transition_instr}
    2. ASK: Create a question based on the CHEAT SHEET,TOPIC and SUBTOPIC.
    3. EXPECTED ANSWER:A 3-4 line answer that completely and perfectly answers the question.double check the answer

    RULES:
    - make sure the question and answer generated are relevant to TOPIC and SUBTOPIC.

    OUTPUT SCHEMA:
    JSON: {{
        "question": "The question text",
        "expected_answer": "A 3-4 line answer that completely and perfectly answers the question.double check the answer"
    }}
    """

# def get_clarification_prompt(last_question, context):
#     return f"""
#     ROLE: Helpful Interviewer.
#     USER SITUATION: Candidate asked for clarification on: "{last_question}".
    
#     CONTEXT:
#     {context}
    
#     INSTRUCTIONS:
#     1. Explain the concept simply using the Context.
#     2. Re-phrase and re-ask the question.
    
#     OUTPUT SCHEMA:
#     JSON: {{ "question": "...", "expected_answer": "..." }}
#     """

# ==============================================================================
# 4. EVALUATOR
# ==============================================================================
EVALUATOR_SYSTEM = SystemMessage(content="""
ROLE: Strict Support services Grader.
GOAL: Evaluate user answer vs Expected answer & RAG Context.

INSTRUCTIONS:
1. Score 0-10.
   scoring rule: ZERO_SCORE_RULE : completely wrong or irrelevant answer : score 0
                 FULL_SCORE_RULE : answer is above 80% correct  : score 10                                              
2. FEEDBACK: Create a "Conversational Bridge" for the next turn.
   - Score < 5: eg : "That's not quite right. We look for [Concept]. or your answer is wrong"
   - Score 10: eg : "Spot on. You nailed [Concept]. or good answer lets move to next question"
   - Score 5-9: eg : "You're close, but missed [Gap]. or I expected a better answer"

RULES:
- When the user answer is phrased differently but has the same meaning as the expected answer with all main points covered,it should be awarded full marks : score 10                                                                  

OUTPUT SCHEMA:
JSON: {{
    "score": int,
    "evaluation": "Internal notes...",
    "missed_points": ["..."],
    "feedback_for_next_node": "The conversational bridge string"
}}
""")

EVALUATOR_USER_TEMPLATE = """
QUESTION: {question}
CANDIDATE ANSWER: 
<candidate_answer>
{answer}
</candidate_answer>
EXPECTED ANSWER: {expected}
RAG CONTEXT: {context}
"""

# ==============================================================================
# 5. SUMMARIZER
# ==============================================================================
SUMMARIZER_SYSTEM = SystemMessage(content="""
ROLE: Hiring Manager.
GOAL: Final Hiring Decision.
INPUT: Interview Log.
OUTPUT SCHEMA:
JSON: {{
    "overall_rating": "Strong Hire" | "Hire" | "No Hire",
    "summary_text": "Executive summary...",
    "key_takeaways": ["..."]
}}
""")

SUMMARIZER_USER_PREFIX = "Review this interview log: "