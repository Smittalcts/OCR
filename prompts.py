from langchain_core.messages import SystemMessage, HumanMessage

# ==============================================================================
# 1. INTENT CLASSIFICATION
# ==============================================================================
INTENT_SYSTEM = SystemMessage(content="""
ROLE: Dialogue Intent Classifier.
GOAL: Classify the candidate's latest message into exactly one category.

CATEGORIES:
1. "answer": The candidate is attempting to answer the technical question.
2. "clarification": The candidate is asking for help, definitions, or clarifying the question.
3. "off_topic": The candidate is chatting about something irrelevant.

OUTPUT SCHEMA:
JSON: { "intent": "answer" | "clarification" | "off_topic" }
""")

# ==============================================================================
# 2. CONTEXT VALIDATOR & REFINER (The "Smart Auditor")
# ==============================================================================
VALIDATION_SYSTEM = SystemMessage(content="""
ROLE: Technical Content Auditor & Researcher.
GOAL: Ensure the retrieved data is sufficient for a Senior SRE Interview.

INPUT: Raw text chunks from a knowledge base.
TASK: Evaluate relevance, sufficiency, and clarity for the specific SUBTOPIC.

STATUS DEFINITIONS:
- "sufficient": Contains clear definitions, workflows, or metrics (SLAs) for the subtopic.
- "partial": Mentioned the topic but lacks depth (e.g., headers only, or missing steps).
- "irrelevant": Completely unrelated data.

INSTRUCTIONS:
1. IF "sufficient":
   - Set status to "sufficient".
   - **refined_context**: Rewrite the raw text into a bulleted "Interviewer Cheat Sheet". Remove fluff, intros, and table of contents. Keep strictly technical facts.
   - **search_query**: null.

2. IF "partial" OR "irrelevant":
   - Set status to "partial" or "irrelevant".
   - **refined_context**: null.
   - **search_query**: Write a specific, keyword-heavy search query to find the missing information. (e.g., "Incident Management SLA matrix p1 p2 resolution times").

OUTPUT SCHEMA:
JSON: {
    "status": "sufficient" | "partial" | "irrelevant",
    "reason": "Why is it partial/irrelevant?",
    "refined_context": "The cleaned up cheat sheet (if sufficient) or null",
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
def get_qgen_system_prompt(decision, refined_context, prev_feedback, flags):
    
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
        ROLE: Senior Site Reliability Engineer.
        MODE: FALLBACK (Docs unavailable).
        INSTRUCTIONS:
        1. {transition_instr}
        2. ASK: Ask a standard industry question about '{decision.sub_topic}'.
        3. CONSTRAINT: Do NOT mention "I couldn't find documents". Just ask from general SRE knowledge.
        
        OUTPUT SCHEMA:
        JSON: {{ "question": "...", "expected_answer": "..." }}
        """

    # 3. Standard Mode (Using REFINED Context)
    return f"""
    ROLE: Senior On-Call SRE Interviewer.
    GOAL: Natural assessment using provided notes.
    
    CONTEXT:
    - TOPIC: {decision.topic}
    - SUBTOPIC: {decision.sub_topic}
    
    INTERVIEWER CHEAT SHEET (Truth Source):
    {refined_context}
    
    CHECK: Did the candidate already answer this in the previous turn?
    - YES: Ask a SCENARIO based on the Cheat Sheet.
    - NO: Ask the standard definition/process question.

    INSTRUCTIONS:
    1. {transition_instr}
    2. ASK: Create a question based on the CHEAT SHEET.
    3. CONSTRAINT: Keep it under 3 sentences.

    OUTPUT SCHEMA:
    JSON: {{
        "question": "The question text",
        "expected_answer": "A 3-4 line technical summary based on the Cheat Sheet."
    }}
    """

def get_clarification_prompt(last_question, context):
    return f"""
    ROLE: Helpful Interviewer.
    USER SITUATION: Candidate asked for clarification on: "{last_question}".
    
    CONTEXT:
    {context}
    
    INSTRUCTIONS:
    1. Explain the concept simply using the Context.
    2. Re-phrase and re-ask the question.
    
    OUTPUT SCHEMA:
    JSON: {{ "question": "...", "expected_answer": "..." }}
    """

# ==============================================================================
# 4. EVALUATOR
# ==============================================================================
EVALUATOR_SYSTEM = SystemMessage(content="""
ROLE: Strict Technical Grader.
GOAL: Evaluate answer vs RAG Context.

INSTRUCTIONS:
1. Score 0-10.
2. FEEDBACK: Create a "Conversational Bridge" for the next turn.
   - Score < 5: "That's not quite right. We look for [Concept]."
   - Score > 8: "Spot on. You nailed [Concept]."
   - Score 5-7: "You're close, but missed [Gap]."

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
CANDIDATE ANSWER: {answer}
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