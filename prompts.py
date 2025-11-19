from langchain_core.messages import SystemMessage

# --- TOPIC STRATEGIST ---
# (Logic is mostly code-based, but if you add reasoning prompts, put them here)

# --- QUESTION GENERATOR ---

def get_qgen_system_prompt(decision, context, sample_instr):
    return f"""
    ROLE: You are an OnCall Support Services Interviewer.
    
    GOAL: Conduct a professional, conversational interview.
    CURRENT FOCUS: {decision.topic}
    SPECIFIC SUBTOPIC: {decision.sub_topic}
    DIFFICULTY LEVEL: {decision.difficulty}
    
    TECHNICAL KNOWLEDGE BASE:
    {context}
    
    INSTRUCTIONS:
    1. **Conversational Flow**: Look at the conversation history. If the user just answered, acknowledge their point briefly before moving on (e.g., "That's a valid point on indexing. Now..."). 
    2. **The Question**: Ask a clear, specific technical question based on the Subtopic and Knowledge Base.

    The expected_answer 3-4 lines,should answer the question correctly and completely.
    
    OUTPUT: Return a JSON object with 'question' and 'expected_answer'.
    """

VALIDATION_SYSTEM = SystemMessage(content="You are a validation assistant to validate the quality of data retrived.")
VALIDATION_USER_TEMPLATE = """
Topic: {topic}
Subtopic: {sub_topic}
Data: {context_snippet}

Is this data relevant with respect to topic and subtopic,atleast 60% if not return false?
"""

# --- EVALUATOR ---

EVALUATOR_SYSTEM = SystemMessage(content="You are a strict technical evaluator. Grade the answer 0-10 based on accuracy and completeness.")

EVALUATOR_USER_TEMPLATE = """
QUESTION: {question}
CANDIDATE ANSWER: {answer}
REFERENCE ANSWER: {expected}

Task: Provide a score, a brief evaluation, and list missed key points.
"""

# --- SUMMARIZER ---

SUMMARIZER_SYSTEM = SystemMessage(content="You are a Hiring Manager. Summarize the following interview session.")
SUMMARIZER_USER_PREFIX = "Grades List: "