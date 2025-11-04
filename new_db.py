import sqlite3
import logging
from typing import List, Dict

# Configure logger for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DB_NAME = "interview_history.db"

def get_db_connection():
    """Establishes a connection to the SQLite database."""
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes the database and creates tables if they don't exist."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Table for individual question-answer interactions
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS interactions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                topic TEXT NOT NULL,
                question TEXT NOT NULL,
                expected_answer TEXT,
                user_answer TEXT,
                evaluation_feedback TEXT,
                evaluation_score INTEGER,
                evaluation_status TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # New table for final topic scores and summaries
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS topic_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT NOT NULL,
                topic TEXT NOT NULL,
                final_score INTEGER NOT NULL,
                summary TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
        logger.info("Database initialized successfully with 'interactions' and 'topic_results' tables.")
    except Exception as e:
        logger.error(f"Error initializing database: {e}", exc_info=True)

def add_question(session_id: str, topic: str, question: str, expected_answer: str) -> int:
    """Adds a new question to the database and returns the record ID."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO interactions (session_id, topic, question, expected_answer) VALUES (?, ?, ?, ?)",
        (session_id, topic, question, expected_answer)
    )
    new_id = cursor.lastrowid
    conn.commit()
    conn.close()
    logger.info(f"Added question for session {session_id} to DB (ID: {new_id}).")
    return new_id

def add_answer_and_evaluation(
    session_id: str, 
    topic: str, 
    user_answer: str, 
    feedback: str, 
    score: int, 
    status: str
):
    """Updates the most recent unanswered question for a topic with the user's answer and evaluation."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        """
        UPDATE interactions 
        SET user_answer = ?, evaluation_feedback = ?, evaluation_score = ?, evaluation_status = ?
        WHERE id = (
            SELECT id FROM interactions 
            WHERE session_id = ? AND topic = ? AND user_answer IS NULL 
            ORDER BY timestamp DESC 
            LIMIT 1
        )
        """,
        (user_answer, feedback, score, status, session_id, topic)
    )
    conn.commit()
    conn.close()
    logger.info(f"Updated answer and evaluation for session {session_id}, topic '{topic}'.")

def add_topic_result(session_id: str, topic: str, score: int, summary: str):
    """Adds the final calculated score and summary for a topic to the database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO topic_results (session_id, topic, final_score, summary) VALUES (?, ?, ?, ?)",
        (session_id, topic, score, summary)
    )
    conn.commit()
    conn.close()
    logger.info(f"Saved final topic result for session {session_id}, topic '{topic}' to DB.")

def get_topic_history_as_string(session_id: str, topic: str) -> str:
    """Retrieves all interactions for a topic and formats them into a single string for prompts."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM interactions WHERE session_id = ? AND topic = ? ORDER BY timestamp ASC",
        (session_id, topic)
    )
    rows = cursor.fetchall()
    conn.close()

    history = []
    for i, row in enumerate(rows):
        history.append(f"Turn {i+1}:")
        history.append(f"  AI Question: {row['question']}")
        if row['user_answer']:
            history.append(f"  User Answer: {row['user_answer']}")
        if row['expected_answer']:
            history.append(f"  Expected answer: {row['expected_answer']}")
        if row['evaluation_feedback']:
            history.append(f"  AI Feedback: {row['evaluation_feedback']} (Score: {row['evaluation_score']})")
        history.append("-" * 20)
    
    formatted_history = "\n".join(history)
    return formatted_history

def get_scores_for_topic(session_id: str, topic: str) -> List[int]:
    """Retrieves all non-null evaluation scores for a given topic in a session."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT evaluation_score FROM interactions WHERE session_id = ? AND topic = ? AND evaluation_score IS NOT NULL ORDER BY timestamp ASC",
        (session_id, topic)
    )
    scores = [row[0] for row in cursor.fetchall()]
    conn.close()
    return scores

def get_all_topic_results_for_session(session_id: str) -> List[Dict]:
    """Retrieves all final topic results for a given session from the database."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT topic, final_score, summary FROM topic_results WHERE session_id = ? ORDER BY timestamp ASC",
        (session_id,)
    )
    results = [dict(row) for row in cursor.fetchall()]
    conn.close()
    logger.info(f"Retrieved {len(results)} final topic results for session {session_id}.")
    return results

