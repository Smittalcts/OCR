# database.py
import sqlite3
import json
import logging
from typing import List, Dict, Any

logger = logging.getLogger(__name__)

DATABASE_FILE = "interview_agent.db"

def get_db_connection():
    """Establishes a connection to the SQLite database."""
    conn = sqlite3.connect(DATABASE_FILE)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Initializes the database and creates tables if they don't exist."""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    # --- User and Role Management ---
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        username TEXT UNIQUE NOT NULL,
        hashed_password TEXT NOT NULL,
        role TEXT NOT NULL CHECK(role IN ('associate', 'manager'))
    )
    """)
    
    # --- Interview Scheduling ---
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS interviews (
        id TEXT PRIMARY KEY,
        associate_id INTEGER NOT NULL,
        manager_id INTEGER NOT NULL,
        topics TEXT NOT NULL,
        status TEXT NOT NULL DEFAULT 'pending',
        final_score INTEGER,
        max_score INTEGER,
        final_summary TEXT,
        created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (associate_id) REFERENCES users (id),
        FOREIGN KEY (manager_id) REFERENCES users (id)
    )
    """)
    
    # --- LangGraph Data Logging (Linked to Interviews) ---
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS questions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        interview_id TEXT NOT NULL,
        topic TEXT NOT NULL,
        question TEXT NOT NULL,
        expected_answer TEXT NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (interview_id) REFERENCES interviews (id)
    )
    """)
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS answers (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        interview_id TEXT NOT NULL,
        topic TEXT NOT NULL,
        user_answer TEXT NOT NULL,
        feedback TEXT NOT NULL,
        score INTEGER NOT NULL,
        status TEXT NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (interview_id) REFERENCES interviews (id)
    )
    """)
    
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS topic_results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        interview_id TEXT NOT NULL,
        topic TEXT NOT NULL,
        score INTEGER NOT NULL,
        summary TEXT NOT NULL,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (interview_id) REFERENCES interviews (id)
    )
    """)
    
    conn.commit()
    conn.close()
    logger.info("Database initialized successfully.")

# --- User Functions ---

def create_user(username: str, hashed_password: str, role: str):
    conn = get_db_connection()
    try:
        conn.execute(
            "INSERT INTO users (username, hashed_password, role) VALUES (?, ?, ?)",
            (username, hashed_password, role)
        )
        conn.commit()
    except sqlite3.IntegrityError:
        return None
    finally:
        conn.close()
    return get_user_by_username(username)

def get_user_by_username(username: str) -> Dict[str, Any]:
    conn = get_db_connection()
    user = conn.execute("SELECT * FROM users WHERE username = ?", (username,)).fetchone()
    conn.close()
    return dict(user) if user else None

def get_all_associates() -> List[Dict[str, Any]]:
    conn = get_db_connection()
    associates = conn.execute("SELECT id, username FROM users WHERE role = 'associate'").fetchall()
    conn.close()
    return [dict(a) for a in associates]

# --- Interview Functions ---

def schedule_interview(interview_id: str, associate_id: int, manager_id: int, topics: List[str]) -> Dict[str, Any]:
    conn = get_db_connection()
    topics_json = json.dumps(topics)
    conn.execute(
        "INSERT INTO interviews (id, associate_id, manager_id, topics, status) VALUES (?, ?, ?, ?, 'pending')",
        (interview_id, associate_id, manager_id, topics_json)
    )
    conn.commit()
    interview = conn.execute("SELECT * FROM interviews WHERE id = ?", (interview_id,)).fetchone()
    conn.close()
    return dict(interview)

def get_associate_interviews(associate_id: int) -> Dict[str, List[Dict[str, Any]]]:
    conn = get_db_connection()
    pending = conn.execute(
        "SELECT id, topics, created_at FROM interviews WHERE associate_id = ? AND status = 'pending' ORDER BY created_at DESC",
        (associate_id,)
    ).fetchall()
    completed = conn.execute(
        "SELECT id, topics, final_score, max_score, final_summary FROM interviews WHERE associate_id = ? AND status = 'completed' ORDER BY created_at DESC",
        (associate_id,)
    ).fetchall()
    conn.close()
    return {
        "pending": [dict(p) for p in pending],
        "completed": [dict(c) for c in completed]
    }

def get_manager_dashboard_data(manager_id: int) -> List[Dict[str, Any]]:
    conn = get_db_connection()
    results = conn.execute("""
        SELECT i.id, i.status, i.final_score, i.max_score, i.created_at, u.username as associate_username
        FROM interviews i
        JOIN users u ON i.associate_id = u.id
        WHERE i.manager_id = ?
        ORDER BY i.created_at DESC
    """, (manager_id,)).fetchall()
    conn.close()
    return [dict(r) for r in results]

def get_interview_details(interview_id: str, user_id: int) -> Dict[str, Any]:
    conn = get_db_connection()
    interview = conn.execute(
        "SELECT * FROM interviews WHERE id = ? AND associate_id = ?",
        (interview_id, user_id)
    ).fetchone()
    conn.close()
    if not interview:
        return None
    details = dict(interview)
    details['topics'] = json.loads(details['topics'])
    return details

def set_interview_status(interview_id: str, status: str):
    conn = get_db_connection()
    conn.execute("UPDATE interviews SET status = ? WHERE id = ?", (status, interview_id))
    conn.commit()
    conn.close()

def update_interview_on_finish(interview_id: str, total_score: int, max_score: int, summary: str):
    conn = get_db_connection()
    conn.execute(
        "UPDATE interviews SET status = 'completed', final_score = ?, max_score = ?, final_summary = ? WHERE id = ?",
        (total_score, max_score, summary, interview_id)
    )
    conn.commit()
    conn.close()

# --- LangGraph Data Logging Functions ---

def add_question(session_id: str, topic: str, question: str, expected_answer: str):
    conn = get_db_connection()
    conn.execute(
        "INSERT INTO questions (interview_id, topic, question, expected_answer) VALUES (?, ?, ?, ?)",
        (session_id, topic, question, expected_answer)
    )
    conn.commit()
    conn.close()

def add_answer_and_evaluation(session_id: str, topic: str, user_answer: str, feedback: str, score: int, status: str):
    conn = get_db_connection()
    conn.execute(
        "INSERT INTO answers (interview_id, topic, user_answer, feedback, score, status) VALUES (?, ?, ?, ?, ?, ?)",
        (session_id, topic, user_answer, feedback, score, status)
    )
    conn.commit()
    conn.close()

def get_topic_history_as_string(session_id: str, topic: str) -> str:
    conn = get_db_connection()
    questions = conn.execute(
        "SELECT question, expected_answer FROM questions WHERE interview_id = ? AND topic = ? ORDER BY timestamp",
        (session_id, topic)
    ).fetchall()
    answers = conn.execute(
        "SELECT user_answer, feedback, score, status FROM answers WHERE interview_id = ? AND topic = ? ORDER BY timestamp",
        (session_id, topic)
    ).fetchall()
    conn.close()

    transcript = []
    for i in range(len(questions)):
        q = questions[i]
        transcript.append(f"AI_QUESTION: {q['question']}")
        if i < len(answers):
            a = answers[i]
            transcript.append(f"USER_ANSWER: {a['user_answer']}")
            transcript.append(f"AI_FEEDBACK: {a['feedback']} (Score: {a['score']}/10, Status: {a['status']})")
    
    return "\n".join(transcript)

def get_scores_for_topic(session_id: str, topic: str) -> List[int]:
    conn = get_db_connection()
    scores = conn.execute(
        "SELECT score FROM answers WHERE interview_id = ? AND topic = ? ORDER BY timestamp",
        (session_id, topic)
    ).fetchall()
    conn.close()
    return [s['score'] for s in scores]

def add_topic_result(session_id: str, topic: str, score: int, summary: str):
    conn = get_db_connection()
    conn.execute(
        "INSERT INTO topic_results (interview_id, topic, score, summary) VALUES (?, ?, ?, ?)",
        (session_id, topic, score, summary)
    )
    conn.commit()
    conn.close()

def get_all_topic_results_for_session(session_id: str) -> List[Dict[str, Any]]:
    conn = get_db_connection()
    results = conn.execute(
        "SELECT topic, score AS final_score, summary FROM topic_results WHERE interview_id = ?",
        (session_id,)
    ).fetchall()
    conn.close()
    return [dict(r) for r in results]