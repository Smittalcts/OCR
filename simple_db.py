import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Any

DB_NAME = "interview_transcripts.db"

def init_db():
    """Initialize the specialized interview database tables."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    # Table for individual interview turns
    c.execute('''
        CREATE TABLE IF NOT EXISTS interview_turns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            timestamp TEXT,
            topic TEXT,
            sub_topic TEXT,
            question TEXT,
            expected_answer TEXT,
            user_answer TEXT,
            score INTEGER,
            evaluation_feedback TEXT,
            metadata TEXT
        )
    ''')
    
    # Table for final summaries
    c.execute('''
        CREATE TABLE IF NOT EXISTS summaries (
            session_id TEXT PRIMARY KEY,
            overall_rating TEXT,
            summary_text TEXT,
            created_at TEXT
        )
    ''')
    conn.commit()
    conn.close()

def log_turn_data(
    session_id: str, topic: str, sub_topic: str, question: str, 
    expected_answer: str, user_answer: str, score: int, 
    evaluation_feedback: str, metadata: Dict[str, Any] = None
):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    timestamp = datetime.now().isoformat()
    meta_str = json.dumps(metadata) if metadata else "{}"
    
    try:
        c.execute(
            """
            INSERT INTO interview_turns (
                session_id, timestamp, topic, sub_topic, question, 
                expected_answer, user_answer, score, evaluation_feedback, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (session_id, timestamp, topic, sub_topic, question, 
             expected_answer, user_answer, score, evaluation_feedback, meta_str)
        )
        conn.commit()
    except Exception as e:
        print(f"DB Log Error: {e}")
    finally:
        conn.close()

def save_summary(session_id: str, rating: str, text: str):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    timestamp = datetime.now().isoformat()
    try:
        c.execute(
            "INSERT OR REPLACE INTO summaries (session_id, overall_rating, summary_text, created_at) VALUES (?, ?, ?, ?)",
            (session_id, rating, text, timestamp)
        )
        conn.commit()
    except Exception as e:
        print(f"DB Summary Error: {e}")
    finally:
        conn.close()

def get_turn_data(session_id: str) -> List[Dict]:
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM interview_turns WHERE session_id = ? ORDER BY id ASC", (session_id,))
    rows = c.fetchall()
    conn.close()
    
    results = []
    for row in rows:
        data = dict(row)
        # Safely parse metadata
        if data.get('metadata'):
            try:
                data['metadata'] = json.loads(data['metadata'])
            except:
                pass
        results.append(data)
    return results