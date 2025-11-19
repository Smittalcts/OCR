import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Any

DB_NAME = "interview_transcripts.db"

def init_db():
    """Initialize the simple text-based database."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    
    # Table for individual chat messages/turns
    c.execute('''
        CREATE TABLE IF NOT EXISTS transcripts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT,
            timestamp TEXT,
            role TEXT,
            content TEXT,
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

def log_message(session_id: str, role: str, content: str, metadata: Dict[str, Any] = None):
    """Logs a message to the DB in plain text."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    timestamp = datetime.now().isoformat()
    meta_str = json.dumps(metadata) if metadata else "{}"
    
    try:
        c.execute(
            "INSERT INTO transcripts (session_id, timestamp, role, content, metadata) VALUES (?, ?, ?, ?, ?)",
            (session_id, timestamp, role, content, meta_str)
        )
        conn.commit()
    except Exception as e:
        print(f"DB Error: {e}")
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
        print(f"DB Error: {e}")
    finally:
        conn.close()

def get_transcript(session_id: str) -> List[Dict]:
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM transcripts WHERE session_id = ? ORDER BY id ASC", (session_id,))
    rows = c.fetchall()
    conn.close()
    return [dict(row) for row in rows]