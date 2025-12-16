import sqlite3
import json
from datetime import datetime
from typing import List, Dict, Any
import bcrypt

DB_NAME = "interview_transcripts.db"

# --- Helper: Direct Bcrypt Hashing ---
def hash_password(password: str) -> str:
    """Hash a password using bcrypt."""
    # bcrypt.hashpw requires bytes for both password and salt
    pwd_bytes = password.encode('utf-8')
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(pwd_bytes, salt)
    return hashed.decode('utf-8')  # Store as string in DB

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

    # Table for Users
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL
        )
    ''')

    # Table for Sessions (Links Session ID to User)
    c.execute('''
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            username TEXT NOT NULL,
            start_time TEXT NOT NULL
        )
    ''')

    # Ensure Fixed Manager Exists & Force Update Role
    manager_user = "admin"
    manager_pass = "admin123"
    manager_hash = hash_password(manager_pass)
    
    try:
        c.execute("SELECT role FROM users WHERE username = ?", (manager_user,))
        existing = c.fetchone()
        if existing:
            c.execute("UPDATE users SET role = ?, password_hash = ? WHERE username = ?", 
                      ("manager", manager_hash, manager_user))
        else:
            c.execute("INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)", 
                      (manager_user, manager_hash, "manager"))
    except Exception as e:
        print(f"Error initializing admin user: {e}")

    conn.commit()
    conn.close()

# --- Logging Functions ---

def log_session_start(session_id: str, username: str):
    """Logs the start of a session mapping it to a user."""
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    timestamp = datetime.now().isoformat()
    try:
        c.execute("INSERT INTO sessions (session_id, username, start_time) VALUES (?, ?, ?)",
                  (session_id, username, timestamp))
        conn.commit()
    except Exception as e:
        print(f"DB Session Log Error: {e}")
    finally:
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

# --- Retrieval Functions ---

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
        if data.get('metadata'):
            try:
                data['metadata'] = json.loads(data['metadata'])
            except:
                pass
        results.append(data)
    return results

def get_summary(session_id: str) -> Dict:
    """Retrieves the specific summary row for a session."""
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM summaries WHERE session_id = ?", (session_id,))
    row = c.fetchone()
    conn.close()
    return dict(row) if row else None

def get_all_sessions_for_admin() -> List[Dict]:
    """Retrieves all sessions joined with summary data for the admin dashboard."""
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    query = """
        SELECT 
            s.session_id, 
            s.username, 
            s.start_time,
            sum.overall_rating,
            sum.created_at as completed_at
        FROM sessions s
        LEFT JOIN summaries sum ON s.session_id = sum.session_id
        ORDER BY s.start_time DESC
    """
    try:
        c.execute(query)
        rows = c.fetchall()
        return [dict(row) for row in rows]
    except Exception as e:
        print(f"DB Admin Fetch Error: {e}")
        return []
    finally:
        conn.close()

# --- User Auth Functions ---

def create_user(username, password, role="candidate"):
    conn = sqlite3.connect(DB_NAME)
    c = conn.cursor()
    password_hash = hash_password(password)
    try:
        c.execute("INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)", 
                  (username, password_hash, role))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False 
    finally:
        conn.close()

def get_user(username):
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    c = conn.cursor()
    c.execute("SELECT * FROM users WHERE username = ?", (username,))
    user = c.fetchone()
    conn.close()
    return dict(user) if user else None