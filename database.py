import psycopg2
from psycopg2.extras import RealDictCursor
import os
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")

def get_db_connection():
    conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
    return conn

def init_db():
    conn = get_db_connection()
    cur = conn.cursor()
    
    cur.execute('''
        DROP TABLE IF EXISTS feedback;
        CREATE TABLE IF NOT EXISTS feedback (
            id SERIAL PRIMARY KEY,
            rating INTEGER NOT NULL CHECK (rating BETWEEN 1 AND 5),
            feedback_text TEXT,
            submitted_text TEXT,
            timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    cur.close()
    conn.close()


def save_feedback(rating: int, feedback_text: str, submitted_text: str):
    """Save user feedback and analyzed text to PostgreSQL."""
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO feedback (rating, feedback_text, submitted_text)
        VALUES (%s, %s, %s)
        """,
        (rating, feedback_text, submitted_text)
    )
    conn.commit()
    cur.close()
    conn.close()