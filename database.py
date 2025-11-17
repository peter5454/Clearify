import psycopg2
from psycopg2.extras import RealDictCursor
import os
# CRITICAL CHANGE: DATABASE_URL is now read directly from the environment
# Cloud Run will inject the value from Secret Manager into this environment variable.
DATABASE_URL = os.getenv("DATABASE_URL")

def get_db_connection():
    """
    Establishes a database connection using the securely provided DATABASE_URL.
    Raises an error if the URL is not available.
    """
    if not DATABASE_URL:
        raise ConnectionError("DATABASE_URL environment variable is not set. Cannot connect to the database.")
        

    conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
    return conn



def save_feedback(rating: int, feedback_text: str, submitted_text: str):
    """Save user feedback and analyzed text to PostgreSQL."""
    conn = None
    try:
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
    except ConnectionError as e:
        print(f"Error saving feedback: {e}")
        # Depending on your app, you might want to return an error or status here
    except Exception as e:
        print(f"Database error saving feedback: {e}")
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()

