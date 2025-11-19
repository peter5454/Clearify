import psycopg2
from psycopg2.extras import RealDictCursor
import logging
import os

# ---------------- Logging Setup ---------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# ---------------- Database URL ---------------- #
DATABASE_URL = os.getenv("DATABASE_URL")

# ---------------- DB Connection ---------------- #
def get_db_connection():
    """
    Establishes a database connection using the securely provided DATABASE_URL.
    Raises an error if the URL is not available.
    """
    if not DATABASE_URL:
        raise ConnectionError("DATABASE_URL environment variable is not set. Cannot connect to the database.")
    conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
    return conn

# ---------------- Save Feedback ---------------- #
def save_feedback(rating: int, feedback_text: str, submitted_text: str):
    """Save user feedback and log success/failure."""
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
        logger.info("Feedback saved successfully: rating=%s", rating)
    except ConnectionError as e:
        logger.exception("Database connection error when saving feedback: %s", e)
    except Exception as e:
        logger.exception("Database error when saving feedback: %s", e)
        if conn:
            conn.rollback()
    finally:
        if conn:
            conn.close()

# ---------------- Database Health Check ---------------- #
def check_db_health():
    """
    Simple database health check.
    Returns True if the DB is reachable, False otherwise.
    """
    conn = None
    try:
        conn = get_db_connection()
        cur = conn.cursor()
        cur.execute("SELECT 1")  # Simple query to test connectivity
        cur.close()
        logger.info("Database health check passed.")
        return True
    except Exception as e:
        logger.exception("Database health check failed: %s", e)
        return False
    finally:
        if conn:
            conn.close()
