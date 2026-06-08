import json
import sqlite3
from pathlib import Path

DB_PATH = Path("/data/talentiq.db")
if not DB_PATH.parent.exists():
    DB_PATH = Path("./talentiq.db")


def get_db():
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    conn = get_db()
    conn.execute("""
        CREATE TABLE IF NOT EXISTS candidates (
            candidate_id TEXT PRIMARY KEY,
            data JSON NOT NULL,
            source TEXT DEFAULT 'manual',
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS jobs (
            job_id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            jd_text TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS shortlists (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id INTEGER NOT NULL,
            candidate_id TEXT NOT NULL,
            rank INTEGER,
            score REAL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(job_id, candidate_id)
        )
    """)
    conn.commit()
    conn.close()


def upsert_candidate(candidate_id: str, data: dict, source: str = "manual"):
    conn = get_db()
    conn.execute(
        "INSERT OR REPLACE INTO candidates (candidate_id, data, source) VALUES (?, ?, ?)",
        (candidate_id, json.dumps(data), source)
    )
    conn.commit()
    conn.close()


def get_all_candidates():
    conn = get_db()
    rows = conn.execute("SELECT candidate_id, data, source, created_at FROM candidates ORDER BY created_at DESC").fetchall()
    conn.close()
    return [{"candidate_id": r[0], **json.loads(r[1]), "_source": r[2], "_created": r[3]} for r in rows]


def get_candidate(candidate_id: str):
    conn = get_db()
    row = conn.execute("SELECT data FROM candidates WHERE candidate_id = ?", (candidate_id,)).fetchone()
    conn.close()
    if row:
        return json.loads(row[0])
    return None


def delete_candidate(candidate_id: str):
    conn = get_db()
    conn.execute("DELETE FROM candidates WHERE candidate_id = ?", (candidate_id,))
    conn.commit()
    conn.close()


def count_candidates():
    conn = get_db()
    row = conn.execute("SELECT COUNT(*) FROM candidates").fetchone()
    conn.close()
    return row[0]


def bulk_insert(candidates: list, source: str = "file"):
    conn = get_db()
    conn.executemany(
        "INSERT OR REPLACE INTO candidates (candidate_id, data, source) VALUES (?, ?, ?)",
        [(c["candidate_id"], json.dumps(c), source) for c in candidates]
    )
    conn.commit()
    conn.close()


def create_job(title: str, jd_text: str) -> int:
    conn = get_db()
    cur = conn.execute("INSERT INTO jobs (title, jd_text) VALUES (?, ?)", (title, jd_text))
    job_id = cur.lastrowid
    conn.commit()
    conn.close()
    return job_id


def get_all_jobs():
    conn = get_db()
    rows = conn.execute("SELECT job_id, title, jd_text, created_at FROM jobs ORDER BY created_at DESC").fetchall()
    conn.close()
    return [{"job_id": r[0], "title": r[1], "jd_text": r[2], "created_at": r[3]} for r in rows]


def get_job(job_id: int):
    conn = get_db()
    row = conn.execute("SELECT job_id, title, jd_text, created_at FROM jobs WHERE job_id = ?", (job_id,)).fetchone()
    conn.close()
    if row:
        return {"job_id": row[0], "title": row[1], "jd_text": row[2], "created_at": row[3]}
    return None


def add_to_shortlist(job_id: int, candidate_id: str, rank: int = 0, score: float = 0):
    conn = get_db()
    conn.execute("INSERT OR REPLACE INTO shortlists (job_id, candidate_id, rank, score) VALUES (?, ?, ?, ?)",
                 (job_id, candidate_id, rank, score))
    conn.commit()
    conn.close()


def remove_from_shortlist(job_id: int, candidate_id: str):
    conn = get_db()
    conn.execute("DELETE FROM shortlists WHERE job_id = ? AND candidate_id = ?", (job_id, candidate_id))
    conn.commit()
    conn.close()


def get_shortlist(job_id: int):
    conn = get_db()
    rows = conn.execute("SELECT candidate_id, rank, score FROM shortlists WHERE job_id = ? ORDER BY rank", (job_id,)).fetchall()
    conn.close()
    return [{"candidate_id": r[0], "rank": r[1], "score": r[2]} for r in rows]


def delete_job(job_id: int):
    conn = get_db()
    conn.execute("DELETE FROM shortlists WHERE job_id = ?", (job_id,))
    conn.execute("DELETE FROM jobs WHERE job_id = ?", (job_id,))
    conn.commit()
    conn.close()
