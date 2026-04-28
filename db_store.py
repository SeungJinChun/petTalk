from __future__ import annotations

import json
import logging
import os
from contextlib import contextmanager
from datetime import date, datetime
from typing import Any, Iterable

logger = logging.getLogger(__name__)


def database_enabled() -> bool:
    return bool(os.environ.get("DATABASE_URL"))


def _database_url() -> str:
    url = os.environ.get("DATABASE_URL", "").strip()
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    # Supabase pooled/direct URLs usually require SSL.
    if url and "sslmode=" not in url:
        joiner = "&" if "?" in url else "?"
        url = f"{url}{joiner}sslmode=require"
    return url


@contextmanager
def db_connection():
    """Open a short-lived Postgres connection.

    Requires:
        pip install "psycopg[binary]"
        DATABASE_URL=postgresql://...
    """
    url = _database_url()
    if not url:
        yield None
        return

    try:
        import psycopg
        from psycopg.rows import dict_row
    except ImportError as exc:
        logger.warning("psycopg is not installed. DB persistence is disabled: %s", exc)
        yield None
        return

    conn = None
    try:
        conn = psycopg.connect(url, row_factory=dict_row)
        yield conn
        conn.commit()
    except Exception:
        if conn is not None:
            conn.rollback()
        logger.exception("Database operation failed")
        yield None
    finally:
        if conn is not None:
            conn.close()


def _json_default(value: Any) -> str:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    if hasattr(value, "value"):
        return value.value
    return str(value)


def to_json(data: Any) -> str:
    if hasattr(data, "model_dump"):
        data = data.model_dump(mode="json")
    return json.dumps(data, ensure_ascii=False, default=_json_default)


def from_json(raw: Any) -> Any:
    if raw is None:
        return None
    if isinstance(raw, (dict, list)):
        return raw
    return json.loads(raw)


def fetch_one(query: str, params: Iterable[Any] = ()) -> dict[str, Any] | None:
    with db_connection() as conn:
        if conn is None:
            return None
        with conn.cursor() as cur:
            cur.execute(query, tuple(params))
            row = cur.fetchone()
            return dict(row) if row else None


def fetch_all(query: str, params: Iterable[Any] = ()) -> list[dict[str, Any]]:
    with db_connection() as conn:
        if conn is None:
            return []
        with conn.cursor() as cur:
            cur.execute(query, tuple(params))
            return [dict(row) for row in cur.fetchall()]


def execute(query: str, params: Iterable[Any] = ()) -> bool:
    with db_connection() as conn:
        if conn is None:
            return False
        with conn.cursor() as cur:
            cur.execute(query, tuple(params))
        return True


# ===== Users =====

def load_user(user_id: str) -> dict[str, Any] | None:
    return fetch_one(
        "select user_id, email, data from app_users where user_id = %s",
        (user_id,),
    )


def save_user(user_id: str, email: str | None, data: Any) -> None:
    execute(
        """
        insert into app_users (user_id, email, data, updated_at)
        values (%s, %s, %s::jsonb, now())
        on conflict (user_id)
        do update set
            email = excluded.email,
            data = excluded.data,
            updated_at = now()
        """,
        (user_id, email, to_json(data)),
    )


# ===== Pet profile/history =====

def load_pet_profile(owner_user_id: str, pet_id: str) -> dict[str, Any] | None:
    row = fetch_one(
        """
        select data
        from pet_profiles
        where owner_user_id = %s and pet_id = %s
        """,
        (owner_user_id, pet_id),
    )
    return from_json(row["data"]) if row else None


def save_pet_profile(owner_user_id: str, pet_id: str, data: Any) -> None:
    execute(
        """
        insert into pet_profiles (owner_user_id, pet_id, data, updated_at)
        values (%s, %s, %s::jsonb, now())
        on conflict (owner_user_id, pet_id)
        do update set data = excluded.data, updated_at = now()
        """,
        (owner_user_id, pet_id, to_json(data)),
    )


def load_pet_history(owner_user_id: str, pet_id: str) -> dict[str, Any] | None:
    row = fetch_one(
        """
        select data
        from pet_histories
        where owner_user_id = %s and pet_id = %s
        """,
        (owner_user_id, pet_id),
    )
    return from_json(row["data"]) if row else None


def save_pet_history(owner_user_id: str, pet_id: str, data: Any) -> None:
    execute(
        """
        insert into pet_histories (owner_user_id, pet_id, data, updated_at)
        values (%s, %s, %s::jsonb, now())
        on conflict (owner_user_id, pet_id)
        do update set data = excluded.data, updated_at = now()
        """,
        (owner_user_id, pet_id, to_json(data)),
    )


# ===== Chat history/messages =====

def load_chat_history(owner_user_id: str, session_key: str) -> list[dict[str, str]] | None:
    row = fetch_one(
        """
        select messages
        from chat_sessions
        where owner_user_id = %s and session_key = %s
        """,
        (owner_user_id, session_key),
    )
    return from_json(row["messages"]) if row else None


def save_chat_history(owner_user_id: str, pet_id: str, session_key: str, messages: Any) -> None:
    execute(
        """
        insert into chat_sessions (owner_user_id, pet_id, session_key, messages, updated_at)
        values (%s, %s, %s, %s::jsonb, now())
        on conflict (owner_user_id, session_key)
        do update set
            pet_id = excluded.pet_id,
            messages = excluded.messages,
            updated_at = now()
        """,
        (owner_user_id, pet_id, session_key, to_json(messages)),
    )


def append_chat_message(owner_user_id: str, pet_id: str, message: Any) -> None:
    payload = message.model_dump(mode="json") if hasattr(message, "model_dump") else message
    message_id = payload.get("message_id")
    execute(
        """
        insert into chat_messages (message_id, owner_user_id, pet_id, session_id, role, text, data, created_at)
        values (%s, %s, %s, %s, %s, %s, %s::jsonb, coalesce(%s::timestamptz, now()))
        on conflict (message_id) do nothing
        """,
        (
            message_id,
            owner_user_id,
            pet_id,
            payload.get("session_id"),
            payload.get("role"),
            payload.get("text"),
            to_json(payload),
            payload.get("created_at"),
        ),
    )


# ===== Extractions =====

def append_extraction(owner_user_id: str, pet_id: str, extraction: Any) -> None:
    payload = extraction.model_dump(mode="json") if hasattr(extraction, "model_dump") else extraction
    execute(
        """
        insert into extraction_results (owner_user_id, pet_id, message_id, data, created_at)
        values (%s, %s, %s, %s::jsonb, now())
        """,
        (owner_user_id, pet_id, payload.get("message_id"), to_json(payload)),
    )


def load_recent_extractions(owner_user_id: str, pet_id: str, limit: int = 10) -> list[dict[str, Any]]:
    rows = fetch_all(
        """
        select data
        from extraction_results
        where owner_user_id = %s and pet_id = %s
        order by created_at desc
        limit %s
        """,
        (owner_user_id, pet_id, limit),
    )
    # Return oldest -> newest for easier reading.
    return [from_json(row["data"]) for row in reversed(rows)]


# ===== AI character memory =====

def load_character_memory(owner_user_id: str) -> dict[str, Any] | None:
    row = fetch_one(
        "select data from ai_character_memories where owner_user_id = %s",
        (owner_user_id,),
    )
    return from_json(row["data"]) if row else None


def save_character_memory(owner_user_id: str, data: Any) -> None:
    execute(
        """
        insert into ai_character_memories (owner_user_id, data, updated_at)
        values (%s, %s::jsonb, now())
        on conflict (owner_user_id)
        do update set data = excluded.data, updated_at = now()
        """,
        (owner_user_id, to_json(data)),
    )


def delete_user_account_data(owner_user_id: str) -> bool:
    with db_connection() as conn:
        if conn is None:
            return False
        with conn.cursor() as cur:
            cur.execute("delete from extraction_results where owner_user_id = %s", (owner_user_id,))
            cur.execute("delete from chat_messages where owner_user_id = %s", (owner_user_id,))
            cur.execute("delete from chat_sessions where owner_user_id = %s", (owner_user_id,))
            cur.execute("delete from pet_histories where owner_user_id = %s", (owner_user_id,))
            cur.execute("delete from pet_profiles where owner_user_id = %s", (owner_user_id,))
            cur.execute("delete from ai_character_memories where owner_user_id = %s", (owner_user_id,))
            cur.execute("delete from app_users where user_id = %s", (owner_user_id,))
        return True
