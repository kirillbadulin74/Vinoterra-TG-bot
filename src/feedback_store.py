"""Лог взаимодействий бота + оценки пользователей (пожелание куратора после защиты).

Хранилище — один SQLite-файл. Назначение — не аналитика ради аналитики, а
продолжение методологии диплома «живая эксплуатация → регрессионные вопросы»:
каждый вопрос куратора/тестеров с ответом, источниками и веткой (main/fallback)
автоматически становится кандидатом в регрессионный набор. Оценки 👍/👎 и
отказы находятся SQL-запросом и указывают, где дополнять базу.

Пользовательских данных в смысле ПДн здесь нет: chat_id — технический
идентификатор, нужный, чтобы отличать сессии тестеров друг от друга.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

_SCHEMA = """
CREATE TABLE IF NOT EXISTS interactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts TEXT NOT NULL,                 -- UTC ISO-8601
    chat_id INTEGER NOT NULL,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    sources TEXT,                     -- top-разделы поиска, по одному в строке
    latency_ms INTEGER,
    llm_branch TEXT,                  -- main | fallback
    retrieval_mode TEXT,              -- hybrid | bm25 (реально использованный)
    is_refusal INTEGER NOT NULL DEFAULT 0,  -- отказ «нет информации» / OOD
    is_error INTEGER NOT NULL DEFAULT 0,    -- ответ не сформирован (исключение)
    feedback TEXT                     -- up | down | NULL (не оценён)
)
"""


class FeedbackStore:
    """Тонкая обёртка над SQLite. Бот однопоточный (long polling), поэтому
    одно соединение без блокировок; фоновый TypingIndicator в БД не пишет."""

    def __init__(self, db_path: str | Path) -> None:
        path = Path(db_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.execute(_SCHEMA)
        self.conn.commit()

    def log_interaction(
        self,
        *,
        chat_id: int,
        question: str,
        answer: str,
        sources: list[str] | None = None,
        latency_ms: int | None = None,
        llm_branch: str | None = None,
        retrieval_mode: str | None = None,
        is_refusal: bool = False,
        is_error: bool = False,
    ) -> int:
        cursor = self.conn.execute(
            "INSERT INTO interactions (ts, chat_id, question, answer, sources, "
            "latency_ms, llm_branch, retrieval_mode, is_refusal, is_error) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                datetime.now(timezone.utc).isoformat(timespec="seconds"),
                chat_id,
                question,
                answer,
                "\n".join(sources) if sources else None,
                latency_ms,
                llm_branch,
                retrieval_mode,
                int(is_refusal),
                int(is_error),
            ),
        )
        self.conn.commit()
        return int(cursor.lastrowid)

    def set_feedback(self, interaction_id: int, vote: str) -> bool:
        """Записывает оценку up/down. False, если записи нет (например, БД
        пересоздали, а кнопка осталась под старым сообщением)."""
        if vote not in ("up", "down"):
            return False
        cursor = self.conn.execute(
            "UPDATE interactions SET feedback = ? WHERE id = ?",
            (vote, interaction_id),
        )
        self.conn.commit()
        return cursor.rowcount > 0

    def close(self) -> None:
        self.conn.close()
