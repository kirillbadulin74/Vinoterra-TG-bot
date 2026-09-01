from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")  # подхватить OPENAI_API_KEY и TELEGRAM_BOT_TOKEN из .env

from src.telegram_bot import BotConfig, run_bot


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

    parser = argparse.ArgumentParser(description="Run the VINOTERRA Telegram bot.")
    parser.add_argument("--base-dir", default="knowledge_base")
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--chat-model", default="gpt-4o-mini")
    parser.add_argument("--poll-timeout", type=int, default=30)
    parser.add_argument(
        "--mode", choices=["bm25", "vector", "hybrid"], default="hybrid",
        help="Retrieval-режим бота. Прод — hybrid.",
    )
    parser.add_argument("--vector-cache", default="vector_index")
    parser.add_argument("--feedback-db", default="feedback/feedback.db")
    parser.add_argument(
        "--feedback-every", type=int, default=3,
        help="Кнопки 👍/👎 под каждым N-м содержательным ответом чата (0 — выключить).",
    )
    args = parser.parse_args()

    config = BotConfig(
        token=os.environ.get("TELEGRAM_BOT_TOKEN", ""),
        base_dir=args.base_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        top_k=args.top_k,
        chat_model=args.chat_model,
        poll_timeout=args.poll_timeout,
        mode=args.mode,
        vector_cache=args.vector_cache,
        feedback_db=args.feedback_db,
        feedback_every=args.feedback_every,
    )
    run_bot(config)


if __name__ == "__main__":
    main()
