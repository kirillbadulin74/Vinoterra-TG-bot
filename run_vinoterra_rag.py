from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")  # подхватить OPENAI_API_KEY из .env

from src.assistant_factory import DEFAULT_VECTOR_CACHE, build_assistant
from src.retrieval import format_result_line


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

    parser = argparse.ArgumentParser(description="Run the VINOTERRA RAG assistant.")
    parser.add_argument("question", nargs="?", help="Question about wine or winemaking.")
    parser.add_argument("--base-dir", default="knowledge_base")
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--chat-model", default="gpt-4o-mini")
    parser.add_argument(
        "--mode", choices=["bm25", "vector", "hybrid"], default="hybrid",
        help="Retrieval-режим. Прод — hybrid. bm25 работает офлайн без ключа.",
    )
    parser.add_argument("--vector-cache", default=DEFAULT_VECTOR_CACHE)
    parser.add_argument("--show-context", action="store_true")
    parser.add_argument(
        "--no-answer",
        action="store_true",
        help="Only show retrieval sources; без генерации ответа.",
    )
    args = parser.parse_args()

    question = args.question or input("Введите вопрос о вине: ").strip()
    if not question:
        raise SystemExit("Question is empty.")

    # hybrid/vector требуют ключ уже на этапе поиска (эмбеддинги базы и запроса).
    if args.mode in ("hybrid", "vector") and not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit(
            f"Режим {args.mode} требует OPENAI_API_KEY (эмбеддинги для поиска).\n"
            "Для офлайн-превью источников без ключа: --mode bm25 --no-answer"
        )

    assistant, chunk_count = build_assistant(
        base_dir=args.base_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        chat_model=args.chat_model,
        mode=args.mode,
        vector_cache=args.vector_cache,
    )

    # Прод-конфиг: hybrid + section expansion, БЕЗ scope-рулбука.
    # include_unexpanded_results=True — expansion добавляет соседей, но не выбрасывает
    # retrieved-чанки ниже порога (см. фикс бага 17b в src/rag.py answer()).
    results = assistant.retrieve(question, mode=args.mode, top_k=args.top_k)
    context_results = assistant.expand_results_by_section(
        results, include_unexpanded_results=True
    )

    print(f"Вопрос: {question}")
    print(
        f"Конфигурация: {args.mode} + section expansion (без scope); "
        f"chunk_size={args.chunk_size}; overlap={args.chunk_overlap}; top_k={args.top_k}"
    )
    print(f"Чанков базы: {chunk_count}")
    print("\nИсточники:")
    for result in context_results:
        print(format_result_line(result))

    if args.show_context:
        print("\nКонтекст:")
        print(assistant.build_context(context_results))

    if args.no_answer:
        return

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit(
            "OPENAI_API_KEY is required to generate an answer. "
            "Use --no-answer to test retrieval without an API key."
        )

    answer = assistant.answer(question, mode=args.mode, top_k=args.top_k)
    print("\nОтвет:")
    print(answer.answer)


if __name__ == "__main__":
    main()
