"""Разовая пред-сборка кеша векторного индекса для hybrid-поиска.

Прод (бот и CLI) работает на hybrid и грузит эмбеддинги всех чанков из кеша
`vector_index/`. Этот скрипт собирает кеш заранее, чтобы первый старт бота был
мгновенным и не тратил API. Запускать при первом деплое и после изменения базы.

    OPENAI_API_KEY=... python build_vector_index.py
    python build_vector_index.py --vector-cache vector_index --base-dir knowledge_base

Требует OPENAI_API_KEY (эмбеддинги text-embedding-3-small).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")  # подхватить OPENAI_API_KEY из .env

from src.assistant_factory import (
    DEFAULT_EMBEDDING_MODEL,
    DEFAULT_VECTOR_CACHE,
    load_or_build_vector_index,
)
from src.chunking import load_knowledge_chunks


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

    parser = argparse.ArgumentParser(description="Build/refresh the VINOTERRA vector index cache.")
    parser.add_argument("--base-dir", default="knowledge_base")
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--chunk-overlap", type=int, default=100)
    parser.add_argument("--vector-cache", default=DEFAULT_VECTOR_CACHE)
    parser.add_argument("--embedding-model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument(
        "--force", action="store_true",
        help="Пересобрать даже при валидном кеше (перезаписать эмбеддинги).",
    )
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Нужен OPENAI_API_KEY для эмбеддингов.")

    chunks = load_knowledge_chunks(
        args.base_dir, chunk_size=args.chunk_size, chunk_overlap=args.chunk_overlap
    )
    print(f"Чанков базы: {len(chunks)}")

    cache = None if args.force else args.vector_cache
    if args.force:
        # Игнорируем существующий кеш: строим и сохраняем заново.
        from src.retrieval import OpenAIEmbeddingClient, VectorIndex

        client = OpenAIEmbeddingClient(model=args.embedding_model)
        print(f"--force: пересобираю эмбеддинги для {len(chunks)} чанков ...")
        index = VectorIndex.from_embedding_client(
            chunks, client, metadata={"model": args.embedding_model}
        )
        index.save(args.vector_cache)
        print(f"Векторный индекс сохранён в {args.vector_cache}")
    else:
        index, _ = load_or_build_vector_index(
            chunks, embedding_model=args.embedding_model, vector_cache=cache
        )
        print(f"Готово. Чанков в индексе: {len(index.chunks)}; кеш: {args.vector_cache}")


if __name__ == "__main__":
    main()
