"""Фабрика ассистента VINOTERRA: общая сборка WineRAGAssistant для точек входа.

Прод переведён на hybrid-поиск (BM25 + vector RRF). В отличие от чистого BM25,
hybrid требует матрицу эмбеддингов всех чанков (VectorIndex). Чтобы не гонять
эмбеддинги через OpenAI при каждом старте, индекс кешируется на диск:

- если кеш есть и число чанков совпадает — грузим (VectorIndex.load), API не тратим;
- иначе строим один раз и сохраняем (VectorIndex.save).

Логика кеширования едина для прод-пути и экспериментов, чтобы они вели себя
одинаково. Кеш можно собрать заранее скриптом build_vector_index.py.

Для mode="bm25" вектор не нужен — ключ OpenAI и сеть на этапе поиска не требуются
(как в исходном проде). Для hybrid/vector нужен OPENAI_API_KEY.
"""

from __future__ import annotations

from pathlib import Path

from src.chunking import load_knowledge_chunks
from src.rag import RetrievalMode, WineRAGAssistant
from src.retrieval import BM25Index, OpenAIEmbeddingClient, VectorIndex

DEFAULT_VECTOR_CACHE = "vector_index"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"

# Режимы, которым нужен векторный индекс (и, значит, ключ OpenAI на этапе поиска).
NEEDS_VECTOR = ("vector", "hybrid")


def load_or_build_vector_index(
    chunks,
    *,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    vector_cache: str | Path | None = DEFAULT_VECTOR_CACHE,
    embedding_client: OpenAIEmbeddingClient | None = None,
) -> tuple[VectorIndex, OpenAIEmbeddingClient]:
    """Грузит векторный индекс из кеша или строит и сохраняет его.

    Возвращает (index, embedding_client) — тот же embedding_client затем нужен
    ассистенту для эмбеддинга пользовательских запросов на рантайме.
    """
    client = embedding_client or OpenAIEmbeddingClient(model=embedding_model)
    cache = Path(vector_cache) if vector_cache else None

    if cache and VectorIndex.exists(cache):
        index = VectorIndex.load(cache)
        if len(index.chunks) == len(chunks):
            return index, client
        print(
            "! Кеш векторного индекса не совпадает по числу чанков "
            f"({len(index.chunks)} != {len(chunks)}), пересчитываю эмбеддинги."
        )

    print(f"Строю векторный индекс: эмбеддинги для {len(chunks)} чанков ...")
    index = VectorIndex.from_embedding_client(chunks, client, metadata={"model": embedding_model})
    if cache:
        index.save(cache)
        print(f"Векторный индекс сохранён в {cache}")
    return index, client


def build_assistant(
    *,
    base_dir: str | Path = "knowledge_base",
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    chat_model: str = "gpt-4o-mini",
    mode: RetrievalMode = "hybrid",
    vector_cache: str | Path | None = DEFAULT_VECTOR_CACHE,
    embedding_model: str = DEFAULT_EMBEDDING_MODEL,
    temperature: float = 0.0,
) -> tuple[WineRAGAssistant, int]:
    """Собирает WineRAGAssistant под нужный режим поиска.

    Для hybrid/vector подтягивает (или строит) векторный индекс и embedding-клиент.
    Для bm25 вектор не создаётся — старт быстрый и офлайновый.
    """
    chunks = load_knowledge_chunks(base_dir, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    bm25_index = BM25Index(chunks)

    vector_index = None
    embedding_client = None
    if mode in NEEDS_VECTOR:
        vector_index, embedding_client = load_or_build_vector_index(
            chunks, embedding_model=embedding_model, vector_cache=vector_cache
        )

    assistant = WineRAGAssistant(
        bm25_index=bm25_index,
        vector_index=vector_index,
        embedding_client=embedding_client,
        chat_model=chat_model,
        temperature=temperature,
    )
    return assistant, len(chunks)
