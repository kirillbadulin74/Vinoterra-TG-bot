from __future__ import annotations

import json
import math
import os
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Protocol, Sequence

from src.chunking import KnowledgeChunk

try:
    import numpy as np
except ImportError:  # pragma: no cover - handled at runtime with a clear error.
    np = None  # type: ignore[assignment]


TOKEN_RE = re.compile(r"[a-zа-яё0-9]+", re.IGNORECASE)
RU_TOKEN_RE = re.compile(r"^[а-яё]+$")
RU_SUFFIXES = (
    "иями",
    "ями",
    "ами",
    "аются",
    "яются",
    "ается",
    "яется",
    "ются",
    "ется",
    "ого",
    "ему",
    "ыми",
    "ими",
    "ать",
    "ять",
    "ить",
    "ешь",
    "ает",
    "яют",
    "уют",
    "ый",
    "ий",
    "ой",
    "ая",
    "яя",
    "ое",
    "ее",
    "ые",
    "ие",
    "ых",
    "их",
    "ую",
    "юю",
    "ам",
    "ям",
    "ах",
    "ях",
    "ов",
    "ев",
    "ей",
    "ом",
    "ем",
    "ть",
    "а",
    "я",
    "ы",
    "и",
    "у",
    "ю",
    "е",
    "о",
)


@dataclass(frozen=True)
class SearchResult:
    chunk: KnowledgeChunk
    score: float
    method: str
    rank: int

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id

    @property
    def source_file(self) -> str:
        return self.chunk.source_file

    @property
    def section_path(self) -> str:
        return self.chunk.section_path


class EmbeddingClient(Protocol):
    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one embedding vector per text."""

    def embed_query(self, text: str) -> list[float]:
        """Return an embedding vector for a user query."""


class OpenAIEmbeddingClient:
    """Small wrapper around the OpenAI embeddings API used in the curator notebook."""

    def __init__(
        self,
        *,
        model: str = "text-embedding-3-small",
        batch_size: int = 64,
        timeout: float = 60.0,
        client: object | None = None,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:  # pragma: no cover - depends on environment.
                raise RuntimeError("Install openai to use OpenAIEmbeddingClient") from exc
            client_kwargs = {}
            base_url = os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENAI_API_BASE")
            if base_url:
                client_kwargs["base_url"] = base_url
            client = OpenAI(**client_kwargs)

        self.client = client
        self.model = model
        self.batch_size = batch_size
        self.timeout = timeout

    def embed_documents(self, texts: Sequence[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = list(texts[start : start + self.batch_size])
            if not batch:
                continue
            vectors.extend(self._embed(batch))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self._embed([text])[0]

    def _embed(self, texts: Sequence[str]) -> list[list[float]]:
        response = self.client.embeddings.create(
            model=self.model,
            input=list(texts),
            timeout=self.timeout,
        )
        data = sorted(response.data, key=lambda item: item.index)
        return [list(item.embedding) for item in data]


class BM25Index:
    def __init__(
        self,
        chunks: Sequence[KnowledgeChunk],
        *,
        k1: float = 1.5,
        b: float = 0.75,
        method_name: str = "bm25",
    ) -> None:
        if not chunks:
            raise ValueError("BM25Index requires at least one chunk")
        self.chunks = list(chunks)
        self.k1 = k1
        self.b = b
        self.method_name = method_name
        self.doc_tokens = [tokenize(chunk.content) for chunk in self.chunks]
        self.doc_lengths = [len(tokens) for tokens in self.doc_tokens]
        self.avg_doc_length = sum(self.doc_lengths) / len(self.doc_lengths)
        self.term_frequencies = [Counter(tokens) for tokens in self.doc_tokens]
        self.idf = self._build_idf()

    def search(self, query: str, *, top_k: int = 8) -> list[SearchResult]:
        query_terms = tokenize(query)
        if not query_terms:
            return []

        scores: list[tuple[int, float]] = []
        for doc_index, frequencies in enumerate(self.term_frequencies):
            score = self._score_document(query_terms, doc_index, frequencies)
            if score > 0:
                scores.append((doc_index, score))

        scores.sort(key=lambda item: item[1], reverse=True)
        results: list[SearchResult] = []
        for rank, (doc_index, score) in enumerate(scores[:top_k], start=1):
            results.append(
                SearchResult(
                    chunk=self.chunks[doc_index],
                    score=score,
                    method=self.method_name,
                    rank=rank,
                )
            )
        return results

    def _build_idf(self) -> dict[str, float]:
        document_count = len(self.doc_tokens)
        document_frequency: Counter[str] = Counter()
        for tokens in self.doc_tokens:
            document_frequency.update(set(tokens))

        return {
            term: math.log(1 + (document_count - frequency + 0.5) / (frequency + 0.5))
            for term, frequency in document_frequency.items()
        }

    def _score_document(
        self,
        query_terms: list[str],
        doc_index: int,
        frequencies: Counter[str],
    ) -> float:
        doc_length = self.doc_lengths[doc_index]
        score = 0.0
        for term in query_terms:
            term_frequency = frequencies.get(term, 0)
            if not term_frequency:
                continue
            idf = self.idf.get(term, 0.0)
            denominator = term_frequency + self.k1 * (
                1 - self.b + self.b * doc_length / self.avg_doc_length
            )
            score += idf * (term_frequency * (self.k1 + 1)) / denominator
        return score


class VectorIndex:
    """Cosine vector index for OpenAI embeddings.

    The submitted notebook used LangChain + FAISS. This class keeps the same
    retrieval idea while avoiding a hard FAISS dependency in the local project.
    """

    def __init__(
        self,
        chunks: Sequence[KnowledgeChunk],
        vectors: Sequence[Sequence[float]],
        *,
        method_name: str = "vector",
        metadata: Mapping[str, object] | None = None,
    ) -> None:
        if np is None:
            raise RuntimeError("Install numpy to use VectorIndex")
        if not chunks:
            raise ValueError("VectorIndex requires at least one chunk")
        if len(chunks) != len(vectors):
            raise ValueError("chunks and vectors must have the same length")

        self.chunks = list(chunks)
        self.method_name = method_name
        self.metadata = dict(metadata or {})
        self.vectors = _normalize_matrix(np.asarray(vectors, dtype=np.float32))
        if self.vectors.ndim != 2:
            raise ValueError("vectors must be a 2D matrix")

    @classmethod
    def from_embedding_client(
        cls,
        chunks: Sequence[KnowledgeChunk],
        embedding_client: EmbeddingClient,
        *,
        method_name: str = "vector",
        metadata: Mapping[str, object] | None = None,
    ) -> "VectorIndex":
        texts = [chunk.content for chunk in chunks]
        vectors = embedding_client.embed_documents(texts)
        return cls(chunks, vectors, method_name=method_name, metadata=metadata)

    @classmethod
    def exists(cls, index_dir: str | Path) -> bool:
        path = Path(index_dir)
        return (
            path.joinpath("vectors.npy").is_file()
            and path.joinpath("chunks.jsonl").is_file()
            and path.joinpath("metadata.json").is_file()
        )

    @classmethod
    def load(cls, index_dir: str | Path, *, method_name: str = "vector") -> "VectorIndex":
        if np is None:
            raise RuntimeError("Install numpy to use VectorIndex")

        path = Path(index_dir)
        vectors_path = path / "vectors.npy"
        chunks_path = path / "chunks.jsonl"
        metadata_path = path / "metadata.json"
        if not vectors_path.is_file() or not chunks_path.is_file():
            raise FileNotFoundError(f"Vector index files not found in {path}")

        metadata = {}
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

        chunks = []
        with chunks_path.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                record = json.loads(line)
                chunks.append(_chunk_from_record(record))

        vectors = np.load(vectors_path)
        return cls(chunks, vectors, method_name=method_name, metadata=metadata)

    @classmethod
    def load_metadata(cls, index_dir: str | Path) -> dict[str, object]:
        metadata_path = Path(index_dir) / "metadata.json"
        if not metadata_path.is_file():
            return {}
        return json.loads(metadata_path.read_text(encoding="utf-8"))

    def save(self, index_dir: str | Path) -> None:
        if np is None:
            raise RuntimeError("Install numpy to use VectorIndex")

        path = Path(index_dir)
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "vectors.npy", self.vectors)

        with (path / "chunks.jsonl").open("w", encoding="utf-8") as file:
            for chunk in self.chunks:
                file.write(json.dumps(_chunk_to_record(chunk), ensure_ascii=False) + "\n")

        metadata = dict(self.metadata)
        metadata["chunk_count"] = len(self.chunks)
        metadata["vector_dimensions"] = int(self.vectors.shape[1])
        (path / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def search(
        self,
        query: str,
        embedding_client: EmbeddingClient,
        *,
        top_k: int = 8,
    ) -> list[SearchResult]:
        if np is None:
            raise RuntimeError("Install numpy to use VectorIndex")
        if not query.strip():
            return []

        query_vector = _normalize_vector(np.asarray(embedding_client.embed_query(query), dtype=np.float32))
        scores = self.vectors @ query_vector
        if scores.size == 0:
            return []

        top_count = min(top_k, len(self.chunks))
        top_indices = np.argsort(scores)[::-1][:top_count]
        results: list[SearchResult] = []
        for rank, doc_index in enumerate(top_indices, start=1):
            results.append(
                SearchResult(
                    chunk=self.chunks[int(doc_index)],
                    score=float(scores[int(doc_index)]),
                    method=self.method_name,
                    rank=rank,
                )
            )
        return results


def hybrid_search(
    *,
    query: str,
    bm25_index: BM25Index,
    vector_index: VectorIndex,
    embedding_client: EmbeddingClient,
    top_k: int = 8,
    candidate_k: int = 24,
    weights: Mapping[str, float] | None = None,
    method_name: str = "hybrid_bm25_vector",
) -> list[SearchResult]:
    bm25_results = bm25_index.search(query, top_k=candidate_k)
    vector_results = vector_index.search(query, embedding_client, top_k=candidate_k)
    return reciprocal_rank_fusion(
        {
            bm25_index.method_name: bm25_results,
            vector_index.method_name: vector_results,
        },
        top_k=top_k,
        weights=weights,
        method_name=method_name,
    )


def tokenize(text: str) -> list[str]:
    normalized = text.lower().replace("ё", "е")
    return [normalize_token(token) for token in TOKEN_RE.findall(normalized)]


def normalize_token(token: str) -> str:
    if not RU_TOKEN_RE.match(token) or len(token) <= 4:
        return token
    for suffix in RU_SUFFIXES:
        if token.endswith(suffix) and len(token) - len(suffix) >= 4:
            return token[: -len(suffix)]
    return token


def reciprocal_rank_fusion(
    results_by_method: Mapping[str, Sequence[SearchResult]],
    *,
    top_k: int = 8,
    weights: Mapping[str, float] | None = None,
    rank_constant: int = 60,
    method_name: str = "hybrid_rrf",
) -> list[SearchResult]:
    if not results_by_method:
        return []

    weights = weights or {}
    fused_scores: defaultdict[str, float] = defaultdict(float)
    chunks_by_id: dict[str, KnowledgeChunk] = {}
    best_rank_by_id: dict[str, int] = {}

    for method, results in results_by_method.items():
        weight = weights.get(method, 1.0)
        for result in results:
            chunks_by_id[result.chunk_id] = result.chunk
            best_rank_by_id[result.chunk_id] = min(
                best_rank_by_id.get(result.chunk_id, result.rank),
                result.rank,
            )
            fused_scores[result.chunk_id] += weight / (rank_constant + result.rank)

    ranked = sorted(fused_scores.items(), key=lambda item: item[1], reverse=True)
    return [
        SearchResult(
            chunk=chunks_by_id[chunk_id],
            score=score,
            method=method_name,
            rank=rank,
        )
        for rank, (chunk_id, score) in enumerate(ranked[:top_k], start=1)
    ]


def format_result_line(result: SearchResult, *, max_text_chars: int = 180) -> str:
    text = re.sub(r"\s+", " ", result.chunk.text).strip()
    if len(text) > max_text_chars:
        text = text[: max_text_chars - 3].rstrip() + "..."
    return (
        f"{result.rank}. [{result.method}] score={result.score:.4f} "
        f"{result.source_file} | {result.section_path} | {text}"
    )


def _normalize_matrix(matrix: "np.ndarray") -> "np.ndarray":
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return matrix / norms


def _normalize_vector(vector: "np.ndarray") -> "np.ndarray":
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        return vector
    return vector / norm


def _chunk_to_record(chunk: KnowledgeChunk) -> dict[str, object]:
    return {
        "chunk_id": chunk.chunk_id,
        "source_file": chunk.source_file,
        "section_index": chunk.section_index,
        "chunk_index": chunk.chunk_index,
        "heading_path": list(chunk.heading_path),
        "start_line": chunk.start_line,
        "end_line": chunk.end_line,
        "text": chunk.text,
        "content": chunk.content,
    }


def _chunk_from_record(record: Mapping[str, object]) -> KnowledgeChunk:
    return KnowledgeChunk(
        chunk_id=str(record["chunk_id"]),
        source_file=str(record["source_file"]),
        section_index=int(record["section_index"]),
        chunk_index=int(record["chunk_index"]),
        heading_path=tuple(str(item) for item in record["heading_path"]),
        start_line=int(record["start_line"]),
        end_line=int(record["end_line"]),
        text=str(record["text"]),
        content=str(record["content"]),
    )
