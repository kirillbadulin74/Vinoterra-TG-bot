"""Smoke-тесты конвейера: чанкинг, BM25, RRF, ветки деградации и лог оценок.

Всё гоняется офлайн: ни OpenAI, ни Telegram, ни `vector_index/` не нужны.
Векторный поиск подменяется заглушкой, эмбеддинги не считаются.
"""

import json
import os
import tempfile
import unittest
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from src.chunking import (
    KnowledgeChunk,
    build_chunk_content,
    chunks_to_jsonl,
    load_knowledge_chunks,
    parse_markdown_sections,
    split_text,
)
from src.feedback_store import FeedbackStore
from src.rag import OUT_OF_DOMAIN_ANSWER, WineRAGAssistant
from src.retrieval import BM25Index, SearchResult, hybrid_search, reciprocal_rank_fusion

KNOWLEDGE_BASE = Path(__file__).parents[1] / "knowledge_base"

SAMPLE_TEXT = (
    "Первое предложение про вино. Второе предложение про сорта винограда. "
    "Третье предложение про регионы.\n\n"
    "Второй абзац про дегустацию. Ещё одно предложение про температуру подачи. "
    "И финальное предложение про хранение.\n\n"
    "Третий абзац про сочетания с едой. Последнее предложение абзаца."
)


@lru_cache(maxsize=1)
def real_chunks() -> tuple[KnowledgeChunk, ...]:
    """База знаний грузится один раз на весь модуль — она большая."""
    return tuple(load_knowledge_chunks(KNOWLEDGE_BASE, chunk_size=500, chunk_overlap=100))


def make_chunk(chunk_id: str, text: str, *, heading: str = "Раздел") -> KnowledgeChunk:
    return KnowledgeChunk(
        chunk_id=chunk_id,
        source_file=f"{chunk_id}.md",
        section_index=0,
        chunk_index=0,
        heading_path=(heading,),
        start_line=1,
        end_line=2,
        text=text,
        content=text,
    )


def make_result(chunk_id: str, rank: int, *, method: str = "bm25") -> SearchResult:
    return SearchResult(
        chunk=make_chunk(chunk_id, f"текст {chunk_id}"),
        score=1.0 / rank,
        method=method,
        rank=rank,
    )


class FakeChatClient:
    """Как в test_consultation_intents: считает промпты и отдаёт заготовки."""

    def __init__(self, responses: list[str] | None = None):
        self.prompts: list[str] = []
        self.responses = responses if responses is not None else ["Ответ-заглушка"]

    @property
    def chat(self):
        return SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.prompts.append(kwargs["messages"][-1]["content"])
        content = self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=content))]
        )


class FailingChatClient:
    def __init__(self, message: str = "LLM недоступна"):
        self.message = message
        self.calls = 0

    @property
    def chat(self):
        return SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.calls += 1
        raise RuntimeError(self.message)


class BrokenVectorIndex:
    """Гео-блок/сбой эмбеддингов: гибридный поиск обязан упасть."""

    method_name = "vector"

    def search(self, query, embedding_client, *, top_k=8):
        raise RuntimeError("embeddings unavailable")


class StubVectorIndex:
    """Отдаёт заранее заданные результаты, чтобы проверить сварку RRF."""

    method_name = "vector"

    def __init__(self, results: list[SearchResult]):
        self._results = list(results)
        self.calls: list[tuple[str, int]] = []

    def search(self, query, embedding_client, *, top_k=8):
        self.calls.append((query, top_k))
        return list(self._results)


class ChunkingTests(unittest.TestCase):
    def test_split_text_keeps_chunks_within_size(self):
        chunks = split_text(SAMPLE_TEXT, chunk_size=120)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk), 120, chunk)

    def test_split_text_preserves_all_words(self):
        chunks = split_text(SAMPLE_TEXT, chunk_size=120)
        self.assertEqual(" ".join(chunks).split(), SAMPLE_TEXT.split())

    def test_split_text_overlap_repeats_tail_of_previous_chunk(self):
        plain = split_text(SAMPLE_TEXT, chunk_size=120)
        overlapped = split_text(SAMPLE_TEXT, chunk_size=120, chunk_overlap=40)

        self.assertEqual(len(plain), len(overlapped))
        self.assertEqual(plain[0], overlapped[0])
        for index in range(1, len(plain)):
            prefix, _, body = overlapped[index].partition("\n\n")
            self.assertTrue(prefix, overlapped[index])
            # Хвост берётся из уже схлопнутых пробелов предыдущего чанка.
            self.assertIn(prefix, " ".join(plain[index - 1].split()))
            self.assertEqual(body, plain[index])

    def test_load_knowledge_chunks_rejects_invalid_parameters(self):
        for kwargs in (
            {"chunk_size": 0},
            {"chunk_size": -10},
            {"chunk_overlap": -1},
            {"chunk_size": 100, "chunk_overlap": 100},
            {"chunk_size": 100, "chunk_overlap": 500},
        ):
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError):
                    load_knowledge_chunks(KNOWLEDGE_BASE, **kwargs)

    def test_load_knowledge_chunks_missing_directory(self):
        with self.assertRaises(FileNotFoundError):
            load_knowledge_chunks(KNOWLEDGE_BASE / "нет-такой-папки")

    def test_load_knowledge_chunks_from_real_base(self):
        chunks = real_chunks()
        self.assertTrue(chunks)

        chunk_ids = [chunk.chunk_id for chunk in chunks]
        self.assertEqual(len(chunk_ids), len(set(chunk_ids)), "chunk_id должны быть уникальны")
        for chunk in chunks:
            self.assertGreaterEqual(len(chunk.text.strip()), 40)
            self.assertTrue(chunk.section_path)
            self.assertIn("Файл-источник:", chunk.content)
            self.assertIn(chunk.section_path, chunk.content)

    def test_parse_markdown_sections_builds_nested_heading_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "region.md"
            path.write_text(
                "# Европа\n\nведущий абзац раздела\n\n"
                "## Франция\n\nпро Бордо и Бургундию\n\n"
                "### Бордо (Bordeaux)\n\nпро левый берег\n",
                encoding="utf-8",
            )
            sections = parse_markdown_sections(path)

        paths = [section.heading_path for section in sections]
        self.assertIn(("Европа",), paths)
        self.assertIn(("Европа", "Франция"), paths)
        self.assertIn(("Европа", "Франция", "Бордо (Bordeaux)"), paths)

    def test_build_chunk_content_carries_source_and_section(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "spain.md"
            path.write_text("# Испания\n\nпро Риоху и Риберу\n", encoding="utf-8")
            [parsed] = parse_markdown_sections(path)

        content = build_chunk_content(parsed, "про Риоху и Риберу")
        self.assertIn("Файл-источник: spain.md", content)
        self.assertIn("Раздел: Испания", content)
        self.assertTrue(content.endswith("про Риоху и Риберу"))

    def test_chunks_to_jsonl_roundtrip(self):
        chunks = real_chunks()[:5]
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "nested" / "chunks.jsonl"
            chunks_to_jsonl(chunks, output)

            records = [
                json.loads(line)
                for line in output.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]

        self.assertEqual(len(records), len(chunks))
        for record, chunk in zip(records, chunks):
            self.assertEqual(record["chunk_id"], chunk.chunk_id)
            self.assertEqual(record["content"], chunk.content)
            self.assertEqual(record["text"], chunk.text)
            self.assertEqual(record["metadata"]["section_path"], chunk.section_path)
            # Тексты в metadata не дублируются — файл не должен пухнуть вдвое.
            self.assertNotIn("content", record["metadata"])
            self.assertNotIn("text", record["metadata"])


class BM25Tests(unittest.TestCase):
    def setUp(self):
        self.chunks = [
            make_chunk("dessert", "Десертные вина подают охлаждёнными до восьми градусов."),
            make_chunk("sparkling", "Игристое вино открывают медленно, держа бутылку под углом."),
            make_chunk("storage", "Хранить бутылки нужно в темноте и без перепадов температуры."),
        ]
        self.index = BM25Index(self.chunks)

    def test_requires_at_least_one_chunk(self):
        with self.assertRaises(ValueError):
            BM25Index([])

    def test_ranks_matching_document_first(self):
        results = self.index.search("как хранить бутылки", top_k=3)

        self.assertTrue(results)
        self.assertEqual(results[0].chunk_id, "storage")
        self.assertEqual(results[0].rank, 1)
        self.assertEqual(results[0].method, "bm25")

    def test_ranks_are_sequential_and_truncated_by_top_k(self):
        results = self.index.search("вино", top_k=2)

        self.assertEqual([item.rank for item in results], list(range(1, len(results) + 1)))
        self.assertLessEqual(len(results), 2)

    def test_query_without_matches_returns_nothing(self):
        self.assertEqual(self.index.search("квазар телескоп орбита"), [])

    def test_empty_and_punctuation_only_queries_return_nothing(self):
        self.assertEqual(self.index.search(""), [])
        self.assertEqual(self.index.search("   "), [])
        self.assertEqual(self.index.search("?!,.;"), [])

    def test_matches_inflected_forms(self):
        # Запрос «десертное вино» должен находить чанк «Десертные вина…»:
        # tokenize срезает суффиксы («десертные» и «десертное» → «десертн»).
        index = BM25Index(
            [
                make_chunk("dessert", "Десертные вина подают охлаждёнными."),
                make_chunk("storage", "Бутылки хранят в темноте."),
            ]
        )

        results = index.search("десертное вино", top_k=3)

        self.assertEqual([item.chunk_id for item in results], ["dessert"])

    def test_method_name_is_propagated(self):
        index = BM25Index(self.chunks, method_name="bm25_scoped")
        results = index.search("игристое", top_k=1)

        self.assertTrue(results)
        self.assertEqual(results[0].method, "bm25_scoped")


class ReciprocalRankFusionTests(unittest.TestCase):
    def test_document_found_by_both_methods_outranks_single_method_documents(self):
        fused = reciprocal_rank_fusion(
            {
                "bm25": [make_result("a", 1), make_result("b", 2)],
                "vector": [make_result("b", 1, method="vector"), make_result("c", 2, method="vector")],
            },
            top_k=10,
        )

        self.assertEqual([item.chunk_id for item in fused], ["b", "a", "c"])
        self.assertEqual([item.rank for item in fused], [1, 2, 3])
        self.assertEqual({item.method for item in fused}, {"hybrid_rrf"})

    def test_weights_decide_which_method_wins(self):
        def fuse(weights):
            return reciprocal_rank_fusion(
                {
                    "bm25": [make_result("from_bm25", 1)],
                    "vector": [make_result("from_vector", 1, method="vector")],
                },
                top_k=10,
                weights=weights,
            )

        self.assertEqual(fuse({"bm25": 1.0, "vector": 3.0})[0].chunk_id, "from_vector")
        self.assertEqual(fuse({"bm25": 3.0, "vector": 1.0})[0].chunk_id, "from_bm25")

    def test_duplicate_chunk_appears_once(self):
        fused = reciprocal_rank_fusion(
            {
                "bm25": [make_result("same", 1)],
                "vector": [make_result("same", 1, method="vector")],
            },
            top_k=10,
        )

        self.assertEqual([item.chunk_id for item in fused], ["same"])
        # Скор суммируется по обеим веткам: 1/61 + 1/61.
        self.assertAlmostEqual(fused[0].score, 2 / 61, places=6)

    def test_top_k_truncates_fused_list(self):
        fused = reciprocal_rank_fusion(
            {"bm25": [make_result(f"doc{i}", i) for i in range(1, 6)]},
            top_k=2,
        )

        self.assertEqual(len(fused), 2)
        self.assertEqual([item.rank for item in fused], [1, 2])

    def test_empty_input_returns_empty(self):
        self.assertEqual(reciprocal_rank_fusion({}), [])

    def test_hybrid_search_wires_bm25_and_vector_together(self):
        vector_hit = real_chunks()[0]
        stub = StubVectorIndex(
            [SearchResult(chunk=vector_hit, score=0.9, method="vector", rank=1)]
        )

        results = hybrid_search(
            query="что предложишь на десерт",
            bm25_index=BM25Index(real_chunks()),
            vector_index=stub,
            embedding_client=object(),
            top_k=3,
            candidate_k=10,
        )

        # candidate_k уходит в обе ветки — от него зависит ширина сварки.
        self.assertEqual(stub.calls, [("что предложишь на десерт", 10)])
        self.assertTrue(results)
        self.assertLessEqual(len(results), 3)
        self.assertEqual({item.method for item in results}, {"hybrid_bm25_vector"})
        self.assertIn(vector_hit.chunk_id, {item.chunk_id for item in results})


class DegradationTests(unittest.TestCase):
    """Ветки деградации: поиск hybrid→bm25, генерация main→fallback, отказы."""

    def test_retrieval_falls_back_to_bm25_when_embeddings_fail(self):
        client = FakeChatClient()
        assistant = WineRAGAssistant(
            bm25_index=BM25Index(real_chunks()),
            vector_index=BrokenVectorIndex(),
            embedding_client=object(),
            chat_client=client,
        )

        answer = assistant.answer("что предложишь на десерт", mode="hybrid", top_k=5)

        self.assertEqual(answer.mode, "hybrid", "запрошенный режим не подменяется")
        self.assertEqual(answer.retrieval_mode_used, "bm25", "реально отработал bm25")
        self.assertTrue(answer.results)
        self.assertTrue(client.prompts, "ответ всё равно дошёл до LLM")

    def test_answer_reports_fallback_llm_branch(self):
        fallback = FakeChatClient(["Ответ из фоллбэк-модели"])
        assistant = WineRAGAssistant(
            bm25_index=BM25Index(real_chunks()),
            chat_client=FailingChatClient("основная модель недоступна"),
        )
        assistant._fallback_client = fallback

        answer = assistant.answer("что предложишь на десерт", mode="bm25", top_k=5)

        self.assertEqual(answer.llm_branch, "fallback")
        self.assertEqual(answer.answer, "Ответ из фоллбэк-модели")
        self.assertTrue(fallback.prompts)

    def test_original_error_propagates_when_fallback_disabled(self):
        assistant = WineRAGAssistant(
            bm25_index=BM25Index(real_chunks()),
            chat_client=FailingChatClient("основная модель недоступна"),
        )

        # Без FALLBACK_API_KEY фоллбэк выключен — поведение прежнее.
        with patch.dict(os.environ, {"FALLBACK_API_KEY": ""}):
            with self.assertRaises(RuntimeError) as ctx:
                assistant.answer("что предложишь на десерт", mode="bm25", top_k=5)

        self.assertIn("основная модель недоступна", str(ctx.exception))

    def test_original_error_propagates_when_fallback_also_fails(self):
        assistant = WineRAGAssistant(
            bm25_index=BM25Index(real_chunks()),
            chat_client=FailingChatClient("основная модель недоступна"),
        )
        assistant._fallback_client = FailingChatClient("фоллбэк тоже упал")

        with self.assertRaises(RuntimeError) as ctx:
            assistant.answer("что предложишь на десерт", mode="bm25", top_k=5)

        # Пробрасывается ИСХОДНАЯ ошибка: она информативнее для диагностики.
        self.assertIn("основная модель недоступна", str(ctx.exception))
        self.assertNotIn("фоллбэк тоже упал", str(ctx.exception))

    def test_refusal_when_nothing_retrieved_skips_llm(self):
        client = FakeChatClient()
        assistant = WineRAGAssistant(
            bm25_index=BM25Index([make_chunk("zuid", "Зюйдвестка")]),
            chat_client=client,
        )

        # «вино» пропускает OOD-гард, но в индексе нет ни одного общего токена.
        answer = assistant.answer("вино про къяркъяр", mode="bm25", top_k=5)

        self.assertEqual(answer.results, [])
        self.assertEqual(
            answer.answer,
            "По этому вопросу нет релевантной информации в предоставленных материалах.",
        )
        self.assertEqual(client.prompts, [], "на отказ LLM не тратится")

    def test_out_of_domain_first_turn_is_refused_without_retrieval(self):
        client = FakeChatClient()
        assistant = WineRAGAssistant(
            bm25_index=BM25Index(real_chunks()),
            chat_client=client,
        )

        answer = assistant.answer("что приготовить на десерт", mode="bm25", top_k=5)

        self.assertEqual(answer.answer, OUT_OF_DOMAIN_ANSWER)
        self.assertEqual(answer.results, [])
        self.assertEqual(client.prompts, [])

    def test_follow_up_turn_is_not_refused_by_domain_guard(self):
        # Внутри диалога эллиптический follow-up не несёт доменных маркеров —
        # отказывать нельзя, пользователь уже в винном контексте.
        client = FakeChatClient(["что предложишь на десерт"])
        assistant = WineRAGAssistant(
            bm25_index=BM25Index(real_chunks()),
            chat_client=client,
        )

        answer = assistant.answer(
            "а что к нему?",
            mode="bm25",
            top_k=5,
            history=[("что предложишь на десерт", "Sauternes, Tokaji и Icewine")],
        )

        self.assertNotEqual(answer.answer, OUT_OF_DOMAIN_ANSWER)
        self.assertTrue(answer.results)


class FeedbackStoreTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        # Вложенная папка: проверяем, что стор создаёт её сам.
        self.db_path = Path(self._tmp.name) / "data" / "feedback.sqlite"
        self.store = FeedbackStore(self.db_path)
        self.addCleanup(self.store.close)

    def row(self, interaction_id: int):
        cursor = self.store.conn.execute(
            "SELECT chat_id, question, answer, sources, latency_ms, llm_branch, "
            "retrieval_mode, is_refusal, is_error, feedback "
            "FROM interactions WHERE id = ?",
            (interaction_id,),
        )
        return cursor.fetchone()

    def log(self, **overrides):
        payload = {
            "chat_id": 42,
            "question": "что предложишь на десерт",
            "answer": "Sauternes",
        }
        payload.update(overrides)
        return self.store.log_interaction(**payload)

    def test_creates_database_in_missing_directory(self):
        self.assertTrue(self.db_path.is_file())

    def test_log_interaction_returns_increasing_ids(self):
        first = self.log()
        second = self.log()

        self.assertIsInstance(first, int)
        self.assertEqual(second, first + 1)

    def test_sources_are_stored_one_per_line(self):
        interaction_id = self.log(sources=["wine_general.md", "wine_persons.md"])

        self.assertEqual(self.row(interaction_id)[3], "wine_general.md\nwine_persons.md")

    def test_empty_sources_become_null(self):
        self.assertIsNone(self.row(self.log(sources=[]))[3])

    def test_branch_and_mode_are_persisted(self):
        interaction_id = self.log(
            latency_ms=1234,
            llm_branch="fallback",
            retrieval_mode="bm25",
            is_refusal=False,
            is_error=False,
        )
        row = self.row(interaction_id)

        self.assertEqual(row[4], 1234)
        self.assertEqual(row[5], "fallback")
        self.assertEqual(row[6], "bm25")
        self.assertEqual(row[7], 0)
        self.assertEqual(row[8], 0)

    def test_refusal_and_error_flags_are_stored_as_ints(self):
        refused = self.log(is_refusal=True)
        errored = self.log(is_error=True)

        self.assertEqual(self.row(refused)[7], 1)
        self.assertEqual(self.row(refused)[8], 0)
        self.assertEqual(self.row(errored)[7], 0)
        self.assertEqual(self.row(errored)[8], 1)

    def test_feedback_is_null_until_voted(self):
        self.assertIsNone(self.row(self.log())[9])

    def test_set_feedback_accepts_up_and_down(self):
        interaction_id = self.log()

        self.assertTrue(self.store.set_feedback(interaction_id, "up"))
        self.assertEqual(self.row(interaction_id)[9], "up")
        self.assertTrue(self.store.set_feedback(interaction_id, "down"))
        self.assertEqual(self.row(interaction_id)[9], "down")

    def test_set_feedback_returns_false_for_unknown_id(self):
        self.assertFalse(self.store.set_feedback(9999, "up"))

    def test_set_feedback_rejects_invalid_vote(self):
        interaction_id = self.log()

        self.assertFalse(self.store.set_feedback(interaction_id, "maybe"))
        self.assertIsNone(self.row(interaction_id)[9])

    def test_rows_survive_reopening(self):
        interaction_id = self.log()
        self.store.close()

        reopened = FeedbackStore(self.db_path)
        self.addCleanup(reopened.close)

        self.assertEqual(reopened.conn.execute("SELECT COUNT(*) FROM interactions").fetchone()[0], 1)
        self.assertTrue(reopened.set_feedback(interaction_id, "up"))
        self.assertEqual(
            reopened.conn.execute(
                "SELECT feedback FROM interactions WHERE id = ?", (interaction_id,)
            ).fetchone()[0],
            "up",
        )


if __name__ == "__main__":
    unittest.main()
