import unittest
from pathlib import Path
from types import SimpleNamespace

from src.chunking import KnowledgeChunk, load_knowledge_chunks
from src.rag import (
    WineRAGAssistant,
    build_retrieval_query,
    filter_consultation_results,
    is_broad_consultation_question,
    is_out_of_domain_question,
    repair_correction_question,
)
from src.retrieval import BM25Index, SearchResult


class FakeChatClient:
    def __init__(self):
        self.prompts = []

    @property
    def chat(self):
        return SimpleNamespace(completions=SimpleNamespace(create=self.create))

    def create(self, **kwargs):
        self.prompts.append(kwargs["messages"][-1]["content"])
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="Ответ-заглушка"))]
        )


class ConsultationIntentTests(unittest.TestCase):
    def test_dessert_request_is_not_rejected_by_domain_guard(self):
        self.assertFalse(is_out_of_domain_question("что предложишь на дессерт"))
        self.assertFalse(is_out_of_domain_question("какое полусладкое выбрать"))

    def test_recipe_request_is_still_out_of_domain(self):
        self.assertTrue(is_out_of_domain_question("что приготовить на десерт"))

    def test_open_recommendation_gets_wine_search_terms(self):
        question = "что предложишь на дессерт"
        query = build_retrieval_query(question)

        self.assertTrue(is_broad_consultation_question(question))
        self.assertIn("десертные вина", query)
        self.assertIn("Sauternes Tokaji Muscat Port Madeira Asti", query)
        self.assertIn("температура подачи", query)

    def test_correction_drops_accidental_geography(self):
        question = "а при чем тут Казахстан и Кыргызстан, я спросил о сортах вина"
        repaired = repair_correction_question(question)

        self.assertIsNotNone(repaired)
        self.assertIn("сорта винограда", repaired)
        self.assertNotIn("Казахстан", repaired)
        self.assertNotIn("Кыргызстан", repaired)

    def test_broad_consultation_ignores_narrow_regional_sources(self):
        def result(source_file: str, chunk_id: str) -> SearchResult:
            chunk = KnowledgeChunk(
                chunk_id=chunk_id,
                source_file=source_file,
                section_index=0,
                chunk_index=0,
                heading_path=("Раздел",),
                start_line=1,
                end_line=2,
                text="текст",
                content="текст",
            )
            return SearchResult(chunk=chunk, score=1.0, method="bm25", rank=1)

        results = filter_consultation_results(
            "что предложишь на десерт",
            [
                result("wine_russia_and_ussr.md", "regional"),
                result("wine_persons.md", "persons"),
                result("wine_general.md", "general"),
            ],
        )

        self.assertEqual(["wine_general.md"], [item.source_file for item in results])

    def test_real_knowledge_base_retrieves_dessert_material(self):
        base_dir = Path(__file__).parents[1] / "knowledge_base"
        chunks = load_knowledge_chunks(base_dir, chunk_size=500, chunk_overlap=100)
        client = FakeChatClient()
        assistant = WineRAGAssistant(
            bm25_index=BM25Index(chunks),
            chat_client=client,
        )

        answer = assistant.answer("что предложишь на дессерт", mode="bm25", top_k=8)

        self.assertTrue(answer.results)
        self.assertNotIn("wine_persons.md", {item.source_file for item in answer.results})
        self.assertIn("Sauternes", client.prompts[-1])
        self.assertIn("температур", client.prompts[-1].lower())

    def test_real_knowledge_base_repairs_variety_follow_up(self):
        base_dir = Path(__file__).parents[1] / "knowledge_base"
        chunks = load_knowledge_chunks(base_dir, chunk_size=500, chunk_overlap=100)
        client = FakeChatClient()
        assistant = WineRAGAssistant(bm25_index=BM25Index(chunks), chat_client=client)

        answer = assistant.answer(
            "а при чем тут Казахстан и Кыргызстан, я спросил о сортах вина",
            mode="bm25",
            top_k=8,
            history=[("предыдущий вопрос", "ошибочный ответ про Казахстан")],
        )

        self.assertIn("сорта винограда", answer.question)
        self.assertNotIn("Казахстан", answer.question)
        self.assertIn("Cabernet Sauvignon", client.prompts[-1])


if __name__ == "__main__":
    unittest.main()
