import unittest
from pathlib import Path
from types import SimpleNamespace

from src.chunking import KnowledgeChunk, load_knowledge_chunks
from src.rag import (
    WineRAGAssistant,
    build_retrieval_query,
    filter_consultation_results,
    is_broad_consultation_question,
    is_correction_question,
    is_more_examples_question,
    is_out_of_domain_question,
    is_post_soviet_scope_question,
    repair_correction_question,
    repair_more_examples_question,
    repair_temperature_ranges,
)
from src.retrieval import BM25Index, SearchResult


class FakeChatClient:
    def __init__(self, responses: list[str] | None = None):
        self.prompts = []
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


class ConsultationIntentTests(unittest.TestCase):
    def test_dessert_request_is_not_rejected_by_domain_guard(self):
        self.assertFalse(is_out_of_domain_question("что предложишь на дессерт"))
        self.assertFalse(is_out_of_domain_question("какое полусладкое выбрать"))

    def test_recipe_request_is_still_out_of_domain(self):
        self.assertTrue(is_out_of_domain_question("что приготовить на десерт"))
        self.assertTrue(is_out_of_domain_question("какие фрукты полезны"))
        self.assertFalse(is_out_of_domain_question("какое вино подать к фруктам"))

    def test_open_recommendation_gets_wine_search_terms(self):
        question = "что предложишь на дессерт"
        query = build_retrieval_query(question)

        self.assertTrue(is_broad_consultation_question(question))
        self.assertIn("десертные вина", query)
        self.assertIn("Sauternes Tokaji Muscat Port Madeira Asti", query)
        self.assertIn("температура подачи", query)

    def test_more_examples_request_gets_new_styles_and_exclusion_rule(self):
        question = "а какие еще есть примеры полусладких вин"
        history = [("предыдущий вопрос", "Sauternes, Tokaji и Icewine")]

        self.assertTrue(is_more_examples_question(question))
        repaired = repair_more_examples_question(question, history)
        self.assertIsNotNone(repaired)
        self.assertIn("новых примеров", repaired)
        self.assertIn("не повторяй", repaired)

        query = build_retrieval_query(question)
        self.assertIn("Vin Santo", query)
        self.assertIn("Recioto", query)

    def test_correction_drops_accidental_geography(self):
        question = "а при чем тут Казахстан и Кыргызстан, я спросил о сортах вина"
        repaired = repair_correction_question(question)

        self.assertIsNotNone(repaired)
        self.assertIn("сорта винограда", repaired)
        self.assertNotIn("Казахстан", repaired)
        self.assertNotIn("Кыргызстан", repaired)

    def test_scoped_how_about_question_keeps_explicit_region(self):
        question = "А как насчет полусладких вин постсоветских стран?"

        self.assertFalse(is_correction_question(question))
        self.assertIsNone(repair_correction_question(question))
        self.assertFalse(is_broad_consultation_question(question))

        query = build_retrieval_query(question)
        self.assertIn("Хванчкара", query)
        self.assertIn("Киндзмараули", query)
        self.assertNotIn("Sauternes", query)

    def test_crimean_how_about_question_keeps_explicit_region(self):
        question = "а как насчет крымских десертных вин?"

        self.assertFalse(is_correction_question(question))
        self.assertFalse(is_broad_consultation_question(question))

        query = build_retrieval_query(question)
        self.assertIn("Крым", query)
        self.assertNotIn("Sauternes", query)
        self.assertNotIn("Массандра", query)

    def test_specific_post_soviet_regions_get_canonical_names(self):
        cases = {
            "А как насчет десертных вин Ставрополья?": "Ставрополье",
            "А как насчет Долины Дона?": "Долина Дона",
            "Есть ли десертные вина в Долине Терека?": "Долина Терека",
            "А что скажешь о нижневолжских десертных винах?": "Нижняя Волга",
            "какие десертные вина есть в России?": "Россия",
        }

        for question, region in cases.items():
            query = build_retrieval_query(question)
            self.assertIn(region, query, question)
            self.assertNotIn("Sauternes", query, question)
            self.assertNotIn("Массандра", query, question)

    def test_umbrella_post_soviet_query_still_gets_wine_names(self):
        query = build_retrieval_query("А как насчет полусладких вин постсоветских стран?")

        self.assertIn("Хванчкара", query)
        self.assertIn("Массандра", query)

    def test_don_and_terek_valley_questions_are_recognized(self):
        questions = (
            "А как насчет Долины Дона?",
            "Но в Долине Дона точно делают кагор",
            "Есть ли десертные вина в Долине Терека",
        )

        for question in questions:
            self.assertFalse(is_correction_question(question), question)
            self.assertFalse(is_broad_consultation_question(question), question)

    def test_russian_region_how_about_questions_keep_explicit_region(self):
        questions = (
            "а как насчет дагестанских десертных вин?",
            "а как насчет кубанских полусладких вин?",
            "а как насчет ставропольских десертных вин?",
            "а как насчет донских ликерных вин?",
        )

        for question in questions:
            self.assertFalse(is_correction_question(question), question)
            self.assertFalse(is_broad_consultation_question(question), question)
            query = build_retrieval_query(question)
            self.assertNotIn("Sauternes", query, question)

    def test_post_soviet_scope_detection(self):
        self.assertTrue(is_post_soviet_scope_question("полусладкие вина постсоветских стран"))
        self.assertTrue(is_post_soviet_scope_question("а как насчет крымских десертных вин?"))
        self.assertTrue(is_post_soviet_scope_question("вина Средней Азии"))
        self.assertFalse(is_post_soviet_scope_question("полусладкое средней цены"))
        self.assertFalse(is_post_soviet_scope_question("какие десертные вина Франции"))

    def test_post_soviet_recommendations_prioritize_russia(self):
        base_dir = Path(__file__).parents[1] / "knowledge_base"
        chunks = load_knowledge_chunks(base_dir, chunk_size=500, chunk_overlap=100)
        assistant = WineRAGAssistant(bm25_index=BM25Index(chunks), chat_client=FakeChatClient())

        prompt = assistant.build_user_prompt(
            "Какие полусладкие вина можно предложить из постсоветских стран?",
            "Хванчкара, Киндзмараули, Массандра, кагор, объемы производства стран бывшего СССР.",
        )

        self.assertIn("ПОДСКАЗКА ПО РЕГИОНАМ", prompt)
        self.assertIn("1) Россия", prompt)
        self.assertIn("Грузия и Азербайджан", prompt)
        self.assertIn("Средняя Азия", prompt)
        self.assertIn("около 1%", prompt)
        self.assertIn("не поставляются", prompt)
        self.assertIn("не включай их в рекомендации", prompt)

    def test_specific_region_hint_forbids_neighbor_wines(self):
        base_dir = Path(__file__).parents[1] / "knowledge_base"
        chunks = load_knowledge_chunks(base_dir, chunk_size=500, chunk_overlap=100)
        assistant = WineRAGAssistant(bm25_index=BM25Index(chunks), chat_client=FakeChatClient())

        prompt = assistant.build_user_prompt(
            "Но в Долине Дона точно делают кагор",
            "Долина Дона: автохтонные ликёрные вина из Красностопа. Кубань: кагоры.",
        )

        self.assertIn("не приписывай ему вина соседних регионов", prompt)

    def test_non_post_soviet_question_has_no_region_priority_hint(self):
        base_dir = Path(__file__).parents[1] / "knowledge_base"
        chunks = load_knowledge_chunks(base_dir, chunk_size=500, chunk_overlap=100)
        assistant = WineRAGAssistant(bm25_index=BM25Index(chunks), chat_client=FakeChatClient())

        prompt = assistant.build_user_prompt(
            "что предложишь на десерт",
            "Sauternes — десертное вино из Бордо.",
        )

        self.assertNotIn("ПОДСКАЗКА ПО РЕГИОНАМ", prompt)

    def test_more_examples_repair_preserves_explicit_region(self):
        question = "а какие еще есть примеры полусладких вин постсоветских стран"
        history = [("предыдущий вопрос", "Sauternes, Tokaji и Icewine")]

        repaired = repair_more_examples_question(question, history)

        self.assertIsNotNone(repaired)
        self.assertIn("постсоветских стран", repaired)
        self.assertIn("3–5 новых примеров", repaired)
        self.assertNotIn("разных винодельческих регионов мира", repaired)

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

    def test_sweetness_correction_requests_new_options_instead_of_repeating(self):
        base_dir = Path(__file__).parents[1] / "knowledge_base"
        chunks = load_knowledge_chunks(base_dir, chunk_size=500, chunk_overlap=100)
        client = FakeChatClient()
        assistant = WineRAGAssistant(bm25_index=BM25Index(chunks), chat_client=client)

        answer = assistant.answer(
            "значит ты только эти полусладкие вина можешь предложить",
            mode="bm25",
            top_k=8,
            history=[
                ("что предложишь на десерт", "Sauternes, Porto Tawny, Porto Vintage"),
                ("а как насчет полусладких", "Sauternes, Tokaji, Icewine, поздний сбор"),
            ],
        )

        self.assertIn("3–5 новых примеров", answer.question)
        self.assertIn("не повторяй", client.prompts[-1].lower())
        self.assertIn("Vin Santo", answer.context)

    def test_real_knowledge_base_honors_post_soviet_scope(self):
        base_dir = Path(__file__).parents[1] / "knowledge_base"
        chunks = load_knowledge_chunks(base_dir, chunk_size=500, chunk_overlap=100)
        client = FakeChatClient(
            responses=[
                "Какие полусладкие вина можно предложить из постсоветских стран?",
                "Ответ-заглушка",
            ]
        )
        assistant = WineRAGAssistant(bm25_index=BM25Index(chunks), chat_client=client)

        answer = assistant.answer(
            "А как насчет полусладких вин постсоветских стран?",
            mode="bm25",
            top_k=8,
            history=[("что предложишь на десерт", "Sauternes, Tokaji и Icewine")],
        )

        self.assertIn("постсоветск", answer.question)
        sources = {item.source_file for item in answer.results}
        self.assertIn("wine_russia_and_ussr.md", sources)
        self.assertIn("Хванчкара", answer.context)
        prompt = client.prompts[-1]
        self.assertIn("постсоветск", prompt)
        self.assertNotIn("ДИАЛОГОВАЯ ПОПРАВКА", prompt)

    def test_real_knowledge_base_honors_dagestan_scope(self):
        base_dir = Path(__file__).parents[1] / "knowledge_base"
        chunks = load_knowledge_chunks(base_dir, chunk_size=500, chunk_overlap=100)
        client = FakeChatClient(
            responses=[
                "Какие десертные вина можно предложить из Дагестана?",
                "Ответ-заглушка",
            ]
        )
        assistant = WineRAGAssistant(bm25_index=BM25Index(chunks), chat_client=client)

        answer = assistant.answer(
            "а как насчет дагестанских десертных вин?",
            mode="bm25",
            top_k=8,
            history=[
                ("а как насчет крымских десертных вин?", "Массандра, портвейн, мадера")
            ],
        )

        sources = {item.source_file for item in answer.results}
        self.assertIn("wine_russia_and_ussr.md", sources)
        self.assertIn("Дагестан", answer.context)
        prompt = client.prompts[-1]
        self.assertIn("Дагестан", prompt)
        self.assertNotIn("ДИАЛОГОВАЯ ПОПРАВКА", prompt)

    def test_real_knowledge_base_honors_stavropol_follow_up(self):
        base_dir = Path(__file__).parents[1] / "knowledge_base"
        chunks = load_knowledge_chunks(base_dir, chunk_size=500, chunk_overlap=100)
        client = FakeChatClient(
            responses=[
                "А как насчет десертных вин Ставрополья?",
                "Ответ-заглушка",
            ]
        )
        assistant = WineRAGAssistant(bm25_index=BM25Index(chunks), chat_client=client)

        answer = assistant.answer(
            "А как насчет Ставрополья?",
            mode="bm25",
            top_k=8,
            history=[
                ("какие полусладкие и десертные вина есть на Кубани", "Кагор Кубанский, Мускат Кубанский")
            ],
        )

        self.assertIn("Ставрополь", answer.context)
        self.assertIn("Прасковейск", answer.context)
        prompt = client.prompts[-1]
        self.assertIn("Ставрополь", prompt)
        self.assertNotIn("ДИАЛОГОВАЯ ПОПРАВКА", prompt)

    def test_real_knowledge_base_honors_don_valley_follow_up(self):
        base_dir = Path(__file__).parents[1] / "knowledge_base"
        chunks = load_knowledge_chunks(base_dir, chunk_size=500, chunk_overlap=100)
        client = FakeChatClient(
            responses=[
                "А как насчет десертных вин Долины Дона?",
                "Ответ-заглушка",
            ]
        )
        assistant = WineRAGAssistant(bm25_index=BM25Index(chunks), chat_client=client)

        answer = assistant.answer(
            "А как еасчет Долины Дона?",
            mode="bm25",
            top_k=8,
            history=[
                ("какие полусладкие и десертные вина есть на Кубани", "Кагор Кубанский, Мускат Кубанский")
            ],
        )

        sources = {item.source_file for item in answer.results}
        self.assertIn("wine_russia_and_ussr.md", sources)
        self.assertIn("Долина Дона", answer.context)
        prompt = client.prompts[-1]
        self.assertIn("Долина Дона", prompt)
        self.assertNotIn("ДИАЛОГОВАЯ ПОПРАВКА", prompt)

    def test_temperature_ranges_are_repaired_after_llm_formatting(self):
        text = (
            "Игристое подается при температуре 68 C. "
            "Белое — 810 C, крепленое — 1012 C, портвейн — 1214 C, "
            "легкое красное — 1416 C."
        )
        repaired = repair_temperature_ranges(text)
        self.assertIn("6–8 °C", repaired)
        self.assertIn("8–10 °C", repaired)
        self.assertIn("10–12 °C", repaired)
        self.assertIn("12–14 °C", repaired)
        self.assertIn("14–16 °C", repaired)
        self.assertNotIn("68 C", repaired)
        self.assertNotIn("810 C", repaired)

    def test_variety_or_origin_question_gets_classification_hint(self):
        base_dir = Path(__file__).parents[1] / "knowledge_base"
        chunks = load_knowledge_chunks(base_dir, chunk_size=500, chunk_overlap=100)
        assistant = WineRAGAssistant(bm25_index=BM25Index(chunks), chat_client=FakeChatClient())

        prompt = assistant.build_user_prompt(
            "ты предлагаешь сорт вина или место происхождения?",
            "Sauternes — стиль вина; Франция — происхождение; Мускат — сорт винограда.",
        )
        self.assertIn("сорт — это виноград", prompt)
        self.assertIn("Не отвечай", prompt)


if __name__ == "__main__":
    unittest.main()
