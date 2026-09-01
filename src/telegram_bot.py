from __future__ import annotations

import json
import os
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from src.assistant_factory import DEFAULT_VECTOR_CACHE, build_assistant as build_rag_assistant
from src.feedback_store import FeedbackStore
from src.rag import DIALOG_HISTORY_TURNS, RetrievalMode, WineRAGAssistant


TELEGRAM_API = "https://api.telegram.org/bot{token}/{method}"
MAX_TELEGRAM_MESSAGE = 3900

# Кнопки 👍/👎 показываются под каждым N-м содержательным ответом чата (не под
# каждым — чтобы не раздражать; отказы и ошибки не считаются). Решение
# пользователя от 2026-07-30: N=3, позже можно поменять через BotConfig.
FEEDBACK_EVERY_DEFAULT = 3


@dataclass(frozen=True)
class BotConfig:
    token: str
    base_dir: str | Path = "knowledge_base"
    chunk_size: int = 500
    chunk_overlap: int = 100
    top_k: int = 8
    chat_model: str = "gpt-4o-mini"
    poll_timeout: int = 30
    mode: RetrievalMode = "hybrid"
    vector_cache: str | Path = DEFAULT_VECTOR_CACHE
    feedback_db: str | Path = "feedback/feedback.db"
    feedback_every: int = FEEDBACK_EVERY_DEFAULT


class TelegramClient:
    def __init__(self, token: str) -> None:
        self.token = token

    def call(self, method: str, payload: dict[str, object]) -> dict[str, object]:
        url = TELEGRAM_API.format(token=self.token, method=method)
        data = urllib.parse.urlencode(payload).encode("utf-8")
        request = urllib.request.Request(url, data=data, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                body = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Telegram HTTP error {exc.code}: {body}") from exc
        result = json.loads(body)
        if not result.get("ok"):
            raise RuntimeError(f"Telegram API error: {result}")
        return result

    def get_me(self) -> dict[str, object]:
        return self.call("getMe", {})

    def get_updates(self, *, offset: int | None, timeout: int) -> list[dict[str, object]]:
        payload: dict[str, object] = {
            "timeout": timeout,
            # callback_query — нажатия кнопок 👍/👎 под ответами.
            "allowed_updates": json.dumps(["message", "callback_query"]),
        }
        if offset is not None:
            payload["offset"] = offset
        result = self.call("getUpdates", payload)
        updates = result.get("result", [])
        if not isinstance(updates, list):
            return []
        return updates

    def send_message(self, chat_id: int, text: str) -> int | None:
        """Отправляет текст (с нарезкой). Возвращает id последнего сообщения —
        нужен, чтобы повесить кнопки 👍/👎 на конец длинного ответа."""
        last_id: int | None = None
        for part in split_message(text):
            payload = {
                "chat_id": chat_id,
                "text": part,
                "disable_web_page_preview": True,
            }
            try:
                result = self.call("sendMessage", {**payload, "parse_mode": "HTML"})
            except RuntimeError:
                # Разметка не прошла (например, непарный тег после нарезки) —
                # отправляем голым текстом, ответ важнее оформления.
                result = self.call("sendMessage", payload)
            message = result.get("result")
            if isinstance(message, dict) and isinstance(message.get("message_id"), int):
                last_id = message["message_id"]
        return last_id

    def send_placeholder(self, chat_id: int, text: str) -> int | None:
        """Отправляет сообщение-заглушку («Ищу ответ…») и возвращает его id.

        None при сбое: заглушка — вспомогательная механика, её ошибка не должна
        ронять обработку вопроса (ответ тогда уйдёт обычным send_message).
        """
        try:
            result = self.call(
                "sendMessage",
                {"chat_id": chat_id, "text": text, "disable_web_page_preview": True},
            )
        except RuntimeError as exc:
            print(f"Placeholder send failed: {exc}", flush=True)
            return None
        message = result.get("result")
        if isinstance(message, dict) and isinstance(message.get("message_id"), int):
            return message["message_id"]
        return None

    def edit_message(self, chat_id: int, message_id: int, text: str) -> bool:
        """Редактирует сообщение (заглушку — в готовый ответ). False при сбое."""
        payload = {
            "chat_id": chat_id,
            "message_id": message_id,
            "text": text,
            "disable_web_page_preview": True,
        }
        try:
            self.call("editMessageText", {**payload, "parse_mode": "HTML"})
            return True
        except RuntimeError:
            try:
                # Как в send_message: разметка не прошла — шлём голым текстом.
                self.call("editMessageText", payload)
                return True
            except RuntimeError as exc:
                print(f"Message edit failed: {exc}", flush=True)
                return False

    def send_chat_action(self, chat_id: int) -> None:
        """Статус «печатает…» в шапке чата (Telegram гасит его через ~5 сек)."""
        try:
            self.call("sendChatAction", {"chat_id": chat_id, "action": "typing"})
        except RuntimeError as exc:
            print(f"sendChatAction failed: {exc}", flush=True)

    def attach_feedback_buttons(self, chat_id: int, message_id: int, interaction_id: int) -> None:
        """Вешает кнопки 👍/👎 на уже отправленное сообщение.

        Без текста-призыва — только две кнопки (решение против раздражения
        пользователей). callback_data несёт id записи лога. Сбой не критичен:
        кнопки — необязательная механика.
        """
        keyboard = json.dumps(
            {
                "inline_keyboard": [[
                    {"text": "\U0001F44D", "callback_data": f"fb:up:{interaction_id}"},
                    {"text": "\U0001F44E", "callback_data": f"fb:down:{interaction_id}"},
                ]]
            }
        )
        try:
            self.call(
                "editMessageReplyMarkup",
                {"chat_id": chat_id, "message_id": message_id, "reply_markup": keyboard},
            )
        except RuntimeError as exc:
            print(f"Feedback buttons attach failed: {exc}", flush=True)

    def remove_feedback_buttons(self, chat_id: int, message_id: int) -> None:
        """Убирает кнопки после нажатия — оценка одноразовая, шум исчезает."""
        try:
            self.call(
                "editMessageReplyMarkup",
                {"chat_id": chat_id, "message_id": message_id, "reply_markup": json.dumps({})},
            )
        except RuntimeError as exc:
            print(f"Feedback buttons removal failed: {exc}", flush=True)

    def answer_callback(self, callback_id: str, text: str = "") -> None:
        """Подтверждение нажатия кнопки (короткий toast, останавливает «часики»)."""
        payload: dict[str, object] = {"callback_query_id": callback_id}
        if text:
            payload["text"] = text
        try:
            self.call("answerCallbackQuery", payload)
        except RuntimeError as exc:
            print(f"answerCallbackQuery failed: {exc}", flush=True)


class TypingIndicator:
    """Держит статус «печатает…», пока формируется ответ (30-60 сек).

    Telegram сбрасывает статус через ~5 секунд, поэтому фоновый поток шлёт
    sendChatAction каждые 4 секунды до вызова stop(). Поток daemon: если
    основной цикл упал, индикатор не удержит процесс живым.
    """

    def __init__(self, client: TelegramClient, chat_id: int, *, interval: float = 4.0) -> None:
        self._client = client
        self._chat_id = chat_id
        self._interval = interval
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop_event.is_set():
            self._client.send_chat_action(self._chat_id)
            self._stop_event.wait(self._interval)

    def __enter__(self) -> "TypingIndicator":
        self._thread.start()
        return self

    def __exit__(self, *exc_info: object) -> None:
        self._stop_event.set()
        self._thread.join(timeout=1.0)


def split_message(text: str) -> list[str]:
    if len(text) <= MAX_TELEGRAM_MESSAGE:
        return [text]

    parts: list[str] = []
    rest = text
    while len(rest) > MAX_TELEGRAM_MESSAGE:
        cut = rest.rfind("\n", 0, MAX_TELEGRAM_MESSAGE)
        if cut < 1000:
            cut = MAX_TELEGRAM_MESSAGE
        parts.append(rest[:cut].strip())
        rest = rest[cut:].strip()
    if rest:
        parts.append(rest)
    return parts


def build_assistant(config: BotConfig) -> tuple[WineRAGAssistant, int]:
    # Прод-конфиг: hybrid (BM25+vector RRF) + section expansion, без scope-рулбука.
    # Векторный индекс грузится из кеша config.vector_cache (или строится один раз).
    return build_rag_assistant(
        base_dir=config.base_dir,
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        chat_model=config.chat_model,
        mode=config.mode,
        vector_cache=config.vector_cache,
    )


HEADING_RULE = "―――――――――――――――――"


def markdown_to_telegram_html(text: str) -> str:
    """Перевод markdown-разметки ответа модели в Telegram HTML.

    Telegram не понимает markdown (** и ###) — без конвертации служебные
    символы видны пользователю как есть. HTML-режим выбран вместо MarkdownV2:
    в MarkdownV2 пришлось бы экранировать почти всю пунктуацию.
    """
    lines: list[str] = []
    for raw_line in text.splitlines():
        # Сначала экранируем HTML-спецсимволы исходного текста.
        line = raw_line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

        # Заголовки '#'..'####' -> разделительная линия + жирная строка.
        heading = re.match(r"^\s{0,3}#{1,4}\s+(.*)$", line)
        if heading is not None:
            title = heading.group(1).strip().rstrip("#").strip()
            title = re.sub(r"\*\*(.+?)\*\*", r"\1", title)  # жирный внутри жирного не нужен
            if lines and lines[-1].strip():
                lines.append("")
            if lines:  # в самом начале ответа линия не нужна
                lines.append(HEADING_RULE)
            lines.append(f"<b>{title}</b>")
            continue

        # **жирный** и маркеры списков.
        line = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", line)
        line = re.sub(r"^(\s*)[-*]\s+", r"\1• ", line)
        lines.append(line)
    return "\n".join(lines)


def format_answer(answer_text: str, sources: list[str]) -> str:
    # Финальная версия бота: источники пользователю не показываем (остаются
    # в консольном логе answer_question для отладки/демонстрации заземления).
    return markdown_to_telegram_html(answer_text)


@dataclass(frozen=True)
class BotAnswer:
    """Ответ + метаданные для лога статистики (см. FeedbackStore)."""

    response: str                 # готовый Telegram HTML
    raw_answer: str               # исходный текст LLM (в лог)
    sources: list[str]
    llm_branch: str
    retrieval_mode: str | None
    is_refusal: bool


def answer_question(
    assistant: WineRAGAssistant,
    question: str,
    *,
    top_k: int,
    mode: RetrievalMode = "hybrid",
    history: list[tuple[str, str]] | None = None,
) -> BotAnswer:
    answer = assistant.answer(question, mode=mode, top_k=top_k, history=history)
    if answer.llm_branch != "main" or (
        answer.retrieval_mode_used is not None and answer.retrieval_mode_used != mode
    ):
        print(
            f"Degraded answer: llm={answer.llm_branch}, "
            f"retrieval={answer.retrieval_mode_used}",
            flush=True,
        )
    sources: list[str] = []
    for result in answer.results:
        source = f"{result.source_file} | {result.section_path}"
        if source not in sources:
            sources.append(source)
    print("Sources:", flush=True)
    for source in sources[:5]:
        print(f"- {source}", flush=True)
    # Отказ («нет информации» / OOD-гард) определяем по канонической фразе —
    # обе ветки rag.py используют её дословно. Под отказами кнопок не будет.
    is_refusal = "нет релевантной информации" in answer.answer
    return BotAnswer(
        response=format_answer(answer.answer, sources),
        raw_answer=answer.answer,
        sources=sources,
        llm_branch=answer.llm_branch,
        retrieval_mode=answer.retrieval_mode_used,
        is_refusal=is_refusal,
    )


SEARCHING_PLACEHOLDER = "Ищу ответ, нужно немного подождать…"


def deliver_answer(
    client: TelegramClient,
    chat_id: int,
    placeholder_id: int | None,
    response: str,
) -> int | None:
    """Доставка ответа: заглушка редактируется в первую часть, хвост — новыми.

    Если заглушки нет (не отправилась) или edit не прошёл (например, ответ
    старше 48 часов уже не редактируется) — весь ответ уходит send_message.
    Возвращает id последнего сообщения ответа (для кнопок 👍/👎).
    """
    parts = split_message(response)
    last_id: int | None = None
    if placeholder_id is not None and parts:
        if client.edit_message(chat_id, placeholder_id, parts[0]):
            parts = parts[1:]
            last_id = placeholder_id
    for part in parts:
        sent_id = client.send_message(chat_id, part)
        if sent_id is not None:
            last_id = sent_id
    return last_id


def handle_message(
    *,
    client: TelegramClient,
    assistant: WineRAGAssistant,
    config: BotConfig,
    message: dict[str, object],
    store: FeedbackStore | None = None,
    chat_counters: dict[int, int] | None = None,
    chat_histories: dict[int, list[tuple[str, str]]] | None = None,
) -> None:
    chat = message.get("chat")
    if not isinstance(chat, dict):
        return
    chat_id = chat.get("id")
    if not isinstance(chat_id, int):
        return

    text = message.get("text")
    if not isinstance(text, str) or not text.strip():
        client.send_message(chat_id, "Напишите вопрос о вине или виноделии.")
        return

    question = text.strip()
    print(f"Incoming message from chat_id={chat_id}: {question[:120]!r}", flush=True)
    if question in {"/start", "/help"}:
        client.send_message(
            chat_id,
            "Я VINOTERRA / Нейро-сомелье. Задайте вопрос о вине, регионах, сортах, "
            "гастрономических сочетаниях или объемах производства.",
        )
        print("Help message sent.", flush=True)
        return

    # Индикатор ожидания (пожелание куратора): ответ занимает 30-60 сек, без
    # обратной связи пользователь не понимает, жив ли бот. Заглушка сразу
    # подтверждает приём вопроса, фоновый TypingIndicator держит «печатает…»,
    # затем заглушка редактируется в готовый ответ.
    placeholder_id = client.send_placeholder(chat_id, SEARCHING_PLACEHOLDER)
    started = time.monotonic()
    bot_answer: BotAnswer | None = None
    history = chat_histories.get(chat_id) if chat_histories is not None else None
    try:
        with TypingIndicator(client, chat_id):
            print("Generating answer...", flush=True)
            bot_answer = answer_question(
                assistant, question, top_k=config.top_k, mode=config.mode, history=history
            )
        response = bot_answer.response
    except Exception as exc:
        print(f"Answer generation failed: {exc}", flush=True)
        response = (
            "Не удалось сформировать ответ. Проверьте OPENAI_API_KEY и повторите вопрос.\n"
            f"Техническая ошибка: {exc}"
        )
    latency_ms = int((time.monotonic() - started) * 1000)

    # Память диалога: пары (вопрос, сырой ответ) на чат — для конденсации
    # follow-up вопросов («А что Есенин?»). Ошибки и отказы в историю не
    # пишем: отказ не даёт контекста для follow-up, а хранить его вредно.
    if (
        chat_histories is not None
        and bot_answer is not None
        and not bot_answer.is_refusal
    ):
        chat_histories.setdefault(chat_id, []).append((question, bot_answer.raw_answer))
        del chat_histories[chat_id][:-DIALOG_HISTORY_TURNS]

    # Лог статистики (пожелание куратора): каждый вопрос с метаданными — это
    # автоматизация конвейера «живая эксплуатация -> регрессионные вопросы».
    # Ошибка лога не должна помешать доставке ответа.
    interaction_id: int | None = None
    if store is not None:
        try:
            interaction_id = store.log_interaction(
                chat_id=chat_id,
                question=question,
                answer=bot_answer.raw_answer if bot_answer else response,
                sources=bot_answer.sources if bot_answer else None,
                latency_ms=latency_ms,
                llm_branch=bot_answer.llm_branch if bot_answer else None,
                retrieval_mode=bot_answer.retrieval_mode if bot_answer else None,
                is_refusal=bot_answer.is_refusal if bot_answer else False,
                is_error=bot_answer is None,
            )
        except Exception as exc:
            print(f"Feedback log failed: {exc}", flush=True)

    last_message_id = deliver_answer(client, chat_id, placeholder_id, response)
    print("Answer sent.", flush=True)

    # Кнопки 👍/👎 — под каждым config.feedback_every-м содержательным ответом
    # этого чата (отказы и ошибки не считаются: оценивать в них нечего).
    if (
        store is not None
        and interaction_id is not None
        and chat_counters is not None
        and bot_answer is not None
        and not bot_answer.is_refusal
        and last_message_id is not None
        and config.feedback_every > 0
    ):
        chat_counters[chat_id] = chat_counters.get(chat_id, 0) + 1
        if chat_counters[chat_id] % config.feedback_every == 0:
            client.attach_feedback_buttons(chat_id, last_message_id, interaction_id)


def handle_callback(
    *,
    client: TelegramClient,
    store: FeedbackStore | None,
    callback: dict[str, object],
) -> None:
    """Нажатие 👍/👎: оценка в лог, кнопки убрать, короткий toast-ответ."""
    callback_id = callback.get("id")
    data = callback.get("data")
    if not isinstance(callback_id, str) or not isinstance(data, str):
        return

    parts = data.split(":")
    if len(parts) != 3 or parts[0] != "fb" or parts[1] not in ("up", "down"):
        client.answer_callback(callback_id)
        return
    vote = parts[1]
    try:
        interaction_id = int(parts[2])
    except ValueError:
        client.answer_callback(callback_id)
        return

    if store is not None:
        try:
            store.set_feedback(interaction_id, vote)
            print(f"Feedback recorded: interaction={interaction_id}, vote={vote}", flush=True)
        except Exception as exc:
            print(f"Feedback write failed: {exc}", flush=True)

    client.answer_callback(callback_id, "Спасибо за оценку!")
    message = callback.get("message")
    if isinstance(message, dict):
        chat = message.get("chat")
        message_id = message.get("message_id")
        if isinstance(chat, dict) and isinstance(chat.get("id"), int) and isinstance(message_id, int):
            client.remove_feedback_buttons(chat["id"], message_id)


def run_bot(config: BotConfig) -> None:
    if not config.token:
        raise SystemExit("TELEGRAM_BOT_TOKEN is required.")
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("OPENAI_API_KEY is required.")

    client = TelegramClient(config.token)
    me = client.get_me().get("result", {})
    username = me.get("username") if isinstance(me, dict) else None
    print(f"Telegram token OK. Bot: @{username or 'unknown'}", flush=True)

    assistant, chunk_count = build_assistant(config)
    store = FeedbackStore(config.feedback_db)
    # Счётчик содержательных ответов на чат — для показа кнопок каждому N-му.
    # Живёт в памяти процесса: после рестарта отсчёт начинается заново, это ок.
    chat_counters: dict[int, int] = {}
    # История диалога на чат (память для follow-up вопросов). Тоже в памяти
    # процесса: после рестарта бот «забывает» диалоги — приемлемо для тестовой
    # эксплуатации, диалог легко начать заново.
    chat_histories: dict[int, list[tuple[str, str]]] = {}
    print(
        "VINOTERRA Telegram bot started: "
        f"mode={config.mode}, chunks={chunk_count}, chunk_size={config.chunk_size}, "
        f"overlap={config.chunk_overlap}, top_k={config.top_k}, model={config.chat_model}, "
        f"feedback_db={config.feedback_db}, feedback_every={config.feedback_every}",
        flush=True,
    )
    print("Waiting for Telegram messages. Press Ctrl+C to stop.", flush=True)

    offset: int | None = None
    while True:
        try:
            updates = client.get_updates(offset=offset, timeout=config.poll_timeout)
            if updates:
                print(f"Received {len(updates)} update(s).", flush=True)
            for update in updates:
                update_id = update.get("update_id")
                if isinstance(update_id, int):
                    offset = update_id + 1
                message = update.get("message")
                if isinstance(message, dict):
                    handle_message(
                        client=client,
                        assistant=assistant,
                        config=config,
                        message=message,
                        store=store,
                        chat_counters=chat_counters,
                        chat_histories=chat_histories,
                    )
                callback = update.get("callback_query")
                if isinstance(callback, dict):
                    handle_callback(client=client, store=store, callback=callback)
        except urllib.error.URLError as exc:
            print(f"Telegram network error: {exc}")
            time.sleep(5)
        except RuntimeError as exc:
            print(f"Telegram API/runtime error: {exc}", flush=True)
            time.sleep(5)
        except KeyboardInterrupt:
            print("VINOTERRA Telegram bot stopped.")
            store.close()
            return
