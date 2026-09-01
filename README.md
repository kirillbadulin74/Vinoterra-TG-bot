# VINOTERRA — Нейро-сомелье

RAG-ассистент по вину и виноделию: Telegram-бот, который отвечает на вопросы,
опираясь на собственную базу знаний, а не на «память» языковой модели. Гибридный
поиск (BM25 + векторный, reciprocal rank fusion), генерация через OpenAI с
фоллбэком на DeepSeek и деградацией поиска до офлайн-BM25 при недоступности API.

Проект вырос из дипломной работы: retrieval-схема выбрана не «на вкус», а
контролируемым экспериментом на новых формулировках вопросов (см. [Как выбрана
схема поиска](#как-выбрана-схема-поиска)).

## Что внутри

- **Гибридный retrieval** — BM25 (лексика: имена, сорта, числа) и векторный поиск
  (семантика) объединяются через RRF; поверх — section expansion (добор соседних
  фрагментов раздела).
- **Заземление ответа** — модель отвечает только из найденных фрагментов базы;
  доменный гард отсекает вопросы вне темы вина.
- **Память диалога** — follow-up-вопросы («А что там у Грузии?») переписываются в
  самостоятельные до поиска, история 3 пары реплик.
- **Устойчивость** — при недоступности OpenAI поиск деградирует до BM25, генерация
  уходит на DeepSeek; бот продолжает отвечать.
- **Фидбэк** — кнопки 👍/👎 под ответами, лог вопросов и оценок в SQLite.

## Архитектура

```
run_telegram_bot.py        точка входа: Telegram long polling
run_vinoterra_rag.py       CLI для отладки и приёмки ответов
build_vector_index.py      разовая пред-сборка кеша эмбеддингов

src/chunking.py            heading-aware разбиение Markdown-базы
src/retrieval.py           BM25, векторный/гибридный поиск, форматирование источников
src/assistant_factory.py   сборка ассистента, кеширование индекса
src/rag.py                 RAG-ядро: RRF, section expansion, гард, память, промпт
src/telegram_bot.py        Telegram-интерфейс поверх ядра
src/feedback_store.py      SQLite-лог вопросов и оценок

knowledge_base/            база знаний (Markdown)
vector_index/              кеш эмбеддингов (готов к использованию)
```

## Запуск

Нужен Python 3.12. Зависимости — в `requirements.txt`.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate    |    Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Скопируйте `.env.example` в `.env` и заполните ключи (`OPENAI_API_KEY`,
`TELEGRAM_BOT_TOKEN`). Кеш эмбеддингов уже лежит в `vector_index/` — пересобирать
не нужно.

```bash
python run_telegram_bot.py
```

Ожидаемый вывод: `Telegram token OK` → `VINOTERRA Telegram bot started: mode=hybrid`
→ `Waiting for Telegram messages`.

Полезные флаги:

```bash
python run_telegram_bot.py --mode bm25          # офлайн-поиск без OpenAI на этапе поиска
python run_telegram_bot.py --feedback-every 5   # кнопки 👍/👎 под каждым 5-м ответом
python run_telegram_bot.py --chat-model gpt-4o  # другая основная модель
```

Проверить поиск без ключа OpenAI (режим bm25 работает офлайн):

```bash
python run_vinoterra_rag.py "Каков объём производства вина в Австралии и Чили?" --mode bm25 --no-answer
```

## Конфигурация

```text
retrieval:      hybrid (BM25 + vector, RRF) + section expansion
embeddings:     text-embedding-3-small (кеш vector_index/)
chunk_size:     500,  overlap: 100,  top_k: 8,  temperature: 0.0
base model:     gpt-4o-mini
quality model:  gpt-4o
fallback:       DeepSeek (vedai.by) — при недоступности OpenAI
```

## Как выбрана схема поиска

Итоговый гибрид выбран по результатам контролируемого эксперимента, а не по
умолчанию. Прежняя схема «BM25 + ручной scope-рулбук» была идеальна на заранее
проработанных вопросах (Hit@1 `1.000`), но на **новых** формулировках оказывалась
худшей из осмысленных (`0.625`) — то есть переобучалась под известные вопросы.
Векторный и гибридный поиск на новых вопросах обобщали лучше (`0.750` / `0.708`),
поэтому ручной scope из прод-пути убран полностью, а section expansion сохранён.

Cross-encoder reranking даёт измеримый, но умеренный выигрыш (gold-in-context
`0.917 → 1.000`) ценой GPU — вынесен в улучшения на будущее, в прод не включён.

## Стек

Python 3.12, OpenAI API (генерация + эмбеддинги), NumPy (векторный индекс),
Telegram Bot API через стандартную библиотеку (без сторонних фреймворков),
SQLite (фидбэк). Внешних RAG-фреймворков нет — поиск, слияние и чанкинг написаны
в проекте.
