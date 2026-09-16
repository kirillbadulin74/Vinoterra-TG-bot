# Вклад в VINOTERRA

1. Создайте ветку от `main` и опишите пользовательский сценарий, который меняется.
2. Для исправлений RAG добавьте регрессионный тест в `tests/`.
3. Запустите `python -m unittest discover -s tests -v` и
   `python -m py_compile src/*.py`.
4. Не добавляйте `.env`, токены, SQLite-файлы, логи и локальные кэши.
5. В pull request укажите, меняются ли retrieval, prompt или только транспорт.
