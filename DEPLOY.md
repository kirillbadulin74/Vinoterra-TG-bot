# Деплой Telegram-бота

Production-процесс запускается одним systemd-сервисом из каталога приложения.
Перед обновлением:

```bash
git pull --ff-only
.venv/bin/pip install -r requirements.txt
python -m unittest discover -s tests -v
python -m py_compile src/*.py
```

Кеш эмбеддингов `vector_index/` в Git не хранится — соберите его при первом
деплое и после любых правок `knowledge_base/`. Шаг требует `OPENAI_API_KEY`
(эмбеддинги `text-embedding-3-small`) и разово расходует API:

```bash
.venv/bin/python build_vector_index.py
```

Перезапустите сервис и проверьте журнал:

```bash
systemctl restart vinoterra-bot.service
systemctl is-active vinoterra-bot.service
journalctl -u vinoterra-bot.service -n 50 --no-pager
```

Сервис должен запускать только один процесс long polling. Секреты передаются
через `.env` и не копируются в Git.
