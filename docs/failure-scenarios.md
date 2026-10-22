# Сценарии сбоев, воспроизведённые намеренно

Код, который видел только удачный путь, не проверен вовсе. Для каждого сбоя из списка роадмапа
ниже указано, что видит клиент, что происходит внутри и какой тест это закрепляет. Ссылки на тесты
проверяет сам набор тестов:
`test_every_scenario_in_the_document_points_at_a_real_test` падает, если тест из этого файла
переименован или удалён.

| Сбой | Что видит клиент | Что происходит внутри |
|---|---|---|
| Некорректный HTTP/JSON | 422, адрес ошибки в `loc` | модель не вызывается, ничего не пишется |
| Валидный JSON против схемы | 422 | то же |
| Таймаут модели | 504 `llm_timeout` | 3 попытки по 60 с с паузами, в БД ничего |
| Провайдер отвечает 429 | при восстановлении 200, иначе 502 | 429 провайдера повторяется; наружу не пробрасывается |
| Провайдер отвечает 503 | то же | то же |
| PostgreSQL недоступна | 503 `database_unavailable` | до вызова модели — модель не зовётся; после — откат |
| Redis недоступен | запросы обслуживаются | кэш, лимит и идемпотентность отключаются на 30 с |
| Переполнение контекста | 400 `context_length_exceeded` | модель не вызывается |
| Невалидный структурированный ответ | 502 `invalid_model_output:{kind}` | повтор с обратной связью, затем отказ; расход записан |
| ~20 одновременных запросов | все 200 | не больше 5 вызовов модели одновременно |
| Повтор с тем же `Idempotency-Key` | тот же ответ, `Idempotent-Replay` | второй генерации и записей нет |
| Отключение клиента при стриминге | — | генерация остановлена, слот свободен, обмен не сохранён |
| Сбой БД внутри транзакции | 503 | откат, обмен не сохранён |
| Неверный или пропавший ключ | 401 | запрос не доходит ни до тела, ни до модели |

Дополнительно: превышение лимита — 429 с `Retry-After`; сбой после начала потока — событие `error`.

## Подробно

### Некорректный HTTP/JSON и нарушение схемы

Запрос, который не разбирается как JSON или не проходит `ChatRequest` (нет сообщения `user`,
`temperature` вне 0..2, лишнее поле), получает 422 в формате FastAPI до любой работы.

- `tests/test_chat.py::test_malformed_json_returns_422`
- `tests/test_chat.py::test_invalid_request_returns_422`

### Таймаут модели

Попытка прерывается через `LLM_TIMEOUT_SECONDS`, это `LLMTimeoutError` с признаком повторяемости.
После паузы с jitter следующая попытка; когда попытки кончились, клиент получает 504. Худший случай
по времени — больше трёх минут, общего дедлайна запроса нет (известное ограничение).

- `tests/test_chat.py::test_slow_provider_returns_504`
- `tests/test_chat_service.py::test_timeout_is_retried_until_success`

### Провайдер отвечает 429 или 503

Оба кода повторяемы. Если провайдер оправился, клиент получает 200 и не замечает сбоя. Если нет,
после исчерпания попыток приходит **502, а не 429**: для нашего клиента это сбой вышестоящего
сервиса, а 429 зарезервирован за превышением его собственного лимита. Число вызовов провайдера
ограничено.

- `tests/test_reliability.py::test_provider_overload_is_retried_and_then_succeeds`
- `tests/test_reliability.py::test_provider_overload_that_does_not_clear_becomes_a_502_after_bounded_retries`
- `tests/test_chat.py::test_transient_provider_error_is_retried`

### PostgreSQL недоступна

Первое обращение к БД бросает ошибку соединения; обработчик отвечает **503 `database_unavailable`**
без текста исходной ошибки (в нём может быть строка подключения). Если диалог указан в запросе,
проверка идёт до вызова модели, и генерация не оплачивается зря. Если сбой случился при записи
после ответа модели, транзакция явно откатывается, а не остаётся открытой на соединении.

Для потока: если заголовки уже отправлены, статус 200 менять поздно, и сбой записи приходит
событием `error: storage_unavailable`, а не обрывом без слов.

- `tests/test_reliability.py::test_database_down_before_the_call_is_a_503_and_the_model_is_not_called`
- `tests/test_reliability.py::test_database_failure_inside_the_transaction_rolls_back_and_is_a_503`
- `tests/test_reliability.py::test_database_down_after_a_stream_has_started_is_reported_inside_the_stream`
- `tests/test_reliability.py::test_database_error_text_never_reaches_the_client`

Эта ветка кода появилась именно из такого прогона: до него неработающая БД давала 500 с
трассировкой в логе и ничего внятного клиенту.

### Redis недоступен

Редис нужен трём возможностям, и каждая выбрала свою деградацию:

| Возможность | При сбое | Почему |
|---|---|---|
| Кэш | промах, отвечает модель | кэш — оптимизация |
| Rate limit | запросы проходят | защищает бюджет, а не корректность; семафор остаётся |
| Идемпотентность | запрос выполняется без защиты | доступность важнее гарантии, повтор не отсекается |
| Статистика кэша | 503 | запасного источника нет |

Во всех случаях после первой ошибки Redis не опрашивается 30 секунд, чтобы запросы не ждали
таймаут (0.5 с) каждый.

- `tests/test_reliability.py::test_every_redis_feature_degrades_without_failing_the_request`
- `tests/test_cache.py::test_redis_outage_falls_back_to_model_and_pauses_cache`
- `tests/test_ratelimit.py::test_requests_pass_when_the_limiter_is_down`
- `tests/test_idempotency.py::test_redis_outage_runs_requests_unprotected`
- `tests/test_cache.py::test_stats_endpoint_returns_503_when_redis_is_down`

### Переполнение контекстного окна

Оценка входа плюс `max_tokens` больше окна: 400, модель не вызывается. Ollama без этой проверки
молча обрезала бы промпт и ответила 200.

- `tests/test_context.py::test_oversized_chat_is_rejected_before_calling_the_model`
- `tests/test_context.py::test_oversized_structured_request_is_rejected`

### Невалидный структурированный ответ

Ответ не JSON или не по схеме: один повтор с показом модели её ошибки, потом 502
`invalid_model_output:{kind}`. Обрыв по `max_tokens` — сразу 502 без повтора. Токены всех попыток
записываются в журнал расхода, включая неудачные.

- `tests/test_analysis.py::test_retries_are_bounded_and_last_reason_is_reported`
- `tests/test_analysis.py::test_truncated_output_fails_fast`
- `tests/test_analysis.py::test_failed_generations_are_logged_too`

### Около 20 одновременных запросов

20 запросов приходят разом через настоящий ASGI-стек. Все получают 200, а модель вызывается не
больше чем в 5 потоках одновременно: остальные ждут слот. Тест считает перекрывающиеся вызовы, а не
верит в лимит на слово.

- `tests/test_reliability.py::test_twenty_concurrent_requests_all_succeed_within_the_concurrency_limit`
- `tests/test_chat_service.py::test_concurrent_provider_calls_never_exceed_limit`

### Повтор запроса с тем же ключом идемпотентности

Первый запрос выполняется и сохраняет ответ; повтор получает тот же ответ байт в байт с
`Idempotent-Replay: true`. Модель не вызывается, сообщения и строка расхода не дублируются.

- `tests/test_idempotency.py::test_retry_with_the_same_key_gets_the_first_response_without_a_second_run`
- `tests/test_idempotency.py::test_simultaneous_requests_with_one_key_run_exactly_once`
- `tests/test_idempotency.py::test_same_key_with_a_different_request_is_rejected`

### Отключение клиента во время стриминга

Тест ведёт ASGI-соединение вручную: клиент получил первый кусок и прислал `http.disconnect`.
Starlette отменяет поток, закрывается цепочка генераторов, закрывается поток к модели, слот
семафора свободен. Тест не ждёт 30-секундную паузу провайдера: он заканчивается за доли секунды.
Обмен не сохраняется, расход не пишется (счётчика токенов у оборванного потока нет).

- `tests/test_reliability.py::test_client_disconnect_stops_the_generation_and_frees_the_slot`
- `tests/test_streaming.py::test_client_disconnect_closes_the_generation_and_frees_the_slot`
- `tests/test_streaming.py::test_failure_after_text_started_becomes_an_error_event_and_stores_nothing`

### Сбой БД внутри транзакции

Обмен (диалог, вопрос, ответ, строка расхода) пишется одной транзакцией. Сбой между вставками
откатывает всё: ни диалога без сообщений, ни ответа без вопроса. Проверено и на фейковой сессии, и
на настоящей PostgreSQL.

- `tests/test_chat.py::test_failure_between_writes_rolls_back_the_whole_exchange`
- `tests/test_db_integration.py::test_failure_between_writes_leaves_nothing_behind`

### Неверный или пропавший ключ

401 с одинаковым телом в обоих случаях и `WWW-Authenticate`. Проверка идёт раньше разбора тела, так
что анонимный клиент не узнаёт формат запроса. Знакомый клиент без нужного права получает 403, а
не 401.

- `tests/test_auth.py::test_every_protected_endpoint_requires_a_key`
- `tests/test_auth.py::test_wrong_key_answers_exactly_like_a_missing_one`
- `tests/test_auth.py::test_known_client_without_the_scope_gets_403_not_401`

### Превышение лимита запросов

- `tests/test_ratelimit.py::test_over_the_limit_is_a_429_with_retry_information`
- `tests/test_ratelimit.py::test_concurrent_requests_never_exceed_the_limit`

## Какие тесты на чём работают

| Уровень | Что настоящее | Что подменено |
|---|---|---|
| Контракт API | FastAPI, обработчики, валидация | провайдер, сессия, Redis |
| `fakeredis` | команды и Lua по семантике Redis | сам сервер |
| ASGI вручную | весь стек, отключение клиента | провайдер, сессия |
| Интеграционные | PostgreSQL и Redis | модель |

Что фейк Redis не покажет и проверяют только настоящие сервер и база: гонка при настоящих
параллельных клиентах по сети, истечение ключей по часам сервера, внешние ключи и каскады, откат
транзакции в самой БД. Эти тесты помечены `integration` и без контейнеров пропускаются.
