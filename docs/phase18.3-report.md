# Отчёт по работе: Phase 18.3 — Per-Candidate VLM Object Description

Полный рабочий отчёт: задача, исходная архитектура, исследование, реализация, итерации,
тестирование, реальные GPU-результаты, проблемы и артефакты. Краткая версия результатов —
в [`phase18.3-results.md`](phase18.3-results.md).

## 1. Исходная задача

Требовалось изменить архитектуру обработки объектов в пайплайне анимации манги:

```
Grounding DINO + SAM2 → bbox → Qwen2.5-VL → animation description → animation model
```

Ключевые требования:
- **Qwen2.5-VL получает одно целое оригинальное изображение + координаты bbox конкретного
  кандидата** — не crop, не визуализацию bbox; модель сама сопоставляет координаты с областью
  изображения.
- **Однозначная система координат**: bbox относится к оригинальному изображению; если Qwen
  требует resize — обеспечить точное соответствие координат (проверить отдельным тестом).
- Qwen определяет: объект в bbox, качество bbox как кандидата, анимируемость,
  движущиеся/неподвижные части, тип движения, направление, амплитуду, характер/скорость,
  ограничения, конфликты с соседями, confidence.
- **Структурированный машинно-парсируемый output**, совместимый с существующим pipeline:
  что анимировать, что нет, как, какие ограничения, уверенность.
- Строгая JSON/schema validation, обработка malformed output, retry/fallback, логирование
  исходного ответа, возможность определить причину отказа.
- **Bbox validation как semantic validation layer**: pass / ambiguous / partial / reject /
  not_animatable — не сводить к одному boolean.
- SAM2 маски сохраняются для последующих стадий, Qwen их не получает.
- Тестирование на реальных данных + набор сценариев (одиночный объект, несколько рядом,
  bbox с несколькими объектами, частичное пересечение, окклюзия, мелкий объект, сложный
  фон, похожие объекты, плохой для анимации объект, частично анимируемый).
- Проверить, что Qwen получает оригинал + координаты, а не crop.
- Интеграция в существующий pipeline, а не отдельный demo.
- Git workflow: ветка, тесты, integration test, реальные изображения, commit, push, PR
  (без само-merge).
- GPU-работы — только на существующем Kaggle worker, с watchdog-пингом каждые ~20 минут.

## 2. Исходная архитектура (на момент старта)

```
run_page_panels / run_pipeline:
  Stage 1  analysis        — Qwen на crop панели → AnimationPlan (что/почему анимировать)
  Stage 2  grounding       — DINO → кандидатные bbox
  Stage 3  validation      — validate_target: bbox plausibility + VLM на CROP bbox
                             + transform geometry
  Stage 4  segmentation    — SAM2 → маска
  Stage 5  mask_semantics  — VLM на crop с overlay маски → ACCEPT/REJECT/ABSTAIN
  Stage 6  animation       — детерминированные CV-трансформы по MotionSpec из
                             keyword-эвристик (_MOTION_HEURISTICS)
  Stage 7-9 reconstruction (LaMa) → compositing → rendering (H.264 + loop-метрики)
```

Проблемы исходной архитектуры относительно задачи:
- Qwen вызывался 4 раза (analysis, validation, mask_semantics, object-description) и
  **ни разу не видел полное изображение с координатами bbox**.
- MotionSpec генерировался эвристиками по `semantic_label`, без учёта реального кандидата.
- Validation работала по crop'у, а не по целому изображению.

## 3. Исследование (подготовительное)

### 3.1 GPU worker (агент A)

Проверено на реальном worker (2×T4, torch 2.10.0+cu128, transformers 5.0.0):

- **Chat template Qwen2.5-VL НЕ поддерживает content type `bbox`** — координаты
  передаются в тексте промпта.
- Токены `<|box_start|>`/`<|box_end|>` — одиночные спец-токены (ids 151648/151649),
  модель обучена на grounding-задачах.
- **image_processor делает smart_resize: округление сторон до кратного 28px**
  (1024×1536 → 1036×1540; 720×5062 → 5068×728). При конфигурационном лимите 1536 наш
  resize + округление даёт 224×1540.
- `rescale_boxes`/`rescale_coordinates` в transformers 5.0 нет — контракт координат
  реализуется вручную.

### 3.2 Локальный код-бейз (агент B)

Изучены конвенции: `VLMClient` protocol, `ModelStage` lifecycle (ADR 0020),
fake-клиенты по маркерам промптов, `Stage` literal, evaluation-зеркала
(`MaskSemanticOutcome` и т.д.), паттерн добавления стадий в `run_pipeline`/
`run_page_panels`.

## 4. Реализованная архитектура (итоговая)

После итераций с пользователем финальная архитектура — **Qwen вызывается ровно один раз
за пайплайн** (одна резидентность на страницу, один `generate()` на панель):

```
Original Page → детекция панелей → scene crops
  ↓
1. GROUNDING — DINO (модель №1, одна загрузка)
     labels от caller (или DEFAULT_ANIMATION_LABELS: character, character_hair,
     flag_banner, weapon, speed_lines, impact_burst)
     → кандидатные bbox (до 3 на label), детерминированный фильтр
  ↓
2. SEGMENTATION — SAM2 (модель №2, одна загрузка)
     → точная маска для каждого кандидата
     (маска сохраняется для анимации; в Qwen НЕ передаётся)
  ↓
3. OBJECT_DESCRIPTION — Qwen2.5-VL (модель №3, ЕДИНСТВЕННЫЙ VLM-вызов,
     одна загрузка на страницу, один generate на панель)
     Вход: ОДНО полное изображение панели + ВСЕ её кандидатные bbox:
       [i] image 0 (WxH px) <|box_start|>(x0,y0),(x1,y1)<|box_end|>
           x0=… y0=… x1=… y1=… label="character"
     Выход: JSON-массив, одна строгая запись на bbox (связка по box_index)
  ↓
4. ANIMATION PLANNING (детерминированный, без VLM)
     PRIMARY = принятый кандидат с max confidence; остальные — SECONDARY
     + transform-geometry гейт (bbox+transform безопасность)
     + cross-panel гейт
  ↓
5. ANIMATION — SAM-маска + MotionSpec (из описания Qwen)
     → generate_transformed_layer (детерминированные CV)
  ↓
6. RECONSTRUCTION (LaMa, модель №4) → COMPOSITING → RENDERING (H.264, loop-метрики)
```

**Удалены**: Qwen-analysis (этап 1), crop-based VLM validation, mask_semantics. Их
детерминированные части (bbox plausibility, transform geometry) сохранены как не-VLM гейты.

### 4.1 Модуль `object_description/`

- **`schema.py`** — `ObjectDescriptionResponse` (pydantic): `box_index`,
  `bbox_assessment` (5 значений), `object_identity`, `matches_semantic_label`,
  `animatable`, `movable_parts`, `static_parts`, `motion_kind`
  (sway/flow/drift/rotate/pulse/breathe/flicker), `direction`, `amplitude_band`,
  `speed_band`, `pivot_hint`, `constraints`, `neighbor_conflicts`, `confidence`,
  `reason`. Кросс-правила: animatable ⇒ motion_kind; drift ⇒ direction; direction для
  не-drift **инертен и срезается** (реальная находка: модель любит заполнять его для
  sway). `ObjectDescriptionBatch = list[ObjectDescriptionResponse]`.
- **`prompt.py`** — координатный контракт:
  - `prepare_image_and_bbox`: downscale до `config.resolution` + округление сторон до
    кратного 28 (`round(x/28)*28`, как у процессора) + **масштабирование bbox ровно
    теми же коэффициентами**.
  - `build_multi_prompt`: одно изображение + все bbox с `<|box_start|>`-токенами,
    явными числами и размерами изображения.
  - Промпт построен как **чтение действия в сцене**: STEP 1 READ THE ACTION (что
    происходит: персонаж бежит, ветер, взрыв, дождь), STEP 2 JUDGE THE CANDIDATE с
    учётом действия, STEP 3 DESCRIBE THE MOTION THE ACTION GIVES THIS OBJECT.
    Правила: текст/пузыри никогда не анимируются; несколько экземпляров ⇒ ambiguous;
    предпочесть строже при неуверенности.
- **`mapping.py`** — детерминированный маппинг описания → schema-valid `MotionSpec`
  (базовые амплитуды/езэинги из существующих эвристик, bands → множители, speed →
  целые циклы для seamless loop).
- **`describe.py`** — `describe_objects(image, candidates, vlm_client, ...)`: один
  `generate()` на все bbox; `_parse_batch` (JSON-массив, box_index, все индексы
  обязательны); 1 recovery re-prompt с повторением допустимых значений; fail-closed на
  кандидата с `rejection_reason`;
  `_NON_ANIMATABLE_IDENTITY_KEYWORDS` (speech_bubble/text/background/panel/lettering…) —
  детерминированный backstop; raw-ответы логируются и сохраняются.

### 4.2 Интеграция

- `pipeline/orchestrator.py` — новый candidate-driven `run_pipeline(image, config,
  labels, ...)`: 4 ModelStage (DINO → SAM → Qwen → LaMa), `_ground_labels`,
  `_segment_candidates`, `_describe_candidates`, `_build_plan` (ранжирование + PRIMARY +
  гейты), переиспользованы `_animate_objects`/`_reconstruct_objects`/
  `_composite_and_render`.
- `pipeline/panels.py` — `run_page_panels` в той же логике: Qwen грузится один раз на
  страницу и обрабатывает все панели; `crop_local_panel_bbox` для локальных координат;
  манифест + resume сохранены.
- `pipeline/types.py` — `Stage` + `object_description`; `ObjectDescriptionResult`;
  `DroppedObjectResult` (3 failing stage).
- `analysis/client.py` — `max_new_tokens` по умолчанию 4096 (батч из 10 боксов требует
  ~2-3k токенов).
- `core/config.py` + `configs/default.yaml` — `enable_object_description_validation: true`.
- `evaluation/` — `ObjectDescriptionOutcome`, schema_version 7, harness под новый shape.
- `scripts/run_phase18_3_e2e.py` — E2E GPU-скрипт с счётчиком VLM-вызовов (доказательство
  «один вызов на панель»).

## 5. Промпт-инжиниринг (реальные находки из GPU-прогонов)

Серия итераций по реальным ответам Qwen:

1. Модель шлёт `null` в `amplitude_band`/`speed_band`/`pivot_hint`/`constraints` →
   опциональные поля стали толерантными (defaults), семантические — строгими.
2. Модель изобретает значения вне enum (`bbox_assessment="static"`,
   `direction="up_down"`) → recovery-промпт повторяет допустимые значения; direction
   для не-drift срезается схемой.
3. Текст не анимируется: модель предлагала sway для text banner → явное правило
   «lettering never animatable», фактически `animatable=false` ⇒ отклонение.
4. Перестраховка: модель читала статичных персонажей как «static pixelated figures» →
   CONTEXT-блок: «обычный персонаж/оружие/ткань — нормальный анимируемый таргет».
5. Критический **false-accept**: recovery-ответ `object_identity="speech_bubble"` с
   `matches=true`+`animatable=true` проходил — добавлен детерминированный identity
   backstop.
6. Батч-ответы обрезались на 512 токенов → 4096.
7. Переформулировка по требованию пользователя: не «что можно анимировать», а
   **«какие действия происходят»** — движение выводится из действия.

## 6. Тестирование

### 6.1 Unit/integration (локально, без GPU)

- **664 теста зелёные**, `ruff check .` и `mypy src` чисто.
- `tests/test_object_description.py` (28 тестов): координатный контракт (bbox
  масштабируется ровно с изображением; клиент получает полное изображение; размеры
  изображения в промпте; native box-токены), schema-валидация (drift⇒direction,
  animatable⇒motion_kind, unknown enum ⇒ fail), fail-closed (все 4 не-pass статуса,
  label mismatch, not_animatable, identity conflict и варианты, malformed→recovery,
  malformed дважды→reject, audit-trail), маппинг (все motion kinds → schema-valid spec,
  единичные векторы, bands).
- `tests/test_pipeline.py` (13 тестов нового потока): E2E PASS-рендер, **ровно один
  VLM-вызов на все кандидаты** (call_count==1, число боксов в промпте == число
  кандидатов), полное изображение а не crop, fail-closed при нуле принятых и при
  unparseable, identity backstop, multi-object ранжирование, label-промпты, default
  labels, run_page_panels с одной VLM-резидентностью.
- `tests/test_lifecycle.py`: одна загрузка/выгрузка на модель; отказ
  object-description держит панели REJECTED; изоляция панельных ошибок.

### 6.2 Реальные GPU-прогоны (Kaggle 2×T4)

- **Координатный контракт**: `prepare_image_and_bbox` → реальный processor: все 5
  размеров `match=True` (1024×1536→1036×1540; 720×5062→224×1540 после лимита 1536;
  600×400→588×392; 1536×1536→1540×1540; 200×220→196×224).
- **Реальные страницы, DINO-кандидаты** (part C, 5 страниц, 25 кандидатов): 6 accepted,
  15 ambiguous, 2 reject, 1 partial, 1 unparseable. angels_of_war_fleet: все 12
  кандидатов корректно отклонены (spaceships/missiles вместо character/weapon);
  villainess: правильный персонаж принят на 3-м ранге DINO; marika: пузыри речи пойманы
  как ambiguous.
- **Итоговый E2E (wind_breaker_sprint)**: **все 4 панели PASS**, реальные 96-кадровые
  H.264 видео; **VLM calls: 2, boxes per call: [10, 9]** — ровно один вызов на панель
  со всеми боксами; декодированные видео: 79–87/95 пар кадров движутся. Примеры
  реальных описаний: «The character is in motion, with visible sweat indicating
  exertion», «The hair is flowing due to the character's movement», «The speed lines
  indicate movement and can be animated», «The flag is waving due to the wind», «The
  impact burst indicates a forceful impact and can be animated».

## 7. Обнаруженные проблемы (и статус)

| Проблема | Статус |
|---|---|
| Chat template не поддерживает bbox content | решено: координаты в тексте + native токены |
| Processor округляет до кратных 28 | решено: контракт воспроизводит геометрию точно (проверено) |
| Обрезка батч-ответа на 512 токенов | решено: 4096 |
| Изобретённые enum-значения | решено: recovery + срез инертных полей |
| False-accept speech_bubble | решено: identity backstop |
| Текст/эффекты читаются как анимируемые/неанимируемые нестабильно | частично: fail-closed; документированный noise модели |
| Шум вердиктов на пограничных случаях (окклюзия, эффекты) | документированный ceiling качества, не дыра безопасности |

## 8. Итоговые артефакты

- Ветка `phase-18.3-vlm-bbox-description`, запушена (коммиты до `c95c136`).
- **PR #14**: https://github.com/cld111/manga-animation/pull/14 (не мёржался).
- `docs/phase18.3-results.md` — краткие результаты фазы; `docs/current-status.md` и
  `docs/pipeline.md` — актуальное состояние проекта.
- Артефакты прогонов (JSON с raw-ответами, видео, манифесты) — под
  `outputs/experiments/` на worker (git-ignored).

**Главный критерий достигнут**: Qwen2.5-VL видит целое оригинальное изображение и по
переданным координатам bbox самостоятельно определяет объект, оценивает пригодность к
анимации и формирует структурированное описание, которое реально используется следующей
стадией пайплайна (анимация работает на SAM-масках + MotionSpec из описания).
