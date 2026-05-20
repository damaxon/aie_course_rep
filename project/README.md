# Итоговый проект по курсу «Инженерия Искусственного Интеллекта»

В этой папке находится итоговый мини-проект по курсу.  
Проект должен демонстрировать применение методов и инструментов инженерии ИИ: работу с данными, модели, пайплайны, сервис, эксперименты и (по возможности) воспроизводимость.

---

## 1. Паспорт проекта

- **Название проекта:** Детекция транспортных средств
- **Автор:** Петрушинский Максим Геннадьевич
- **Группа:** БАСО-01-24
- **Контакт:** @damaxon

- **Краткое описание:**  
  > Проект посвящён построению сервиса детекции объектов дорожной сцены.
  > Используется detection CV модель для определения транспортных объектов на изображении.
  > Проект может использоваться для систем умного города, мониторинга дорожного трафика и задач автопилота.
  > Результат - REST API, принимающий на вход изображение (кадр видео) и возвращающий список найденных объектов с координатами bounding box и confidence score.

---

## 2. Структура проекта

Проект организован в следующей структуре:

- `report.md` - отчёт по проекту (постановка задачи, данные, эксперименты, результаты).
- `self-checklist.md` - чеклист самопроверки проекта перед сдачей.
- `notebooks/` - экспериментальные ноутбуки:
  - EDA;
  - baseline модель;
  - эксперименты с detection-моделями.
- `src/` - основной код проекта:
  - `src/data/` - CLI и модули загрузки, организации и подготовки данных;
  - `src/data/datasets/` - Dataset-классы для detection-задачи;
  - `src/data/loaders.py` - создание dataloaders для обучения и валидации;
  - `src/data/transforms.py` - transforms для object detection;
  - `src/models/` - построение detection-модели, обучение, evaluation и inference;
  - `src/api.py` - FastAPI-приложение;
  - `src/cli.py` - CLI-команды для проверки путей, артефактов и запуска сервиса;
  - `src/logger.py` - базовое логирование работы приложения.
- `data/` - структура хранения данных:
  - `raw/`;
  - `organized/`;
  - `processed/`.
- `configs/` - конфигурационные файлы:
  - `.env.example`;
  - `kaggle.json` (локально, не хранится в репозитории).
- `logs/` - автоматически создаваемая директория для логов API:
  - `app.log` - основной лог-файл сервиса.
- `tests/` - smoke/sanity тесты проекта.
- `artifacts/` - артефакты проекта:
  - `models/` - локальное хранение весов обученной модели;
  - `metrics/` - metadata и метрики финальной модели;
  - `baseline_models/` - результаты baseline-эксперимента;
  - `vehicle_detection/` - некоторые результаты и чекпоинты экспериментов обучения моделей детекции.
- `pyproject.toml` - зависимости и конфигурация uv-проекта.
- `uv.lock` - lock-файл окружения.

---

## 3. Требования и установка

### 3.1. Требования

- Python `3.14`;
- `uv`;
- доступ к интернету для загрузки зависимостей, датасета и pretrained-весов;
- Kaggle API token для автоматической загрузки датасета.

### 3.2. Установка окружения

```bash
# Перейти в папку проекта
cd project

# Установить зависимости
uv sync

# Установить dev-зависимости
uv sync --group dev
```

Проверить установку можно командами:

```bash
uv run python -V
uv run python -m src.cli paths
```

Если необходимо загрузить датасет через Kaggle API, вручную добавьте:

```text
configs/kaggle.json
```

Внутри JSON-файла должен быть ваш индивидуальный ключ в формате (информацию как получить API-ключ можно найти на сайте: `kaggle.com`):
```json
{"username":"<username>","key":"<key>"}
```

---

## 4. Как запустить проект

### 4.1. Подготовка данных

```bash
cd project

# Скачать датасет
uv run python -m src.data.cli download

# Организация данных
uv run python -m src.data.cli organize

# Подготовка данных
uv run python -m src.data.cli prepare
```

В результате будут сформированы данные в:

```text
data/processed/detection/
```

### 4.2. CLI-команды проекта

В проекте реализованы CLI-команды для основных сценариев работы с данными и сервисом. 

Команды для работы с данными:

```bash
# Скачать исходный датасет (при наличии файла configs/kaggle.json описанного в пункте 3.2.)
uv run python -m src.data.cli download

# Организовать структуру raw-данных
uv run python -m src.data.cli organize

# Подготовить processed-данные для обучения detection-модели
uv run python -m src.data.cli prepare

# Полный pipeline: download -> organize -> prepare
uv run python -m src.data.cli full

# Очистка raw-данных 
uv run python -m src.data.cli clean
```

Сервисные команды:

```bash
# Показать основные пути проекта
uv run python -m src.cli paths

# Проверить наличие артефактов модели
uv run python -m src.cli check-artifacts

# Запустить API через CLI
uv run python -m src.cli run-api
```

CLI используется для воспроизводимого запуска пайплайна без ручного выполнения отдельных Python-скриптов.

### 4.3. Веса модели

Из-за ограничения GitHub на размер файлов (`100 MB`) файл обученной модели не хранится напрямую в репозитории.

Скачать веса модели можно через GitHub Releases:

```text
https://github.com/damaxon/aie_course_rep/releases/tag/v0.7
```

После скачивания файл:

```text
best_detector.pt
```

необходимо поместить в директорию:

```text
project/artifacts/models/
```

Если директория отсутствует, её можно создать командой:

```bash
cd project

mkdir artifacts/models
```

Итоговая структура:

```text
project/
└── artifacts/
    └── models/
        └── best_detector.pt
```

Metadata финальной модели хранится в репозитории:

```text
project/artifacts/metrics/best_detector_meta.json
```

Проверить наличие артефактов можно командой:

```bash
uv run python -m src.cli check-artifacts
```

### 4.4. Повторное обучение модели

В проекте реализован модуль повторного обучения detection-модели:

```text
src/models/train_detection.py
```

Обучение detection-модели является длительным и ресурсоёмким процессом. В исходных экспериментах обучение на полном наборе данных занимало несколько часов, поэтому inference-сервис по умолчанию использует уже сохранённый артефакт модели:

```text
artifacts/models/best_detector.pt
```

Модуль обучения нужен для воспроизводимости проекта и повторного обучения модели при необходимости.

При новом обучении модель сначала сохраняется в отдельную директорию run-артефактов:

```text
artifacts/runs/detection/
```

Текущая лучшая модель не должна перезаписываться случайно. Для замены `best_detector.pt` предусмотрены дополнительные подтверждения в конфигурации обучения.

Основная логика обучения включает:
- создание dataloaders;
- построение detection-модели;
- обучение по эпохам;
- evaluation на validation-части;
- выбор лучшей модели по `mAP@0.5`;
- early stopping;
- сохранение history, metadata и checkpoint;
- возможность promote нового run в best-model.

Пример программного запуска обучения:

```python
from src.models import DetectionTrainConfig, fit_detection_model

config = DetectionTrainConfig(
    epochs=10,
    batch_size=4,
    confirm_retrain=True,
    confirm_long_run=True,
    promote_to_best=False,
)

result = fit_detection_model(config)
print(result)
```

Если необходимо заменить текущую лучшую модель, требуется явно включить:

```python
promote_to_best=True
confirm_overwrite_best=True
```

Это сделано для защиты от случайного запуска долгого обучения и случайной потери уже сохранённой лучшей модели.

### 4.5. Запуск сервиса (API)

```bash
cd project

uv run uvicorn src.api:app --reload
```

Сервис запускается на:

```text
http://127.0.0.1:8000
```

Swagger UI (рекомендуемый формат работы с сервисом):

```text
http://127.0.0.1:8000/docs
```

### 4.6. Проверка API

Основные endpoints:

- `GET /health` - проверка состояния сервиса;
- `GET /info` - информация о загруженной модели, классах и метриках;
- `POST /predict` - детекция объектов на изображении;
- `POST /visualize-detections` - визуализация bounding boxes на изображении по результатам `/predict`. JSON с результатами можно передать текстом или отдельным JSON-файлом.

Рекомендуемый сценарий проверки:

1. Открыть Swagger UI:

```text
http://127.0.0.1:8000/docs
```

2. Выполнить `GET /health`.

3. Выполнить `GET /info`.

4. Загрузить изображение в `POST /predict`.

5. Скопировать JSON-ответ из `/predict` и вместе с исходным изображением передать в `POST /visualize-detections`.

### 4.7. Логи сервиса

При запуске API автоматически создаётся директория:

```text
logs/
```

Основной лог-файл сервиса:

```text
logs/app.log
```

В логах фиксируются:
- запуск сервиса и загрузка артефактов модели;
- входящие HTTP-запросы;
- endpoint, HTTP-метод и status code;
- время обработки запроса;
- параметры inference-запроса;
- количество найденных объектов;
- ошибки чтения изображений и JSON.

Пример проверки логов после нескольких запросов:

```bash
cat logs/app.log
```

Либо открыть файл вручную:

```text
logs/app.log
```

### 4.8. Типовые сценарии запуска

Для полного развёртывания проекта после клонирования репозитория необходимы:
- Kaggle API token для загрузки датасета;
- файл весов `best_detector.pt`, скачанный из GitHub Releases.

#### Сценарий 1: запуск только inference-сервиса

Если данные уже подготовлены, а веса модели скачаны из GitHub Releases:

```bash
cd project

uv sync
uv run python -m src.cli check-artifacts
uv run uvicorn src.api:app --reload
```

#### Сценарий 2: полный запуск с подготовкой данных

```bash
cd project

uv sync
uv sync --group dev

uv run python -m src.data.cli download
uv run python -m src.data.cli organize
uv run python -m src.data.cli prepare

uv run python -m src.cli check-artifacts
uv run uvicorn src.api:app --reload
```

#### Сценарий 3: проверка проекта перед сдачей

```bash
cd project

uv run pytest tests
uv run python -m src.cli check-artifacts
uv run uvicorn src.api:app --reload
```

---

## 5. Данные

Для проекта используется:

> Udacity Self Driving Car Dataset

Датасет содержит:
- изображения дорожной сцены;
- bounding boxes;
- классы транспортных объектов.

Используемые классы:
- biker;
- car;
- pedestrian;
- trafficLight;
- trafficLight-Green;
- trafficLight-GreenLeft;
- trafficLight-Red;
- trafficLight-RedLeft;
- trafficLight-Yellow;
- trafficLight-YellowLeft;
- truck.

Полные `raw/organized/processed` данные не хранятся в репозитории из-за большого размера.
Данные загружаются автоматически через pipeline проекта (необходимо указать свой Kaggle API ключ по инструкции из пункта `3.2`).

Pipeline данных состоит из трёх основных этапов:

1. `download`  
   Загружает исходный датасет через Kaggle API в директорию `data/raw/`.

2. `organize`  
   Приводит структуру исходных файлов к единому виду, ожидаемому проектом.

3. `prepare`  
   Формирует processed-слой данных для обучения и inference:
   - копирует изображения в `data/processed/detection/images/`;
   - создаёт единый файл аннотаций `data/processed/detection/annotations.csv`;
   - нормализует названия колонок;
   - сохраняет данные в формате, удобном для `DetectionDataset`.

Основной файл processed-аннотаций:

```text
data/processed/detection/annotations.csv
```

Ожидаемые колонки: 

```text
image, label, xmin, ymin, xmax, ymax, image_width, image_height
```

---

## 6. Эксперименты и модели

Экспериментальная часть проекта находится в директории:

```text
notebooks/
```

Основные ноутбуки:
- `01_eda.ipynb` - разведочный анализ данных после этапа `organize` данных;
- `02_baseline_detection.ipynb` - baseline detection-модель;
- `03_detection_experiments.ipynb` - основные эксперименты с detection-моделями и выбор финальной модели.

### Baseline

В качестве baseline использовалась `Faster R-CNN ResNet50 FPN` с предобученными COCO-весами без дообучения на целевом датасете.

Baseline нужен для оценки качества универсальной pretrained-модели до fine-tuning на Udacity Self Driving Car Dataset.

Для корректной оценки выполнялся mapping классов COCO в классы проекта.

Метрики baseline:
- `precision@0.5`;
- `recall@0.5`;
- `f1@0.5`;
- `mAP`;
- `mAP@0.5`;
- `mAP@0.75`.

### Основные эксперименты

В основном экспериментальном ноутбуке были обучены и сравнены detection-модели torchvision.

Финальная модель выбрана по ключевой метрике:
```text
mAP@0.5
```

В качестве финальной модели используется:
```text
fasterrcnn_resnet50_fpn
```

Сохранённые артефакты финальной модели:
- `artifacts/models/best_detector.pt`;
- `artifacts/metrics/best_detector_meta.json`.

---

## 7. Тесты 

В проекте реализованы smoke/sanity тесты:
- проверка процессинга данных;
- проверка DetectionDataset;
- проверка dataloaders;
- проверка artifacts;
- проверка API;
- проверка утилит для предсказания.

Тесты расположены в директории:

```text
tests/
```

Основные файлы тестов:
- `test_data_processing.py` - проверка processed-данных и формата аннотаций;
- `test_detection_dataset.py` - проверка `DetectionDataset`;
- `test_dataloaders.py` - проверка dataloaders;
- `test_model_artifacts.py` - проверка наличия и структуры model artifacts;
- `test_model_metadata.py` - проверка metadata финальной модели;
- `test_prediction_utils.py` - проверка утилит визуализации/prediction;
- `test_api.py` - проверка `/health`.

Команда запуска:

```bash
cd project

uv run pytest tests
```

---

## 8. Демонстрация на защите

На защите планирую:

1. Кратко показать структуру проекта:
  - `src/`
  - `notebooks/`
  - `tests/`
  - `artifacts/`

2. Показать data pipeline:
  - `download`;
  - `organize`;
  - `prepare`;
  - файл `data/processed/detection/annotations.csv`.

3. Показать EDA, baseline и основные эксперименты:
  - EDA организованных данных;
  - baseline notebook;
  - основной detection experiment notebook;
  - выбор финальной модели по `mAP@0.5`.

4. Проверить наличие артефактов модели:

```bash
cd project

uv run python -m src.cli check-artifacts
```

5. Запустить API сервис через:

```bash
cd project

uv run uvicorn src.api:app --reload
```

6. Продемонстрировать:
- `/health`
- `/info`
- `/predict`
- `/visualize-detections`

7. Показать файл логов после выполнения запросов:

```text
logs/app.log
```

В логах можно увидеть:
- факт запуска сервиса;
- загрузку модели;
- обращения к `/health`, `/info`, `/predict`, `/visualize-detections`;
- время обработки запросов;
- количество найденных объектов.

8. Показать inference на реальном изображении и визуализацию bounding boxes.

---

## 9. Ограничения и дальнейшая работа

В текущей версии:
- inference выполняется только для изображений;
- отсутствует batch/video inference;
- реализовано базовое логирование API через стандартный модуль `logging`;
- в inference-сервисе используется одна финальная detection-модель;
- повторное обучение реализовано как отдельный модуль, но не запускается автоматически из API.

В дальнейшем проект можно расширить:
- добавить CLI-команду для безопасного запуска повторного обучения;
- добавить автоматическую загрузку весов модели из GitHub Releases;
- добавить batch inference;
- добавить video inference;
- добавить Docker deployment.

---

## 10. Оценка проекта

Итоговая оценка за проект выставляется по пятибалльной шкале (2-5).

Ориентиры для оценки:

- **2** – проект не принят:
  - не выполняются минимальные требования (сервис не запускается, отсутствует ключевой функционал);
  - грубые нарушения требований курса;
  - явный плагиат или формальная имитация работы.
- **3** – проект принят, но реализован на базовом уровне:
  - минимальный функционал есть;
  - по чеклисту `self-checklist.md` выполнено **меньше 5** пунктов;
  - эксперименты и документация слабо проработаны.
- **4** – хороший, рабочий проект:
  - сервис запускается по `README.md`, `/predict` использует реальную модель;
  - есть данные, EDA и эксперименты с метриками;
  - структура кода и конфигураций в целом адекватна;
  - по чеклисту выполнено **не менее 5** пунктов.
- **5** – сильный, хорошо проработанный проект:
  - аккуратно реализован сервис и пайплайн;
  - проведены осмысленные эксперименты и обоснован выбор финальной модели;
  - есть базовая наблюдаемость и работа с конфигами/секретами;
  - документация позволяет быстро понять и воспроизвести решение;
  - по чеклисту выполнено **не менее 9** пунктов.

Чеклист `self-checklist.md` служит для самопроверки студента и как подсказка при проверке.  
Окончательное решение по оценке остаётся за преподавателем и может учитывать:

- качество реализации внутри каждого пункта чеклиста;
- дополнительные сильные стороны проекта (нестандартные решения, дополнительные функции, продвинутая ML-часть и т.п.);
- соблюдение требований курса и дедлайнов.

---
