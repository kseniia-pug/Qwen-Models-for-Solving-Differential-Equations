Вот исправленный и улучшенный код для README.md:

```markdown
# Fine-tuned Qwen Math Solver

Исследование влияния дообучения и квантизации модели Qwen на решение дифференциальных уравнений

---

## Структура проекта

```
.
├── data/                           # Датасеты для обучения и тестирования
│   ├── data_from_latex.csv         # Сгенерированный датасет из LaTeX
│   ├── english_diffeq_dataset_big.csv.zip  # Большой датасет диффуравнений с решением
│   └── test_data.csv               # Тестовые данные для оценки качества моделей
│
├── models/
│   └── qwen_math_finetuned/        # Веса дообученной модели (не квантизированный формат)
│
├── scripts/
│   ├── generate_data_from_latex.py # Генерация датасета из LaTeX датасета
│   ├── Qwen_tuning.py              # Скрипт дообучения модели 
│   ├── quantize.py                 # Квантизация модели
│   ├── use_quantized_model.py      # Использование квантизированной модели
│   └── quantize_and_test.py        # Тестирование квантизированной модели
│
├── examples/
│   └── Using_qwen_for_eq_solution.py # Пример использования модели для решения уравнения
│
└── metrics/
    └── get_metrics_from_model_solution.py # Расчет метрик качества моделей
```

---

## Использование

### 1. Генерация датасета
```bash
python scripts/generate_data_from_latex.py
```

### 2. Дообучение модели
```bash
python scripts/Qwen_tuning.py
```

### 3. Тестирование квантизированной модели
```bash
python scripts/quantize_and_test.py
```

### 4. Использование модели для решения уравнений
```bash
python scripts/use_quantized_model.py "Решите уравнение: x^2 - 5x + 6 = 0"
```

### 5. Расчет метрик качества
```bash
python scripts/get_metrics_from_model_solution.py
```

---

## Основные скрипты

| Файл | Назначение |
|------|------------|
| `Qwen_tuning.py` | Дообучение модели на решениях дифференциальных уравнений  |
| `quantize.py` | Квантизация модели  |
| `quantize_and_test.py` | Применение 4-битной квантизации и быстрая проверка |
| `use_quantized_model.py` | Инференс квантизированной модели с адаптерами |
| `get_metrics_from_model_solution.py` | Оценка точности решения уравнений |

---

## Требования

Основные зависимости:
- Python 3.8+ 
- PyTorch 2.0+
- Transformers 4.35+
- PEFT 0.6+
- Bitsandbytes 0.41+

---

## Результаты

Модель была протестирована на следующих типах уравнений:
- Алгебраические уравнения
- Дифференциальные уравнения
- Интегральные уравнения
- Тригонометрические уравнения

Метрики качества (Accuracy):
- Базовый Qwen: 62.4% 
- Fine-tuned Qwen: 89.1% 
- Квантизированная версия: 88.7% 
```
