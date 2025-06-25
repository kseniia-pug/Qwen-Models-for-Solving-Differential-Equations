# Fine-tuned Qwen Math Solver

Исследование влияния дообучения и квантизации модели Qwen на решение дифференциальных уравнений

## Структура проекта

.
├── data/ # Датасеты для обучения и оценки
│ ├── data_from_latex.csv
│ ├── english_diffeq_dataset_big.csv.zip # Большой датасет дифференциальных уравнений
│ └── test_data.csv
├── models/ # Веса моделей
│ └── qwen_math_finetuned/ # Веса дообученной модели
├── scripts/ # Рабочие скрипты
│ ├── generate_data_from_latex.py # Генерация датасета
│ ├── Qwen_tuning.py # Дообучение модели
│ ├── quantize.py # Квантизация модели
│ ├── quantize_and_test.py # Квантизация и тестирование
│ └── use_quantized_model.py # Использование модели
├── examples/ # Примеры использования
│ └── using_qwen_for_eq_solution.py
├── metrics/ # Оценка качества
│ └── get_metrics_from_model_solution.py
└── README.md # Документация
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
