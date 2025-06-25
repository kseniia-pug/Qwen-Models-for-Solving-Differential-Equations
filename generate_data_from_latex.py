import os
import re
import random
import pandas as pd
import sympy as sp
from sympy import Function, Derivative, dsolve, symbols, Eq, exp, checkodesol
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

# Конфигурация
DATASET_PATH = "all.csv"
OUTPUT_PATH = "english_diffeq_dataset.csv"
NUM_NEW_SAMPLES = 50000
SAVE_INTERVAL = 500
MAX_RETRIES = 5

# Определение символьных переменных
x, t = symbols('x t')
y = Function('y')

# Кэш уравнений для ускорения генерации
SIMPLE_EQUATIONS = {
    "first_order": [Eq(Derivative(y(x), x), k*x) for k in range(1, 10)],
    "implicit": [Eq(y(x)*Derivative(y(x), x), k*x) for k in range(1, 5)],
    "laplace": [
        Eq(Derivative(y(t), t, 2) + a*Derivative(y(t), t) + b*y(t), c*exp(-d*t))
        for a in [2, 3, 4] for b in [15, 20, 25] for c in [10, 15, 20] for d in [1, 2]
    ]
}

def extract_boxed_solution(text):
    """Извлечение решения из LaTeX выражения в boxed"""
    match = re.search(r'\\boxed\{(.*?)\}', text, re.DOTALL)
    return match.group(1).strip() if match else text.strip()

def format_boxed_solution(solution):
    """Форматирование решения в boxed LaTeX"""
    return f"\\boxed{{{solution.strip()}}}"

def replace_constants(text):
    """Замена констант в уравнениях для генерации вариаций"""
    return re.sub(r'\d+', lambda m: str(max(1, int(m.group()) + random.randint(-1,1))), text)

def rephrase_english(text):
    """Перефразирование английского текста с использованием синонимов"""
    synonyms = {
        "solve": "find the solution to",
        "equation": "differential equation",
        "Solution": "Explanation",
        "integrate": "compute the integral",
        "derivative": "rate of change",
        "constant": "integration constant"
    }
    for word, synonym in synonyms.items():
        text = text.replace(word, synonym)
    return text

def validate_solution(eq, sol):
    """Проверка корректности решения дифференциального уравнения"""
    return checkodesol(eq, sol)[0]

def generate_laplace_equation():
    """Генерация уравнений, решаемых методом Лапласа"""
    for _ in range(MAX_RETRIES):
        try:
            eq = random.choice(SIMPLE_EQUATIONS["laplace"])
            y0 = random.randint(-3, 3)
            yp0 = random.randint(-5, 5)
            sol_expr = dsolve(eq, y(t), ics={y(t).subs(t,0): y0, Derivative(y(t),t).subs(t,0): yp0})
            
            if validate_solution(eq, sol_expr):
                eq_str = f"${sp.latex(eq)}$ with $y(0) = {y0}$, $y'(0) = {yp0}$"
                return {
                    'equation': eq_str,
                    'solution': "Apply Laplace transform to solve the equation",
                    'answer': format_boxed_solution(sp.latex(sol_expr.rhs))
                }
        except:
            continue
    return None

def generate_implicit_ode():
    """Генерация неявных дифференциальных уравнений"""
    eq = random.choice(SIMPLE_EQUATIONS["implicit"])
    sol = dsolve(eq, y(x))
    return {
        'equation': f"${sp.latex(eq)}$",
        'solution': "Use implicit differentiation",
        'answer': format_boxed_solution(sp.latex(sol.rhs))
    }

def generate_first_order():
    """Генерация уравнений первого порядка"""
    eq = random.choice(SIMPLE_EQUATIONS["first_order"])
    sol = dsolve(eq, y(x))
    return {
        'equation': f"${sp.latex(eq)}$",
        'solution': "Integrate both sides directly",
        'answer': format_boxed_solution(sp.latex(sol.rhs))
    }

def generate_series_solution():
    """Генерация уравнений, решаемых методом рядов"""
    return generate_first_order()

def generate_new_equation(chapter):
    """Выбор генератора по типу уравнения"""
    if "Laplace" in chapter:
        return generate_laplace_equation()
    elif "First" in chapter and "implicit" in chapter:
        return generate_implicit_ode()
    elif "First" in chapter:
        return generate_first_order()
    elif "Series" in chapter:
        return generate_series_solution()
    return generate_first_order()

def generate_sample(args):
    """Генерация одного примера уравнения"""
    try:
        row = df.sample(1).iloc[0]
        chapter = row['chapter']
        
        if "Laplace" in chapter:
            return generate_new_equation(chapter)
        
        equation = replace_constants(row['equation'])
        solution = rephrase_english(row['solution'])
        answer = format_boxed_solution(extract_boxed_solution(row['answer']))
        
        return {
            'chapter': chapter,
            'equation': equation,
            'solution': solution,
            'answer': answer
        }
    except Exception as e:
        return None

def main():
    """Основная функция генерации датасета"""
    # Загрузка существующего датасета
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Файл {DATASET_PATH} не найден")
    
    global df
    df = pd.read_csv(DATASET_PATH, delimiter='@')
    print(f"Загружен датасет с {len(df)} примерами")

    # Проверка существующих результатов
    if os.path.exists(OUTPUT_PATH):
        existing_df = pd.read_csv(OUTPUT_PATH)
        results = existing_df.to_dict('records')
        print(f"Продолжение генерации с {len(results)} существующими примерами")
    else:
        results = []
        print("Начало генерации с нуля")

    # Расчет необходимого количества новых примеров
    total_needed = NUM_NEW_SAMPLES - len(results)
    if total_needed <= 0:
        print("Достигнуто необходимое количество примеров!")
        return
    
    print(f"Генерация {total_needed} новых дифференциальных уравнений...")
    
    # Многопоточная генерация
    with Pool(processes=cpu_count()) as pool:
        for result in tqdm(pool.imap_unordered(generate_sample, 
                                             range(total_needed), 
                                             total=total_needed):
            if result:
                results.append(result)
                
                # Периодическое сохранение
                if len(results) % SAVE_INTERVAL == 0:
                    pd.DataFrame(results).to_csv(OUTPUT_PATH, index=False)
    
    # Финальное сохранение
    pd.DataFrame(results).to_csv(OUTPUT_PATH, index=False)
    print(f"Генерация завершена! Создано {len(results)} дифференциальных уравнений")

if __name__ == "__main__":
    main()
