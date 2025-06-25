import pandas as pd
import re
import Levenshtein as lev
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import sympy as sp
from sympy.parsing.latex import parse_latex
import warnings

# Конфигурация
INPUT_CSV = "model_answers.csv"
OUTPUT_CSV = "metrics_results.csv"

# Игнорирование предупреждений
warnings.filterwarnings("ignore", module="sympy")

def normalize_math(text):
    """Нормализация математических выражений"""
    if not text:
        return ""
    
    # Удаление пробелов и специальных символов
    text = re.sub(r'\s+', '', text)
    text = re.sub(r'\\left|\\right|\\,|\\:|\\;|\\!', '', text)
    
    # Стандартизация функций
    replacements = {
        '\\dfrac': '\\frac',
        '\\partial': 'd',
        '{\\partial}': 'd',
        '{}': '',
        '^{\\prime}': "'",
        "''": "''",
        '\\text{d}': 'd',
        '\\mathrm{d}': 'd',
        '{\\Delta}': '\\Delta',
        '\u2009': '',
        '\u2212': '-',
        '\\exp': 'e^{',
        '\\operatorname': '',
        '\\cdot': '',
        '\\times': '',
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    # Стандартизация констант
    text = re.sub(r'C[_{]?\d?}', 'C', text)
    text = re.sub(r'\\mathrm\s*{[cC]}', 'C', text)
    text = re.sub(r'K[_{]?\d?}', 'C', text)
    
    return text.strip()

def check_symbolic_equivalence(true_expr, pred_expr):
    """Проверка математической эквивалентности"""
    try:
        expr_true = sp.simplify(parse_latex(true_expr))
        expr_pred = sp.simplify(parse_latex(pred_expr))
        
        if expr_true == expr_pred:
            return 1.0
        
        diff = sp.simplify(expr_true - expr_pred)
        if diff.is_constant():
            return 0.9
            
        if sp.simplify(expr_true - expr_pred).equals(0):
            return 1.0
            
        return 0.0
    
    except Exception:
        return 0.0

def calculate_metrics(true_norm, pred_norm):
    """Вычисление метрик качества"""
    symbolic_score = check_symbolic_equivalence(true_norm, pred_norm)
    em = int(true_norm == pred_norm)
    
    # BLEU Score
    smoothie = SmoothingFunction().method4
    try:
        bleu = sentence_bleu(
            [list(true_norm)],
            list(pred_norm),
            smoothing_function=smoothie,
            weights=(0.5, 0.5))
    except:
        bleu = 0.0
    
    # Levenshtein Similarity
    distance = lev.distance(true_norm, pred_norm)
    max_len = max(len(true_norm), len(pred_norm))
    lev_sim = 1 - (distance / max_len) if max_len > 0 else 1.0
    
    return {
        'symbolic_score': symbolic_score,
        'exact_match': em,
        'bleu_score': bleu,
        'levenshtein_similarity': lev_sim,
        'levenshtein_distance': distance
    }

def main():
    """Основная функция расчета метрик"""
    print(f"Calculating metrics for {INPUT_CSV}...")
    
    # Загрузка данных
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} rows")
    
    metrics_results = []
    
    for idx, row in df.iterrows():
        true_ans = row['true_answer']
        model_solution = row['model_solution']
        
        # Пропуск ошибок
        if "[ERROR]" in str(model_solution) or not model_solution:
            metrics_results.append({
                **row.to_dict(),
                "true_normalized": "",
                "pred_normalized": "",
                "symbolic_score": 0.0,
                "exact_match": 0,
                "bleu_score": 0.0,
                "levenshtein_similarity": 0.0,
                "levenshtein_distance": 100
            })
            continue
        
        try:
            true_norm = normalize_math(true_ans)
            pred_norm = normalize_math(model_solution)
            
            metrics = calculate_metrics(true_norm, pred_norm)
            
            metrics_results.append({
                **row.to_dict(),
                "true_normalized": true_norm,
                "pred_normalized": pred_norm,
                **metrics
            })
            
        except Exception as e:
            print(f"Error processing row {idx}: {str(e)}")
            metrics_results.append({
                **row.to_dict(),
                "true_normalized": "",
                "pred_normalized": "",
                "symbolic_score": 0.0,
                "exact_match": 0,
                "bleu_score": 0.0,
                "levenshtein_similarity": 0.0,
                "levenshtein_distance": 100
            })
    
    # Сохранение результатов
    print(f"Saving results to {OUTPUT_CSV}...")
    metrics_df = pd.DataFrame(metrics_results)
    metrics_df.to_csv(OUTPUT_CSV, index=False)
    
    # Анализ результатов
    valid_results = [r for r in metrics_results if "ERROR" not in r.get('model_reasoning', '') and r.get('model_solution')]
    
    if valid_results:
        symbolic_matches = sum(1 for r in valid_results if r['symbolic_score'] >= 0.9)
        exact_matches = sum(1 for r in valid_results if r['exact_match'] == 1)
        total_valid = len(valid_results)
        
        summary = {
            "total_equations": len(df),
            "valid_responses": total_valid,
            "symbolic_match": symbolic_matches,
            "symbolic_accuracy": f"{symbolic_matches/total_valid:.2%}",
            "exact_match": exact_matches,
            "exact_accuracy": f"{exact_matches/total_valid:.2%}",
            "avg_bleu_score": sum(r['bleu_score'] for r in valid_results) / total_valid,
            "avg_levenshtein_similarity": sum(r['levenshtein_similarity'] for r in valid_results) / total_valid,
        }
        
        print("\nSummary statistics:")
        for k, v in summary.items():
            print(f"{k}: {v}")
    else:
        print("No valid results for statistics")
    
    print("✅ Metrics calculation completed!")

if __name__ == "__main__":
    main()
