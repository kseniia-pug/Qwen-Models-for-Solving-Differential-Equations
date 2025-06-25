import torch
import pandas as pd
import re
import gc
from tqdm import tqdm
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Конфигурация
MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B"
DATASET_PATH = "test_data.csv"
OUTPUT_CSV = "model_answers_qwen_orig.csv"

# Промпт для модели
MATH_PROMPT = """Solve the following differential equation step by step. 
Provide the final solution in LaTeX format enclosed in \\boxed{{}}.

Equation: {equation}

Reasoning:
"""

def extract_equation(text: str) -> str:
    """Извлечение уравнения из ответа модели"""
    # Приоритет 1: Стандартный boxed
    boxed_match = re.search(r'\\boxed{(.*?)}', text, re.DOTALL)
    if boxed_match:
        return boxed_match.group(1).strip()
    
    # Приоритет 2: Упрощенные варианты boxed
    simple_boxed_match = re.search(r'\\boxed\s*(\S+)', text)
    if simple_boxed_match:
        return simple_boxed_match.group(1).strip()
    
    # Приоритет 3: LaTeX окружения
    latex_envs = [
        r'\\begin{equation\*?}(.*?)\\end{equation\*?}',
        r'\\begin{align\*?}(.*?)\\end{align\*?}',
        r'\\begin{gather\*?}(.*?)\\end{gather\*?}'
    ]
    for pattern in latex_envs:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            content = match.group(1).strip()
            if '\\\\' in content:
                content = content.split('\\\\')[-1]
            return content
    
    # Приоритет 4: Inline LaTeX
    latex_match = re.search(r'\$(.*?)\$', text)
    if latex_match:
        return latex_match.group(1).strip()
    
    # Приоритет 5: Ключевые фразы
    key_phrases = [
        "solution is", "answer is", "therefore", 
        "final solution", "result is", "="
    ]
    for phrase in key_phrases:
        idx = text.find(phrase)
        if idx != -1:
            result = text[idx + len(phrase):].strip()
            result = re.split(r'[\.\n]', result)[0]
            return result
    
    return text.strip()

def main():
    print(f"Loading model {MODEL_NAME}...")
    
    # Определение типа данных
    if torch.cuda.is_available():
        torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        torch_dtype = torch.float32

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch_dtype,
        trust_remote_code=True
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.eval()
    print("Model loaded successfully!")

    # Загрузка датасета
    print("Loading dataset...")
    df = pd.read_csv(DATASET_PATH)
    dataset = Dataset.from_pandas(df)
    print(f"Total equations: {len(dataset)}")

    def solve_equation(equation: str) -> tuple:
        prompt = MATH_PROMPT.format(equation=equation)
        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=2000,
                do_sample=True,
                temperature=0.3,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return prompt, response

    # Генерация ответов
    print("Generating answers...")
    results = []

    for idx, row in tqdm(enumerate(dataset), total=len(dataset), desc="Processing"):
        eq = row['equation']
        true_ans = row['answer']

        try:
            prompt, model_answer = solve_equation(eq)
            extracted_eq = extract_equation(model_answer)
            
            results.append({
                "equation": eq,
                "true_answer": true_ans,
                "model_reasoning": model_answer,
                "model_solution": extracted_eq
            })
            
        except Exception as e:
            results.append({
                "equation": eq,
                "true_answer": true_ans,
                "model_reasoning": f"[ERROR] {str(e)}",
                "model_solution": ""
            })
        
        if idx % 10 == 0:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Сохранение результатов
    print(f"Saving results to {OUTPUT_CSV}...")
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUTPUT_CSV, index=False)
    print(f"Done! Generated {len(results_df)} answers.")

if __name__ == "__main__":
    main()
