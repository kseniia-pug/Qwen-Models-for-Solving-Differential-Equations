import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    pipeline,
    set_seed
)
from peft import LoraConfig, get_peft_model, PeftModel
import gc
import os
import Levenshtein as lev
import time
from tqdm import tqdm

# Конфигурация
MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B"
DATASET_PATH = "english_diffeq_dataset.csv"
OUTPUT_DIR = "./qwen_math_finetuned"
LOSS_PLOT_PATH = "training_loss.png"
TEST_OUTPUT_CSV = "test_results.csv"

# Параметры модели
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 8
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
WARMUP_RATIO = 0.05
MAX_SEQ_LENGTH = 512  
EVAL_STEPS = 100 

MAX_NEW_TOKENS = 256
INFERENCE_TEMPERATURE = 0.1

MATH_PROMPT = """Solve the differential equation. Provide the final solution in LaTeX format enclosed in \\boxed{{}}.
Equation: {equation}
Reason step by step, then box the final answer.

Solution:"""

def normalize_math(text):
    """Normalize mathematical expressions"""
    if not text or not isinstance(text, str):
        return ""
    
    text = re.sub(r'\s+|\\left|\\right|\\,|\\:|\\;|\\!|\\text|\\mathrm', '', text)
    replacements = {
        '\\dfrac': '\\frac', '\\partial': 'd', '{\\partial}': 'd', '{}': '',
        '^{\\prime}': "'", "''": "''", '{\\Delta}': '\\Delta', '\u2009': '',
        '\u2212': '-', '\\exp': 'e^{', '\\cdot': '', '\\times': ''
    }
    
    for old, new in replacements.items():
        text = text.replace(old, new)
    
    text = re.sub(r'C[_{]?\d?}', 'C', text)
    text = re.sub(r'K[_{]?\d?}', 'C', text)
    
    return text.strip()

def extract_solution(text):
    """Extract solution from generated text"""
    if '\\boxed{' in text:
        start = text.find('\\boxed{') + 7
        depth = 1
        result = []
        for char in text[start:]:
            if char == '{':
                depth += 1
            elif char == '}':
                depth -= 1
                if depth == 0:
                    break
            if depth > 0:
                result.append(char)
        return ''.join(result).strip()
    
    patterns = [
        r'\\boxed\{(.*?)\}',
        r'Solution:\s*(.*?)$',
        r'Answer:\s*(.*?)$'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
    
    return text[:200].strip()

def calculate_metrics(true_norm, pred_norm):
    """Calculate evaluation metrics"""
    symbolic_score = 1.0 if true_norm == pred_norm else 0.0
    em = int(true_norm == pred_norm)
    distance = lev.distance(true_norm, pred_norm)
    max_len = max(len(true_norm), len(pred_norm), 1)
    lev_sim = 1 - (distance / max_len)
    
    return {
        'symbolic_score': symbolic_score,
        'exact_match': em,
        'levenshtein_similarity': lev_sim,
        'levenshtein_distance': distance
    }

def load_dataset(path):
    """Load and preprocess dataset"""
    df = pd.read_csv(path)
    df = df.dropna(subset=["equation", "answer"])
    
    df["text"] = df.apply(
        lambda row: MATH_PROMPT.format(equation=row["equation"]) + " " + row["answer"], 
        axis=1
    )
    return df

def tokenize_dataset(dataset, tokenizer):
    """Tokenize dataset efficiently"""
    return dataset.map(
        lambda examples: tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding=False
        ),
        batched=True,
        remove_columns=dataset.column_names,
        num_proc=4
    )

def plot_loss_curves(history, output_dir):
    """Plot training and validation loss curves"""
    train_losses = [log["loss"] for log in history if "loss" in log and "eval_loss" not in log]
    train_steps = [log["step"] for log in history if "loss" in log and "eval_loss" not in log]
    
    plt.figure(figsize=(12, 6))
    plt.plot(train_steps, train_losses, 'b-o', label='Training Loss')
    
    eval_losses = [log["eval_loss"] for log in history if "eval_loss" in log]
    eval_steps = [log["step"] for log in history if "eval_loss" in log]
    
    if eval_losses:
        plt.plot(eval_steps, eval_losses, 'r-o', label='Validation Loss')
    
    plt.title("Training & Validation Loss")
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_dir, "loss_curves.png"))

def main():
    """Main training and evaluation function"""
    set_seed(42)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    start_time = time.time()
    
    print(f"Fine-tuning {MODEL_NAME} for math problem solving")
    
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        device_map="auto"
    )
    
    # Load and prepare dataset
    df = load_dataset(DATASET_PATH)
    train_df, test_df = train_test_split(df, test_size=0.1, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42)
    
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    
    print(f"Train size: {len(train_dataset)}")
    print(f"Validation size: {len(val_dataset)}")
    print(f"Test size: {len(test_df)}")
    
    # Tokenize datasets
    tokenized_train = tokenize_dataset(train_dataset, tokenizer)
    tokenized_val = tokenize_dataset(val_dataset, tokenizer)
    
    # Configure LoRA
    peft_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, peft_config)
    model.print_trainable_parameters()
    
    # Configure training
    total_steps = (len(tokenized_train) * NUM_EPOCHS) / (BATCH_SIZE * GRAD_ACCUM_STEPS)
    warmup_steps = int(total_steps * WARMUP_RATIO)
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_steps=warmup_steps,
        num_train_epochs=NUM_EPOCHS,
        logging_steps=50,
        evaluation_strategy="steps",
        eval_steps=EVAL_STEPS,
        save_strategy="steps",
        save_steps=EVAL_STEPS,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        bf16=torch.cuda.is_bf16_supported(),
        remove_unused_columns=True,
        gradient_checkpointing=True,
        save_total_limit=2,
        optim="adamw_torch",
    )
    
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val,
        data_collator=data_collator,
    )

    # Start training
    print("Starting training...")
    train_result = trainer.train()
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    # Save metrics and plots
    print(f"Training completed in {train_result.metrics['train_runtime']:.2f} seconds")
    plot_loss_curves(trainer.state.log_history, OUTPUT_DIR)
    
    # Evaluate model
    print("Testing model...")
    model = PeftModel.from_pretrained(model, OUTPUT_DIR).merge_and_unload()
    
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device=model.device,
        max_new_tokens=MAX_NEW_TOKENS
    )
    
    # Generate predictions
    results = []
    for i, row in tqdm(test_df.iterrows(), total=len(test_df)):
        prompt = MATH_PROMPT.format(equation=row["equation"])
        
        try:
            output = pipe(prompt)[0]["generated_text"]
            solution = extract_solution(output)
            
            true_norm = normalize_math(row["answer"])
            pred_norm = normalize_math(solution)
            
            metrics = calculate_metrics(true_norm, pred_norm)
            
            results.append({
                "equation": row["equation"],
                "true_answer": row["answer"],
                "predicted_answer": solution,
                "output": output,
                **metrics
            })
        except Exception as e:
            print(f"Error processing sample {i}: {e}")

    # Save results
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(OUTPUT_DIR, TEST_OUTPUT_CSV), index=False)
    
    # Calculate final metrics
    symbolic_acc = results_df["symbolic_score"].mean()
    exact_match = results_df["exact_match"].mean()
    lev_sim = results_df["levenshtein_similarity"].mean()
    
    print("\nFinal metrics:")
    print(f"Symbolic Accuracy: {symbolic_acc:.4f}")
    print(f"Exact Match: {exact_match:.4f}")
    print(f"Levenshtein Similarity: {lev_sim:.4f}")
    
    print(f"\nAll results saved to {OUTPUT_DIR}")
    print(f"Total execution time: {(time.time() - start_time)/60:.2f} minutes")

if __name__ == "__main__":
    main()
