import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline
from peft import PeftModel

MODEL_NAME = "Qwen/Qwen2.5-Math-1.5B"
ADAPTER_PATH = "./qwen_math_finetuned"

def generate(prompt: str):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        trust_remote_code=True,
        device_map="auto"
    )
    
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH, device_map="auto")
    
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        device_map="auto",
        max_new_tokens=150,
        temperature=0.2
    )
    
    return pipe(prompt)[0]['generated_text']

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
        result = generate(prompt)
        print(result)
    else:
        print("Usage: python use_quantized_model.py <your_prompt>")
