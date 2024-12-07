from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch
import pandas as pd
from datasets import load_dataset  
from metrics import metrics, perplexity, self_bleu, repetition, zipf_coefficient  
from strategies import strategies, get_decoding_functions 
from prompts import prompts

# Connect to Hugging Face
login()

# Instantiate Mistral 7B model
model_name = "mistralai/Mistral-7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# Configure bitsandbytes for 8-bit loading
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    bnb_8bit_use_double_quant=True,
    bnb_8bit_dtype=torch.float16,
)

# Load model 
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    offload_folder="offload",  
    torch_dtype=torch.float16
)

# Number of outputs per strategy per prompt
num_outputs = 2

# Generate outputs
all_outputs = {prompt: {strategy: [] for strategy in strategies} for prompt in prompts}
for i, prompt in enumerate(prompts, 1):
    print(f"Processing Prompt {i}/{len(prompts)}: {prompt}")
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to("cuda")
    decoding_functions = get_decoding_functions(inputs, model, tokenizer)  
    for j, (strategy, decode_func) in enumerate(zip(strategies, decoding_functions), 1):
        print(f"  Generating outputs for Strategy {j}/{len(strategies)}: {strategy}")
        for k in range(num_outputs):
            torch.cuda.empty_cache()  
            output = decode_func()
            all_outputs[prompt][strategy].append(tokenizer.decode(output[0], skip_special_tokens=True))
            print(f"    Generated Output {k+1}/{num_outputs} for Strategy: {strategy}")

# Combine outputs
combined_outputs = {strategy: [] for strategy in strategies}
for prompt_outputs in all_outputs.values():
    for strategy, generations in prompt_outputs.items():
        combined_outputs[strategy].extend(generations)

# Load prewritten dataset for perplexity
dataset = load_dataset("wikitext", "wikitext-103-v1")

# Extract 10K tokens from prewritten dataset
max_tokens = 10000
total_tokens = 0
prewritten_texts = []
for example in dataset['test']['text']:
    if example.strip():  
        tokens = tokenizer.tokenize(example)
        num_tokens = len(tokens)      
        if total_tokens + num_tokens > max_tokens:  
            break  
        prewritten_texts.append(example)  
        total_tokens += num_tokens  

# Add computed metrics to metrics dictionary
for strategy, generations in combined_outputs.items():
    print(f"Computing metrics for Strategy: {strategy}")
    metrics["Strategy"].append(strategy)
    metrics["Perplexity"].append(perplexity(prewritten_texts, model, tokenizer)) 
    metrics["Self-BLEU"].append(self_bleu(all_outputs))
    metrics["Repetition (%)"].append(repetition(generations, tokenizer))  
    metrics["Zipf Coefficient"].append(zipf_coefficient(generations, tokenizer))  

# Convert dictionary to dataframe and print
metrics_df = pd.DataFrame(metrics)
print(metrics_df)

# Save metrics to CSV file
metrics_df.to_csv("metrics.csv", index=False)
print("Metrics saved to 'metrics.csv'")