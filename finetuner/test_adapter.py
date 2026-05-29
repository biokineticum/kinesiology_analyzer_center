import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base_model_id = "unsloth/Llama-3.2-3B-Instruct"
adapter_id = "./outputs/stable_lora"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model_id)

print("Loading base model in 4-bit...")
from transformers import BitsAndBytesConfig
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.float16
)
model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    device_map="auto"
)

# 1. Test Base Model on "hi"
print("\n--- TEST 1: Base Model (WITHOUT adapter) on 'hi' ---")
messages = [{"role": "user", "content": "hi"}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs = model.generate(
        **inputs, 
        max_new_tokens=40, 
        temperature=0.1, 
        do_sample=True, 
        eos_token_id=tokenizer.eos_token_id
    )
response_base = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print("Base model response:", response_base.strip())

# Load adapter
print("\nLoading adapter...")
peft_model = PeftModel.from_pretrained(model, adapter_id)

# 2. Test Fine-Tuned Model on "hi"
print("\n--- TEST 2: Fine-Tuned Model (WITH adapter) on 'hi' ---")
with torch.no_grad():
    outputs_ft = peft_model.generate(
        **inputs, 
        max_new_tokens=40, 
        temperature=0.1, 
        do_sample=True, 
        eos_token_id=tokenizer.eos_token_id
    )
response_ft = tokenizer.decode(outputs_ft[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print("Fine-tuned model response (on 'hi'):", response_ft.strip())

# 3. Test Fine-Tuned Model on a dataset question (WITHOUT CONTEXT)
print("\n--- TEST 3: Fine-Tuned Model on scientific prompt (WITHOUT context) ---")
scientific_prompt = "What is the average body mass of the participants?"
print(f"Asking: '{scientific_prompt}'")
messages_sci = [{"role": "user", "content": scientific_prompt}]
prompt_sci = tokenizer.apply_chat_template(messages_sci, tokenize=False, add_generation_prompt=True)
inputs_sci = tokenizer(prompt_sci, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs_sci = peft_model.generate(
        **inputs_sci, 
        max_new_tokens=60, 
        temperature=0.1, 
        do_sample=True, 
        eos_token_id=tokenizer.eos_token_id
    )
response_sci = tokenizer.decode(outputs_sci[0][inputs_sci.input_ids.shape[1]:], skip_special_tokens=True)
print("Fine-tuned model response (no context):", response_sci.strip())

# 4. Test Fine-Tuned Model on a dataset question (WITH CONTEXT)
print("\n--- TEST 4: Fine-Tuned Model on scientific prompt (WITH context) ---")
scientific_prompt_with_context = "What is the average body mass of the participants?\n\nContext:\nbody mass: 73.76 ± 10.16 kg"
print(f"Asking: '{scientific_prompt_with_context}'")
messages_sci_ctx = [{"role": "user", "content": scientific_prompt_with_context}]
prompt_sci_ctx = tokenizer.apply_chat_template(messages_sci_ctx, tokenize=False, add_generation_prompt=True)
inputs_sci_ctx = tokenizer(prompt_sci_ctx, return_tensors="pt").to("cuda")

with torch.no_grad():
    outputs_sci_ctx = peft_model.generate(
        **inputs_sci_ctx, 
        max_new_tokens=60, 
        temperature=0.1, 
        do_sample=True, 
        eos_token_id=tokenizer.eos_token_id
    )
response_sci_ctx = tokenizer.decode(outputs_sci_ctx[0][inputs_sci_ctx.input_ids.shape[1]:], skip_special_tokens=True)
print("Fine-tuned model response (with context):", response_sci_ctx.strip())
print("----------------------------------------------------------------")
