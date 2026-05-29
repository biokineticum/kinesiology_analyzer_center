import os
import glob
from pathlib import Path
import json
import streamlit as st
import subprocess

# Force Unsloth to use the local self-contained convert_hf_to_gguf.py to bypass the new llama.cpp 'conversion' import bug
local_llama_path = Path("~/.unsloth/llama.cpp").expanduser().resolve()
if (local_llama_path / "convert_hf_to_gguf.py").exists():
    os.environ["UNSLOTH_LLAMA_CPP_SCRIPTS_DIR"] = str(local_llama_path)

# Monkeypatch Triton 3.6+ on Windows to prevent PyTorch Inductor triton_key import errors, ASTSource key assertions, and signature type errors
try:
    import triton.compiler.compiler as tcc
    if not hasattr(tcc, 'triton_key'):
        tcc.triton_key = lambda: "triton_3.6.0_windows_patch"
    if hasattr(tcc, 'ASTSource') and not getattr(tcc.ASTSource.__init__, '_is_patched', False):
        original_ast_init = tcc.ASTSource.__init__
        def patched_ast_init(self, fn, signature, constexprs=None, attrs=None):
            # 1. Transform signature keys to strings if they are integers (compatibility with PyTorch Inductor)
            if signature is not None:
                new_signature = {}
                for k, v in signature.items():
                    new_k = k
                    if isinstance(k, int) and hasattr(fn, 'arg_names') and k < len(fn.arg_names):
                        new_k = fn.arg_names[k]
                    elif not isinstance(k, str):
                        new_k = str(k)
                    new_signature[new_k] = v
                signature = new_signature
                
            # 2. Transform constexprs keys to tuples
            if constexprs is not None:
                new_constexprs = {}
                for k, v in constexprs.items():
                    new_k = k
                    if isinstance(k, str):
                        try:
                            new_k = (fn.arg_names.index(k),)
                        except ValueError:
                            pass
                    elif not isinstance(k, tuple):
                        new_k = (k,)
                    new_constexprs[new_k] = v
                constexprs = new_constexprs
                
            original_ast_init(self, fn, signature, constexprs, attrs)
        patched_ast_init._is_patched = True
        tcc.ASTSource.__init__ = patched_ast_init
except Exception:
    pass

# We wrap the Heavy imports in a try-except block so the Streamlit UI can still load 
# and show an error message if Unsloth is not installed properly.
try:
    from datasets import load_dataset, Dataset
    from unsloth import FastLanguageModel
    from unsloth import is_bfloat16_supported
    from transformers import TrainerCallback
    from trl import SFTTrainer
    from transformers import TrainingArguments
    UNSLOTH_AVAILABLE = True
except Exception as e:
    UNSLOTH_AVAILABLE = False
    UNSLOTH_ERROR = str(e)

st.set_page_config(page_title="🧠 Unsloth Fine-Tuner", layout="wide")

st.title("🧠 Ollama Unsloth Fine-Tuner")
st.caption("Fine-tune Gemma 2B models using your generated JSONL datasets.")

FINETUNER_DIR = Path(__file__).parent.resolve()

def get_datasets():
    return glob.glob(str(FINETUNER_DIR / "*.jsonl"))

datasets = get_datasets()

if not UNSLOTH_AVAILABLE:
    st.error(f"Unsloth or its dependencies are not installed correctly.\nError: {UNSLOTH_ERROR}")
    st.info("Please ensure you have installed Unsloth on Windows following the official installation guide.")
    st.stop()

def get_ollama_models_and_mappings():
    installed = []
    try:
        import subprocess
        result = subprocess.run(["ollama", "list"], capture_output=True, text=True, encoding="utf-8")
        if result.returncode == 0:
            lines = result.stdout.strip().split("\n")[1:]
            for line in lines:
                parts = line.split()
                if parts:
                    installed.append(parts[0])
    except Exception:
        pass

    mappings = {}
    for name in installed:
        lower_name = name.lower()
        if "gemma" in lower_name:
            if "9b" in lower_name:
                mappings[name] = "unsloth/gemma-2-9b-it"
            else:
                mappings[name] = "unsloth/gemma-2-2b-it"
        elif "llama" in lower_name:
            if "1b" in lower_name:
                mappings[name] = "unsloth/Llama-3.2-1B-Instruct"
            elif "3b" in lower_name:
                mappings[name] = "unsloth/Llama-3.2-3B-Instruct"
            elif "8b" in lower_name:
                mappings[name] = "unsloth/llama-3-8b-Instruct"
            else:
                mappings[name] = "unsloth/Llama-3.2-3B-Instruct"
        elif "qwen" in lower_name:
            if "1.5b" in lower_name:
                mappings[name] = "unsloth/Qwen2.5-1.5B-Instruct"
            elif "coder" in lower_name:
                mappings[name] = "unsloth/Qwen2.5-Coder-1.5B-Instruct"
            else:
                mappings[name] = "unsloth/Qwen2.5-7B-Instruct"
        else:
            mappings[name] = "unsloth/gemma-2-2b-it"
    return installed, mappings

installed_ollama, mappings = get_ollama_models_and_mappings()

with st.sidebar:
    st.header("⚙️ Configuration")
    
    if not datasets:
        st.warning("No .jsonl datasets found in the finetuner directory.")
        selected_datasets = []
    else:
        dataset_names = [Path(f).name for f in datasets]
        selected_dataset_names = st.multiselect("Select Datasets to Stack", dataset_names, default=dataset_names)
        selected_datasets = [FINETUNER_DIR / name for name in selected_dataset_names]
        
    st.divider()
    st.subheader("🤖 Base Model Selection")
    
    source_mode = st.radio(
        "Choose Base Model Source:",
        options=["Choose from local Ollama models", "Choose from recommended HF models", "Custom HuggingFace Model"],
        index=0 if installed_ollama else 1
    )
    
    if source_mode == "Choose from local Ollama models" and installed_ollama:
        selected_ollama = st.selectbox("Select Local Ollama Model", installed_ollama)
        base_model_name = mappings.get(selected_ollama, "unsloth/gemma-2-2b-it")
        st.info(f"Mapped to Hugging Face Model for Unsloth: `{base_model_name}`")
    elif source_mode == "Choose from recommended HF models":
        recommended_models = {
            "Llama 3.2 3B Instruct (Recommended - Best balance)": "unsloth/Llama-3.2-3B-Instruct",
            "Gemma 2 2B Instruct (Fast, Light)": "unsloth/gemma-2-2b-it",
            "Llama 3.2 1B Instruct (Ultra Fast)": "unsloth/Llama-3.2-1B-Instruct",
            "Qwen 2.5 1.5B Instruct (Very fast & capable)": "unsloth/Qwen2.5-1.5B-Instruct",
            "Gemma 2 9B Instruct (High Quality, requires stronger GPU)": "unsloth/gemma-2-9b-it",
            "Llama 3 8B Instruct (Popular & solid)": "unsloth/llama-3-8b-Instruct"
        }
        selected_model_label = st.selectbox("Select Recommended Model", options=list(recommended_models.keys()))
        base_model_name = recommended_models[selected_model_label]
    else:
        base_model_name = st.text_input("HuggingFace Model Name (e.g. unsloth/mistral-7b-v0.3)", value="unsloth/gemma-2-2b-it")

    st.divider()
    st.subheader("🛡️ Training Engine")
    engine_mode = st.radio(
        "Select Training Engine:",
        options=["Unsloth (Fast, Triton)", "Stable HF PEFT (Bypass Windows Triton Bugs)"],
        index=1,
        help="If Unsloth outputs gibberish/Arabic or collapses, switch to 'Stable HF PEFT'. This uses standard PyTorch CUDA operations which are 100% mathematically stable on Windows."
    )

    st.divider()
    st.subheader("🧬 LoRA Adaption Settings")
    lora_r = st.slider("LoRA Rank (r)", min_value=8, max_value=128, value=8, step=8, help="Higher rank lets the model learn more complex details but uses more memory. 8 is recommended for small specialized datasets to prevent overfitting.")
    lora_alpha = st.slider("LoRA Alpha", min_value=8, max_value=128, value=8, step=8, help="Usually matches LoRA Rank or scales proportionally. 8 is recommended for small datasets.")

    st.divider()
    st.subheader("⚡ Hyperparameters")
    max_seq_length = st.number_input("Max Sequence Length", min_value=512, max_value=8192, value=2048, step=512)
    epochs = st.number_input("Epochs", min_value=1, max_value=10, value=1)
    batch_size = st.number_input("Batch Size", min_value=1, max_value=16, value=2)
    lr = st.number_input("Learning Rate", value=5e-6, format="%.7f", step=1e-6, help="Lower learning rate (like 0.0000050) keeps the model smart and general. Higher rates will overfit and cause brain-fry.")
    weight_decay = st.number_input("Weight Decay", value=0.01, format="%.4f")
    warmup_steps = st.number_input("Warmup Steps", min_value=0, value=5)
    
    st.divider()
    st.subheader("💾 Export Settings")
    ollama_model_name = st.text_input("New Ollama Model Name", value="biomechanizator_v3")
    quant_method = st.selectbox("Quantization Method", ["q4_k_m", "q5_k_m", "q8_0", "f16"], index=0, help="q4_k_m is recommended for most setups. f16 is unquantized.")

class CustomCompletionOnlyCollator:
    def __init__(self, response_template, tokenizer):
        self.response_template = response_template
        self.tokenizer = tokenizer
        self.response_token_ids = tokenizer.encode(response_template, add_special_tokens=False)

    def __call__(self, features):
        import torch
        batch = {}
        input_ids = [torch.tensor(f['input_ids']) for f in features]
        # Robust fallback for attention mask if stripped by trainer
        attention_mask = [torch.tensor(f.get('attention_mask', [1] * len(f['input_ids']))) for f in features]
        
        # Ensure pad_token is set
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        padded = self.tokenizer.pad(
            {'input_ids': input_ids, 'attention_mask': attention_mask},
            padding=True,
            return_tensors="pt"
        )
        batch['input_ids'] = padded['input_ids']
        batch['attention_mask'] = padded['attention_mask']
        
        # Build labels: default to -100 (masked)
        labels = batch['input_ids'].clone()
        labels.fill_(-100)
        
        for i in range(len(features)):
            seq = batch['input_ids'][i].tolist()
            response_idx = -1
            n = len(self.response_token_ids)
            for j in range(len(seq) - n + 1):
                if seq[j:j+n] == self.response_token_ids:
                    response_idx = j + n
                    break
            
            if response_idx != -1:
                # Unmask tokens from response template up to sequence end
                for k in range(response_idx, len(seq)):
                    if batch['attention_mask'][i][k] != 0:
                        labels[i][k] = batch['input_ids'][i][k]
                        
        batch['labels'] = labels
        return batch

class StreamlitProgressCallback(TrainerCallback):
    def __init__(self, progress_bar, status_text, chart_element):
        self.progress_bar = progress_bar
        self.status_text = status_text
        self.chart_element = chart_element
        self.loss_history = []
        self.max_steps = 0

    def on_train_begin(self, args, state, control, **kwargs):
        self.max_steps = state.max_steps

    def on_step_end(self, args, state, control, **kwargs):
        if self.max_steps > 0:
            progress = state.global_step / self.max_steps
            self.progress_bar.progress(min(progress, 1.0))
            
            loss_val = 'N/A'
            if state.log_history:
                # Look for the last loss log entry
                for log in reversed(state.log_history):
                    if 'loss' in log:
                        loss_val = log['loss']
                        self.loss_history.append({"step": state.global_step, "loss": loss_val})
                        try:
                            import pandas as pd
                            self.chart_element.line_chart(pd.DataFrame(self.loss_history).set_index("step"))
                        except Exception:
                            pass
                        break
                        
            self.status_text.text(f"Step {state.global_step} / {self.max_steps} | Loss: {loss_val}")

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = "<eos>" # Placeholder, will be replaced by model's actual eos token

def formatting_prompts_func(examples):
    instructions = examples["instruction"]
    inputs       = examples["input"]
    outputs      = examples["output"]
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
        text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
        texts.append(text)
    return { "text" : texts, }

if selected_datasets:
    st.subheader("📊 Wybrane Zbiory Danych (Stack)")
    
    total_records = 0
    for ds in selected_datasets:
        try:
            with open(ds, 'r', encoding='utf-8') as f:
                lines = [l.strip() for l in f if l.strip()]
                num_records = len(lines)
                total_records += num_records
                
                with st.expander(f"📄 {ds.name} ({num_records} rekordów)"):
                    preview = [json.loads(line) for line in lines[:3]]
                    st.json(preview)
        except Exception as e:
            st.error(f"Błąd odczytu {ds.name}: {e}")
            
    st.info(f"🧬 Łączna liczba rekordów do przetrenowania: **{total_records}**")

    if st.button("🚀 Start Fine-Tuning", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        st.subheader("📈 Wykres Straty (Training Loss)")
        chart_element = st.empty()
        
        try:
            # Clear VRAM before starting to ensure clean state on 6GB VRAM GPUs
            import gc
            import torch
            gc.collect()
            torch.cuda.empty_cache()

            status_text.text(f"Loading base model ({base_model_name})...")
            
            if engine_mode == "Stable HF PEFT (Bypass Windows Triton Bugs)":
                status_text.text(f"Loading base model ({base_model_name}) in stable HF 4-bit mode (Bypassing Windows Triton)...")
                from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
                from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
                import torch
                
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16 if is_bfloat16_supported() else torch.float16
                )
                
                model = AutoModelForCausalLM.from_pretrained(
                    base_model_name,
                    quantization_config=bnb_config,
                    device_map="auto"
                )
                tokenizer = AutoTokenizer.from_pretrained(base_model_name)
                tokenizer.pad_token = tokenizer.eos_token
                
                model = prepare_model_for_kbit_training(model)
                
                # Automatically detect correct chat template parts
                from unsloth.chat_templates import get_chat_template
                model_lower = base_model_name.lower()
                if "llama" in model_lower:
                    instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n"
                    response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n"
                    chat_template_name = "llama-3"
                elif "gemma" in model_lower:
                    instruction_part = "<start_of_turn>user\n"
                    response_part = "<start_of_turn>model\n"
                    chat_template_name = "gemma"
                elif "qwen" in model_lower:
                    instruction_part = "<|im_start|>user\n"
                    response_part = "<|im_start|>assistant\n"
                    chat_template_name = "qwen2.5"
                else:
                    instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n"
                    response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n"
                    chat_template_name = "llama-3"

                tokenizer = get_chat_template(
                    tokenizer,
                    chat_template = chat_template_name,
                )
                tokenizer.tokenizer_class = "PreTrainedTokenizerFast"
                EOS_TOKEN = tokenizer.eos_token
                
                peft_config = LoraConfig(
                    r=lora_r,
                    lora_alpha=lora_alpha,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                    lora_dropout=0.05,
                    bias="none",
                    task_type="CAUSAL_LM"
                )
                model = get_peft_model(model, peft_config)
                model.gradient_checkpointing_enable()
            else:
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name = base_model_name,
                    max_seq_length = max_seq_length,
                    dtype = None,
                    load_in_4bit = True,
                )
                
                # Automatically detect correct chat template parts for response masking and generation
                from unsloth.chat_templates import get_chat_template
                model_lower = base_model_name.lower()
                if "llama" in model_lower:
                    instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n"
                    response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n"
                    chat_template_name = "llama-3"
                elif "gemma" in model_lower:
                    instruction_part = "<start_of_turn>user\n"
                    response_part = "<start_of_turn>model\n"
                    chat_template_name = "gemma"
                elif "qwen" in model_lower:
                    instruction_part = "<|im_start|>user\n"
                    response_part = "<|im_start|>assistant\n"
                    chat_template_name = "qwen2.5"
                else:
                    instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n"
                    response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n"
                    chat_template_name = "llama-3"

                tokenizer = get_chat_template(
                    tokenizer,
                    chat_template = chat_template_name,
                )
                tokenizer.tokenizer_class = "PreTrainedTokenizerFast"
                EOS_TOKEN = tokenizer.eos_token
     
                status_text.text(f"Applying LoRA adapters (Rank: {lora_r}, Alpha: {lora_alpha})...")
                model = FastLanguageModel.get_peft_model(
                    model,
                    r = lora_r,
                    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                    lora_alpha = lora_alpha,
                    lora_dropout = 0,
                    bias = "none",
                    use_gradient_checkpointing = "unsloth",
                    random_state = 3407,
                    use_rslora = False,
                    loftq_config = None,
                )

            status_text.text("Sanitizing, loading, and mapping datasets...")
            data_list = []
            for file in selected_datasets:
                with open(file, 'r', encoding='utf-8') as f:
                    for line_num, line in enumerate(f):
                        if not line.strip(): continue
                        try:
                            item = json.loads(line)
                            
                            # Force all fields to be strings to prevent PyArrow schema crashes
                            instruction = str(item.get("instruction", ""))
                            input_text = str(item.get("input", ""))
                            
                            output_val = item.get("output", "")
                            if isinstance(output_val, list):
                                output_text = "\n".join(str(x) for x in output_val)
                            elif isinstance(output_val, dict):
                                output_text = json.dumps(output_val, ensure_ascii=False)
                            else:
                                output_text = str(output_val)
                                
                            data_list.append({
                                "instruction": instruction,
                                "input": input_text,
                                "output": output_text
                            })
                        except Exception as e:
                            st.warning(f"Skipped invalid JSON on line {line_num+1} in {file.name}: {e}")
                            
            if engine_mode == "Stable HF PEFT (Bypass Windows Triton Bugs)":
                def tokenize_function(example):
                    uc = example["instruction"]
                    if example["input"].strip():
                        uc += f"\n\nContext:\n{example['input']}"
                    messages = [
                        {"role": "user", "content": uc},
                        {"role": "assistant", "content": example["output"]}
                    ]
                    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
                    tokenized = tokenizer(text, truncation=True, max_length=max_seq_length)
                    return tokenized
                
                dataset = Dataset.from_list(data_list)
                dataset = dataset.map(tokenize_function, remove_columns=dataset.column_names)
                
                collator = CustomCompletionOnlyCollator(
                    response_template=response_part,
                    tokenizer=tokenizer
                )
                
                from trl import SFTConfig
                training_args = SFTConfig(
                    output_dir="outputs",
                    per_device_train_batch_size=batch_size,
                    gradient_accumulation_steps=4,
                    warmup_steps=warmup_steps,
                    num_train_epochs=epochs,
                    learning_rate=lr,
                    fp16=not is_bfloat16_supported(),
                    bf16=is_bfloat16_supported(),
                    logging_steps=1,
                    optim="paged_adamw_8bit",
                    weight_decay=weight_decay,
                    seed=3407,
                    max_grad_norm=1.0,
                )
                
                status_text.text("Configuring Stable HF Trainer...")
                trainer = SFTTrainer(
                    model=model,
                    processing_class=tokenizer,
                    train_dataset=dataset,
                    data_collator=collator,
                    args=training_args
                )
            else:
                # Define localized formatting function to correctly use the model's official chat template
                def formatting_prompts_func(examples):
                    instructions = examples["instruction"]
                    inputs       = examples["input"]
                    outputs      = examples["output"]
                    texts = []
                    for instruction, input_text, output in zip(instructions, inputs, outputs):
                        user_content = instruction
                        if input_text.strip():
                            user_content += f"\n\nContext:\n{input_text}"
                            
                        messages = [
                            {"role": "user", "content": user_content},
                            {"role": "assistant", "content": output}
                        ]
                        # Apply the tokenizer's official chat template dynamically
                        text = tokenizer.apply_chat_template(messages, tokenize = False, add_generation_prompt = False)
                        texts.append(text)
                    return { "text" : texts, }

                dataset = Dataset.from_list(data_list)
                dataset = dataset.map(formatting_prompts_func, batched = True)

                status_text.text("Configuring Unsloth Trainer...")
                
                # Calculate total steps for progress bar
                steps_per_epoch = len(dataset) // batch_size
                total_steps = steps_per_epoch * epochs

                trainer = SFTTrainer(
                    model = model,
                    tokenizer = tokenizer,
                    train_dataset = dataset,
                    dataset_text_field = "text",
                    max_seq_length = max_seq_length,
                    dataset_num_proc = 2,
                    packing = False, # Packing must be False for response masking
                    args = TrainingArguments(
                        per_device_train_batch_size = batch_size,
                        gradient_accumulation_steps = 4,
                        warmup_steps = warmup_steps,
                        num_train_epochs = epochs,
                        learning_rate = lr,
                        fp16 = not is_bfloat16_supported(),
                        bf16 = is_bfloat16_supported(),
                        logging_steps = 1,
                        optim = "adamw_8bit",
                        weight_decay = weight_decay,
                        lr_scheduler_type = "linear",
                        seed = 3407,
                        output_dir = "outputs",
                        max_grad_norm = 1.0, # Explicitly limit gradient norm to prevent weight explosion
                    ),
                )
                
                # Mask out the user instructions and PDF inputs, so the model ONLY trains on the target responses.
                # This protects the weights from collapsing/repeating when exposed to noisy scientific PDF pages.
                from unsloth.chat_templates import train_on_responses_only
                trainer = train_on_responses_only(
                    trainer,
                    instruction_part = instruction_part,
                    response_part = response_part,
                )
            
            # Add custom Streamlit callback with chart element
            trainer.add_callback(StreamlitProgressCallback(progress_bar, status_text, chart_element))
 
            status_text.text("Training started...")
            trainer_stats = trainer.train()
            
            status_text.text(f"Training completed! Exporting to GGUF with {quant_method} quantization (this will take a while)...")
            
            export_dir = FINETUNER_DIR / "exported_model"
            export_dir.mkdir(exist_ok=True)
            
            if engine_mode == "Stable HF PEFT (Bypass Windows Triton Bugs)":
                status_text.text("Saving trained PEFT adapter weights and tokenizer...")
                model.save_pretrained("outputs/stable_lora")
                tokenizer.save_pretrained("outputs/stable_lora")
                
                # Dynamic JSON patch to replace "TokenizersBackend" with "PreTrainedTokenizerFast"
                try:
                    import json
                    config_file = Path("outputs/stable_lora/tokenizer_config.json")
                    if config_file.exists():
                        with open(config_file, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        if data.get("tokenizer_class") == "TokenizersBackend":
                            data["tokenizer_class"] = "PreTrainedTokenizerFast"
                            with open(config_file, "w", encoding="utf-8") as f:
                                json.dump(data, f, indent=2, ensure_ascii=False)
                except Exception:
                    pass
                
                # Explicitly delete old model and trainer, and empty CUDA cache
                # to free up the 6GB VRAM before loading the model again in Unsloth.
                status_text.text("Freeing GPU memory before GGUF conversion...")
                import gc
                import torch
                del model
                if 'trainer' in locals():
                    del trainer
                gc.collect()
                torch.cuda.empty_cache()
                
                # Reload model and adapter in Unsloth for the 100% stable in-memory merge and GGUF export!
                status_text.text("Loading trained model in Unsloth for GGUF conversion...")
                model, tokenizer = FastLanguageModel.from_pretrained(
                    model_name = "outputs/stable_lora",
                    max_seq_length = max_seq_length,
                    dtype = None,
                    load_in_4bit = True,
                )
                tokenizer.tokenizer_class = "PreTrainedTokenizerFast"
            
            # Save the model to GGUF using selected quantization method
            tokenizer.tokenizer_class = "PreTrainedTokenizerFast"
            model.save_pretrained_gguf(export_dir, tokenizer, quantization_method = quant_method)
            
            status_text.text("Creating Ollama Modelfile...")
            
            # Unsloth sometimes appends _gguf to the directory or saves it with a different name
            gguf_files = list(FINETUNER_DIR.rglob("*.gguf"))
            if not gguf_files:
                st.error("Failed to find exported GGUF file.")
            else:
                # Get the most recently created gguf file just in case there are multiple
                gguf_path = sorted(gguf_files, key=lambda p: p.stat().st_mtime, reverse=True)[0]
                
                # Dynamically construct the template based on the base model family
                model_lower = base_model_name.lower()
                template_str = ""
                if "llama" in model_lower:
                    template_str = """TEMPLATE \"\"\"{{ if .System }}<|start_header_id|>system<|end_header_id|>

{{ .System }}<|eot_id|>{{ end }}{{ if .Prompt }}<|start_header_id|>user<|end_header_id|>

{{ .Prompt }}<|eot_id|>{{ end }}<|start_header_id|>assistant<|end_header_id|>

{{ .Response }}<|eot_id|>\"\"\"
PARAMETER stop "<|start_header_id|>"
PARAMETER stop "<|end_header_id|>"
PARAMETER stop "<|eot_id|>"
"""
                elif "gemma" in model_lower:
                    template_str = """TEMPLATE \"\"\"{{ if .System }}<start_of_turn>system
{{ .System }}<end_of_turn>
{{ end }}{{ if .Prompt }}<start_of_turn>user
{{ .Prompt }}<end_of_turn>
{{ end }}<start_of_turn>model
{{ .Response }}<end_of_turn>\"\"\"
PARAMETER stop "<start_of_turn>"
PARAMETER stop "<end_of_turn>"
"""
                elif "qwen" in model_lower:
                    template_str = """TEMPLATE \"\"\"{{ if .System }}<|im_start|>system
{{ .System }}<|im_end|>
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}<|im_end|>
{{ end }}<|im_start|>assistant
{{ .Response }}<|im_end|>\"\"\"
PARAMETER stop "<|im_start|>"
PARAMETER stop "<|im_end|>"
"""

                modelfile_content = f'''FROM "{gguf_path.resolve()}"
{template_str}PARAMETER temperature 0.3
PARAMETER top_p 0.9
'''
                modelfile_path = gguf_path.parent / "Modelfile"
                
                with open(modelfile_path, "w") as f:
                    f.write(modelfile_content)
                
                status_text.text(f"Importing to Ollama as '{ollama_model_name}'...")
                
                # Execute ollama create
                result = subprocess.run(
                    ["ollama", "create", ollama_model_name, "-f", str(modelfile_path)],
                    capture_output=True, text=True
                )
                
                if result.returncode == 0:
                    status_text.text("Done!")
                    st.success(f"✅ Model successfully fine-tuned and imported into Ollama as '{ollama_model_name}'!")
                else:
                    st.error(f"Failed to import to Ollama:\n{result.stderr}")
                    
        except Exception as e:
            st.error(f"An error occurred during fine-tuning: {e}")
            import traceback
            st.code(traceback.format_exc())
