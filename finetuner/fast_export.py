import os
import subprocess
from pathlib import Path
from unsloth import FastLanguageModel

FINETUNER_DIR = Path(__file__).parent.resolve()
max_seq_length = 2048
quant_method = "q4_k_m"
export_dir = FINETUNER_DIR / "exported_model"
ollama_model_name = "biomechanizator_v9"

print("Loading model and tokenizer in Unsloth from stable_lora...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "outputs/stable_lora",
    max_seq_length = max_seq_length,
    dtype = None,
    load_in_4bit = True,
)

print("Applying tokenizer class fix...")
tokenizer.tokenizer_class = "PreTrainedTokenizerFast"

print(f"Exporting to GGUF in {quant_method} (this will take a minute)...")
model.save_pretrained_gguf(export_dir, tokenizer, quantization_method = quant_method)

print("GGUF Export complete! Constructing Ollama Modelfile...")
gguf_files = list(FINETUNER_DIR.rglob("*.gguf"))
if not gguf_files:
    print("Error: GGUF file not found!")
    exit(1)

gguf_path = sorted(gguf_files, key=lambda p: p.stat().st_mtime, reverse=True)[0]

template_str = """TEMPLATE \"\"\"{{ if .System }}<|start_header_id|>system<|end_header_id|>

{{ .System }}<|eot_id|>{{ end }}{{ if .Prompt }}<|start_header_id|>user<|end_header_id|>

{{ .Prompt }}<|eot_id|>{{ end }}<|start_header_id|>assistant<|end_header_id|>

{{ .Response }}<|eot_id|>\"\"\"
PARAMETER stop "<|start_header_id|>"
PARAMETER stop "<|end_header_id|>"
PARAMETER stop "<|eot_id|>"
"""

modelfile_content = f'''FROM "{gguf_path.resolve()}"
{template_str}PARAMETER temperature 0.3
PARAMETER top_p 0.9
'''
modelfile_path = gguf_path.parent / "Modelfile"

with open(modelfile_path, "w") as f:
    f.write(modelfile_content)

print(f"Re-importing into Ollama as '{ollama_model_name}'...")
result = subprocess.run(
    ["ollama", "create", ollama_model_name, "-f", str(modelfile_path)],
    capture_output=True, text=True
)

if result.returncode == 0:
    print(f"\nSUCCESS! Model successfully imported into Ollama as '{ollama_model_name}'!")
else:
    print(f"\nOllama import failed: {result.stderr}")
