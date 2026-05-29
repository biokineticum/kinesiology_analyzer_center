import os
import re
import glob
from pathlib import Path
import streamlit as st
try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
import pandas as pd
import plotly.express as px
import pymupdf as fitz
import json

INBOX_DIR = Path("inbox")
PROCESSED_DIR = Path("processed")

INBOX_DIR.mkdir(exist_ok=True)
PROCESSED_DIR.mkdir(exist_ok=True)

st.set_page_config(page_title="📊 Kinesiology Analyzer Center", layout="wide")
st.title("📊 KINESIOLOGY ANALYZER CENTER")
st.caption("Advanced Biomechanical & Isokinetic Data Analytics")

MODEL_NAME = "nemotron-3-nano:4b"

def generate_llm_response(provider, model, prompt, api_key=None, temperature=0.4, max_tokens=4096, context_size=32768):
    if provider == "Local Ollama":
        if not OLLAMA_AVAILABLE:
            raise ImportError("The local 'ollama' Python library is not installed. Please install it using pip.")
        import ollama
        response = ollama.chat(
            model=model,
            messages=[{'role': 'user', 'content': prompt}],
            options={
                "num_ctx": context_size,
                "num_predict": max_tokens,
                "temperature": temperature,
            }
        )
        msg = response.get('message', {})
        content = msg.get('content', '') or ''
        thinking = msg.get('thinking', '') or ''
        return content, thinking
    elif provider == "Google Gemini API":
        if not api_key or not api_key.strip():
            raise ValueError("Google Gemini API Key is missing. Please enter your API key in the sidebar settings.")
        import requests
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "maxOutputTokens": max_tokens
            }
        }
        res = requests.post(url, headers=headers, json=payload, timeout=120)
        if res.status_code == 200:
            res_json = res.json()
            try:
                content = res_json['candidates'][0]['content']['parts'][0]['text']
                return content, ""
            except Exception:
                raise ValueError(f"Failed to parse Gemini API response. Response: {res_json}")
        else:
            raise ValueError(f"Google Gemini API Error (Status {res.status_code}): {res.text}")
    elif provider == "OpenAI API":
        if not api_key or not api_key.strip():
            raise ValueError("OpenAI API Key is missing. Please enter your API key in the sidebar settings.")
        import requests
        url = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        res = requests.post(url, headers=headers, json=payload, timeout=120)
        if res.status_code == 200:
            res_json = res.json()
            try:
                content = res_json['choices'][0]['message']['content']
                return content, ""
            except Exception:
                raise ValueError(f"Failed to parse OpenAI API response. Response: {res_json}")
        else:
            raise ValueError(f"OpenAI API Error (Status {res.status_code}): {res.text}")
    else:
        raise ValueError(f"Unsupported AI provider: {provider}")

def get_provider_models(provider):
    if provider == "Google Gemini API":
        return ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-2.0-flash-thinking-exp", "gemini-2.5-flash"]
    elif provider == "OpenAI API":
        return ["gpt-4o-mini", "gpt-4o", "gpt-3.5-turbo"]
    else:  # Local Ollama
        local_models = []
        if OLLAMA_AVAILABLE:
            try:
                ollama_models = ollama.list()
                local_models = [m['name'] for m in ollama_models.get('models', [])]
            except Exception:
                pass
        if not local_models:
            local_models = ["llama3.2:3b", "nemotron-3-nano:4b"]
        return local_models

def parse_biodex_pdf(pdf_path):
    try:
        doc = fitz.open(str(pdf_path))
        final_records = []
        
        for page_num, page in enumerate(doc):
            words = page.get_text("words")
            
            # Zbieranie i sortowanie tekstu z zachowaniem struktury tabelarycznej (grupowanie Y)
            lines_dict = {}
            for w in words:
                y_val = round(w[1] / 12) * 12
                lines_dict.setdefault(y_val, []).append((w[0], w[4]))
            
            lines = []
            for y, line_words in sorted(lines_dict.items()):
                line_text = " ".join([x[1] for x in sorted(line_words)])
                lines.append(line_text)

            player_name = "Brak"
            speed = "60"
            
            for line in lines:
                if "Patient Name:" in line:
                    m = re.search(r'Patient Name:\s+(.*?)\s+Date:', line)
                    if m:
                        player_name = m.group(1).strip()
                    break
                    
            for line in lines:
                m = re.search(r'Extension \((\d+) deg/s\)', line)
                if m:
                    speed = m.group(1)
                    break
                    
            records = {
                "Extension_Left": {"Ruch": "Extension", "Strona": "Left"},
                "Extension_Right": {"Ruch": "Extension", "Strona": "Right"},
                "Flexion_Left": {"Ruch": "Flexion", "Strona": "Left"},
                "Flexion_Right": {"Ruch": "Flexion", "Strona": "Right"},
            }
            
            mapping = [
                ("Number of Reps", "Number of Reps", 4),
                ("Peak Torque (N-m)", "Peak Torque (N-m)", 6),
                ("Avg. Peak Torque (N-m)", "Avg. Peak Torque (N-m)", 4),
                ("Peak Torque/BW (%)", "Peak Torque/BW (%)", 4),
                ("Time to Peak Torque (msec)", "Time to Peak Torque (msec)", 4),
                ("Angle of Peak Torque (deg)", "Angle of Peak Torque (deg)", 4),
                ("Torque at 30 (deg)", "Torque at 30 (deg)", 6),
                ("Torque at 0.18 (sec)", "Torque at 0.18 (sec)", 6),
                ("CV (%)", "CV (%)", 4),
                ("Max. Rep Total Work (J)", "Max. Rep Total Work (J)", 6),
                ("Work/BW (%)", "Max. Rep Total Work/BW (J)", 4), 
                ("Total Work (J)", "Total Work (J)", 6),
                ("Work First Third (J)", "Work First Third (J)", 4),
                ("Work Last Third (J)", "Work Last Third (J)", 4),
                ("Work Fatigue (%)", "Work Fatigue (%)", 4),
                ("Avg. Power (W)", "Avg. Power (W)", 6),
                ("Acceleration Time (msec)", "Acceleration Time (msec)", 4),
                ("Deceleration Time (msec)", "Deceleration Time (msec)", 4),
                ("ROM (deg)", "ROM (deg)", 2),
                ("AGON/ANTAG Ratio (%)", "AGON/ANTAG Ratio (%)", 2),
            ]
            
            found_data = False
            
            for line in lines:
                clean_line = re.sub(r'\(Rep \d+\)', '', line).strip()
                clean_line = clean_line.replace("(N•m)", "(N-m)")
                
                for prefix, target_key, structure in mapping:
                    if clean_line.startswith(prefix):
                        nums = re.findall(r'[\d\.]+', clean_line.replace(prefix, ""))
                        
                        if structure == 6:
                            if len(nums) >= 4:
                                found_data = True
                                records["Extension_Left"][target_key] = nums[0]
                                records["Extension_Right"][target_key] = nums[1]
                                if target_key == "Peak Torque (N-m)" and len(nums) >= 6:
                                    records["Extension_Left"]["Deficit (%)"] = nums[2]
                                    records["Extension_Right"]["Deficit (%)"] = nums[2]
                                    
                                if len(nums) >= 6:
                                    records["Flexion_Left"][target_key] = nums[3]
                                    records["Flexion_Right"][target_key] = nums[4]
                                    if target_key == "Peak Torque (N-m)":
                                        records["Flexion_Left"]["Deficit (%)"] = nums[5]
                                        records["Flexion_Right"]["Deficit (%)"] = nums[5]
                                else:
                                    records["Flexion_Left"][target_key] = nums[2]
                                    records["Flexion_Right"][target_key] = nums[3]
                            elif len(nums) >= 2:
                                found_data = True
                                records["Extension_Left"][target_key] = nums[0]
                                records["Extension_Right"][target_key] = nums[1]
                        
                        elif structure == 4:
                            if len(nums) >= 4:
                                found_data = True
                                records["Extension_Left"][target_key] = nums[0]
                                records["Extension_Right"][target_key] = nums[1]
                                records["Flexion_Left"][target_key] = nums[2]
                                records["Flexion_Right"][target_key] = nums[3]
                            elif len(nums) >= 2:
                                found_data = True
                                records["Extension_Left"][target_key] = nums[0]
                                records["Extension_Right"][target_key] = nums[1]
                                
                        elif structure == 2:
                            if len(nums) >= 2:
                                found_data = True
                                records["Extension_Left"][target_key] = nums[0]
                                records["Extension_Right"][target_key] = nums[1]
                                records["Flexion_Left"][target_key] = nums[0]
                                records["Flexion_Right"][target_key] = nums[1]
                        break
                        
            if found_data:
                all_fields = [
                    "Number of Reps", "Avg. Peak Torque (N-m)", "Peak Torque (N-m)", "Deficit (%)",
                    "Peak Torque/BW (%)", "Time to Peak Torque (msec)", "Angle of Peak Torque (deg)", 
                    "Torque at 30 (deg)", "Torque at 0.18 (sec)", "CV (%)", "Max. Rep Total Work (J)", 
                    "Max. Rep Total Work/BW (J)", "Total Work (J)", "Work First Third (J)", 
                    "Work Last Third (J)", "Work Fatigue (%)", "Avg. Power (W)", 
                    "Acceleration Time (msec)", "Deceleration Time (msec)", "ROM (deg)", "AGON/ANTAG Ratio (%)"
                ]
                for k, rec in records.items():
                    rec["Zawodnik"] = player_name
                    rec["Prędkość (°/s)"] = speed
                    rec["Plik PDF"] = pdf_path.name
                    rec["Test"] = "EXT/FLEX"
                    
                    for field in all_fields:
                        if field not in rec:
                            rec[field] = "-"
                            
                    final_records.append(rec)
                    
        return final_records
    except Exception as e:
        st.warning(f"Error parsing Biodex PDF {pdf_path.name}: {e}")
        return []

with st.sidebar:
    st.header("⚙️ Settings")
    
    # 🤖 AI Provider Settings
    st.subheader("🤖 AI Provider Settings")
    ai_provider = st.selectbox(
        "Select LLM Provider:",
        ["Local Ollama", "Google Gemini API", "OpenAI API"]
    )
    
    api_key = None
    
    if ai_provider == "Google Gemini API":
        api_key = st.text_input("Google Gemini API Key:", type="password", help="Enter your Gemini API key.")
        st.caption("[Get a free Gemini API key from Google AI Studio](https://aistudio.google.com/)")
    elif ai_provider == "OpenAI API":
        api_key = st.text_input("OpenAI API Key:", type="password", help="Enter your OpenAI API key.")
        
    st.divider()
    st.subheader("⚙️ Model Parameters")
    ctx_size = st.slider("Context size (tokens)", 4096, 131072, 32768, step=4096)
    max_tokens = st.slider("Max output tokens", 1024, 16384, 4096, step=512)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.4, step=0.1)
    
    if st.button("🧹 Clear session"):
        st.session_state.clear()
        
    st.divider()
    st.header("📊 Data Analysis")
    
    source_type = st.radio("Choose Data Source:", ["Excel/CSV File", "Parse Raw Biodex PDFs"])
    
    selected_data_file = None
    if source_type == "Excel/CSV File":
        data_files = glob.glob(str(INBOX_DIR / "*.xlsx")) + glob.glob(str(INBOX_DIR / "*.csv"))
        if not data_files:
            st.caption("No Excel/CSV files found in inbox/")
            selected_data_file = None
        else:
            data_names = [Path(f).name for f in data_files]
            selected_data_name = st.selectbox("Select Data File", data_names)
            selected_data_file = INBOX_DIR / selected_data_name
    else:
        pdf_files = glob.glob(str(INBOX_DIR / "*.pdf"))
        if not pdf_files:
            st.caption("No PDF files found in inbox/")
        else:
            pdf_names = [Path(f).name for f in pdf_files]
            selected_biodex_names = st.multiselect("Select Biodex PDFs to Parse", pdf_names, key="biodex_parse_multiselect")
            
            if selected_biodex_names:
                if st.button("⚡ Parse Biodex PDFs", type="primary", key="biodex_parse_btn"):
                    with st.spinner("Parsing Biodex reports..."):
                        all_records = []
                        for name in selected_biodex_names:
                            pdf_path = INBOX_DIR / name
                            records = parse_biodex_pdf(pdf_path)
                            if records:
                                all_records.extend(records)
                        
                        if all_records:
                            parsed_df = pd.DataFrame(all_records)
                            
                            # Clean numeric columns (convert string '-' or empty to NaN, and convert to numeric)
                            cols_to_skip = ["Zawodnik", "Plik PDF", "Test", "Ruch", "Strona", "Prędkość (°/s)"]
                            for col in parsed_df.columns:
                                if col not in cols_to_skip:
                                    parsed_df[col] = pd.to_numeric(parsed_df[col].replace('-', None), errors='coerce')
                            
                            processed_file = PROCESSED_DIR / "biodex_parsed_dataset.csv"
                            parsed_df.to_csv(processed_file, index=False, encoding='utf-8')
                            st.session_state['parsed_biodex_df'] = parsed_df
                            st.success(f"Parsed {len(all_records)} records!")
                        else:
                            st.error("No Biodex data extracted.")
                            
                if 'parsed_biodex_df' in st.session_state:
                    st.caption("Using parsed Biodex data")

    st.divider()
    
    st.header("📄 Batch PDF Parser")
    pdf_files = glob.glob(str(INBOX_DIR / "*.pdf"))
    if not pdf_files:
        st.caption("No PDF files found in inbox/")
        selected_pdfs = []
    else:
        pdf_names = [Path(f).name for f in pdf_files]
        selected_pdf_names = st.multiselect("Select PDFs to Batch Process", pdf_names)
        selected_pdfs = [INBOX_DIR / name for name in selected_pdf_names]

@st.cache_data
def load_data(file_path):
    if not file_path:
        return None
    try:
        if file_path.suffix.lower() == '.csv':
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)
        return df
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return None

@st.cache_data
def load_pdf(file_path):
    if not file_path or file_path.suffix.lower() != '.pdf':
        return None
    try:
        doc = fitz.open(file_path)
        if doc.needs_pass:
            st.warning(f"Skipping {file_path.name}: PDF is encrypted or password protected.")
            return None
            
        text = ""
        for page in doc:
            text += page.get_text() + "\n\n"
        return text
    except Exception as e:
        st.warning(f"Skipping {file_path.name} due to read error: {e}")
        return None

if selected_pdfs:
    preview_text = load_pdf(selected_pdfs[0]) if selected_pdfs else ""
    if preview_text:
        tab1, tab2, tab3 = st.tabs(["📄 Document Preview", "🧠 AI RAG Biomechanics Assistant", "🛠️ Bulk Dataset Generator"])
        
        with tab1:
            st.subheader(f"Extracted Text (Previewing 1 of {len(selected_pdfs)})")
            st.text_area("Content", value=preview_text, height=600, disabled=True)
            
        with tab2:
            st.subheader("🧠 AI RAG Biomechanics Assistant")
            st.caption("Ask questions or generate reports directly from the selected scientific PDFs using RAG.")
            
            col_m1, col_m2 = st.columns(2)
            with col_m1:
                available_models = get_provider_models(ai_provider)
                rag_model = st.selectbox("Select Model for RAG:", available_models, key="rag_model_select")
            with col_m2:
                preset = st.selectbox("Preset Analysis Tasks:", [
                    "Custom Question (Type below)",
                    "Generate Clinical Assessment Report",
                    "Identify Kinetic Asymmetries & Imbalances",
                    "Extract Specific Numerical Biomechanical Metrics"
                ])
                
            preset_prompts = {
                "Custom Question (Type below)": "",
                "Generate Clinical Assessment Report": "Please write a comprehensive sports physiotherapy clinical assessment report based on the biomechanical measurements in the text. Focus on deficits, range of motion, force parameters, and outline concrete exercises for rehabilitation.",
                "Identify Kinetic Asymmetries & Imbalances": "Identify and analyze all kinetic asymmetries, muscular imbalances, or joint velocity deficits reported in the text. Highlight statistically significant differences (p-values) and discuss injury risks.",
                "Extract Specific Numerical Biomechanical Metrics": "Create a structured summary of all key numerical metrics found in the document, including peak torques (N-m), deficit percentages (%), work (J), and power (W) across different speeds."
            }
            
            rag_query = st.text_area("Your Question / Analysis Instructions:", 
                                     value=preset_prompts[preset], 
                                     height=120,
                                     placeholder="e.g., What is the reported maximal velocity of the toe marker during the left roundhouse kick?")
            
            with st.expander("⚙️ RAG Search Parameters (Advanced)"):
                rag_chunk_size = st.slider("RAG Chunk size (chars)", 1000, 30000, 15000, step=1000)
                rag_top_k = st.slider("Retrieve Top K most relevant chunks", 1, 8, 3, step=1)
                
            if st.button("🚀 Analyze & Generate Diagnosis", type="primary"):
                if not rag_query.strip():
                    st.error("Please enter a question or instruction.")
                else:
                    with st.spinner("Extracting text and retrieving relevant context..."):
                        # Step 1: Chunk all selected PDFs using smart sentence boundaries
                        rag_chunks = []
                        
                        def smart_chunk_text(text, max_chunk_size=1500, overlap=300):
                            import re
                            paragraphs = text.split("\n\n")
                            chunks = []
                            current_chunk = []
                            current_size = 0
                            
                            for para in paragraphs:
                                para = para.strip()
                                if not para:
                                    continue
                                
                                # Split by sentence boundaries, respecting common biomechanical abbreviations (e.g. vs, approx, st.dev)
                                sentences = re.split(r'(?<!\bvs)(?<!\bapprox)(?<!\bdev)(?<=[.!?])\s+', para)
                                
                                for sentence in sentences:
                                    sentence = sentence.strip()
                                    if not sentence:
                                        continue
                                    
                                    sentence_len = len(sentence)
                                    if sentence_len > max_chunk_size:
                                        if current_chunk:
                                            chunks.append(" ".join(current_chunk))
                                            current_chunk = []
                                            current_size = 0
                                        # Force split giant sentence
                                        for j in range(0, sentence_len, max_chunk_size - overlap):
                                            chunks.append(sentence[j:j+max_chunk_size])
                                        continue
                                    
                                    if current_size + sentence_len + 1 > max_chunk_size:
                                        chunks.append(" ".join(current_chunk))
                                        # Sentence-level overlap
                                        overlap_chunk = []
                                        overlap_size = 0
                                        for s in reversed(current_chunk):
                                            if overlap_size + len(s) + 1 <= overlap:
                                                overlap_chunk.insert(0, s)
                                                overlap_size += len(s) + 1
                                            else:
                                                break
                                        current_chunk = overlap_chunk
                                        current_size = overlap_size
                                        
                                    current_chunk.append(sentence)
                                    current_size += sentence_len + 1
                                    
                            if current_chunk:
                                chunks.append(" ".join(current_chunk))
                            return chunks

                        for pdf_file in selected_pdfs:
                            text = load_pdf(pdf_file)
                            if text:
                                chunks = smart_chunk_text(text, max_chunk_size=rag_chunk_size, overlap=int(rag_chunk_size * 0.2))
                                for i, chunk in enumerate(chunks):
                                    if len(chunk.strip()) > 80:
                                        rag_chunks.append({
                                            "source": pdf_file.name,
                                            "content": chunk.strip(),
                                            "start_idx": i
                                        })
                                        
                        if not rag_chunks:
                            st.error("No valid text found in the selected PDFs to query.")
                        else:
                            # Step 2: Retrieve Top K using Robust TF-IDF + Numeric Boost Hybrid Search
                            import math
                            from collections import Counter
                            
                            def tokenize(t):
                                return [w.strip(".,;:?!()[]{}'\"").lower() for w in t.split() if len(w.strip(".,;:?!()[]{}'\"")) > 0]
                                
                            # Compute Document Frequency (DF) across chunks
                            all_terms = []
                            for chunk_obj in rag_chunks:
                                all_terms.append(set(tokenize(chunk_obj["content"])))
                                
                            N = len(rag_chunks)
                            df_dict = Counter()
                            for term_set in all_terms:
                                for term in term_set:
                                    df_dict[term] += 1
                                    
                            # Compute IDF
                            idf = {}
                            for term, count in df_dict.items():
                                idf[term] = math.log(1 + N / (1 + count))
                                
                            query_tokens = tokenize(rag_query)
                            query_numbers = [t for t in query_tokens if any(char.isdigit() for char in t)]
                            
                            scored_chunks = []
                            for chunk_obj in rag_chunks:
                                chunk_content = chunk_obj["content"]
                                chunk_tokens = tokenize(chunk_content)
                                chunk_counter = Counter(chunk_tokens)
                                
                                # Compute TF-IDF overlap
                                score = 0.0
                                for q_term in set(query_tokens):
                                    if q_term in chunk_counter:
                                        tf = chunk_counter[q_term]
                                        term_idf = idf.get(q_term, 0.5)
                                        score += tf * term_idf
                                        
                                # Normalize by chunk length logarithm to avoid favoring excessively long chunks
                                score = score / (1 + math.log(1 + len(chunk_tokens)))
                                
                                # Special Numeric matching boost
                                number_matches = 0
                                for num in query_numbers:
                                    if num in chunk_content.lower():
                                        number_matches += 1
                                        
                                if query_numbers:
                                    score += (number_matches / len(query_numbers)) * 0.5
                                    
                                scored_chunks.append((score, chunk_obj))
                                
                            # Sort by score descending and take Top K
                            scored_chunks.sort(key=lambda x: x[0], reverse=True)
                            top_matches = scored_chunks[:rag_top_k]
                            
                            # Construct prompt with matched context
                            context_str = ""
                            for rank, (score, match) in enumerate(top_matches):
                                if score > 0.001 or rank == 0:  # Include at least the top match
                                    context_str += f"\n--- Source: {match['source']} (Relevance Score: {score:.2f}) ---\n{match['content']}\n"
                                    
                            # Step 3: Run Ollama
                            system_instruction = (
                                "You are a world-class expert in sports science, clinical biomechanics, human kinetics, and athletic performance analysis. "
                                "Your goal is to answer the user's question or perform the analysis using ONLY the provided scientific PDF extracts (Context). "
                                "Provide a professional, extremely precise, and academically rigorous analysis, citing specific numerical values, standard deviations, p-values, speeds, forces, and segments. "
                                "You can handle all topics related to sports science, including boxing punches, karate kicks, joint velocities, muscular deficits, and athletic rehabilitation. "
                                "If the context doesn't contain the answer, explain clearly and politely that the provided research documents do not contain enough details to answer, rather than hallucinating."
                            )
                            
                            full_prompt = (
                                f"{system_instruction}\n\n"
                                f"CONTEXT EXTRACTS FROM PDFs:\n{context_str}\n\n"
                                f"USER INSTRUCTION:\n{rag_query}"
                            )
                            
                            try:
                                content, thinking = generate_llm_response(
                                    provider=ai_provider,
                                    model=rag_model,
                                    prompt=full_prompt,
                                    api_key=api_key,
                                    temperature=temperature,
                                    max_tokens=max_tokens,
                                    context_size=ctx_size
                                )
                                
                                if thinking.strip():
                                    with st.expander("🤔 RAG Reasoning Process"):
                                        st.markdown(thinking)
                                        
                                if content.strip():
                                    st.success("✅ RAG Analysis Complete")
                                    st.markdown(content)
                                else:
                                    st.warning("Empty response from Ollama.")
                                    
                                # Show retrieved sources
                                st.divider()
                                st.subheader("🔍 Retrieved Context Sources")
                                for rank, (score, match) in enumerate(top_matches):
                                    with st.expander(f"📄 Source {rank+1}: {match['source']} (Relevance: {score:.2f})"):
                                        st.caption(f"Chunk character offset: {match['start_idx']}")
                                        st.text_area("Snippet", value=match['content'], height=200, disabled=True, key=f"source_snippet_{rank}")
                                        
                            except Exception as e:
                                st.error(f"Error during RAG analysis: {e}")
                                
        with tab3:
            st.subheader("Generate Unsloth Fine-Tuning Dataset")
            available_models = get_provider_models(ai_provider)
            gen_model = st.selectbox("Select Model for Dataset Generation:", available_models, key="dataset_gen_model_select")
            
            system_prompt = st.text_area("System Instruction for the LLM:", 
                value="You are an expert biomechanics researcher creating a high-quality dataset for fine-tuning a clinical AI. "
                      "Based on the following text extract, generate 3 highly specific, complex, and scientifically accurate Q&A pairs. "
                      "Structure your output strictly as a JSON list of objects with exactly these keys: 'instruction', 'input', and 'output'. "
                      "- 'instruction': A specific analytical question or task derived from the text. "
                      "- 'input': Any necessary context or numerical data from the text required to answer. Leave empty (\"\") if the instruction is self-contained. "
                      "- 'output': The detailed, expert-level explanation or analytical answer based ONLY on the text. "
                      "Only return a valid JSON array. Do not include markdown formatting like ```json.",
                height=180)
                
            chunk_size_chars = st.slider("Chunk size (characters)", 1000, 10000, 4000, step=500)
            resume_chunk_idx = st.number_input("Wznów od fragmentu (wpisz 0 aby zacząć od nowa)", min_value=0, value=0, help="Jeśli apka się scrashowała np. na 20000 fragmencie, wpisz tu 20000, aby pominąć pierwsze 20k fragmentów.")
            filter_non_biomech = st.checkbox("Filtruj fragmenty niepowiązane z biomechaniką (LLM)", value=False, help="LLM oceni każdy fragment i odrzuci te niepowiązane z biomechaniką (np. bibliografia). Wyłączenie tego znacznie przyspiesza proces i zapobiega błędom klasyfikacji.")
            
            # Initialize session state for dataset generation
            if "gen_done" not in st.session_state:
                st.session_state.gen_done = False
            if "gen_count" not in st.session_state:
                st.session_state.gen_count = 0
            if "gen_output_file" not in st.session_state:
                st.session_state.gen_output_file = None

            if st.button("🚀 Generate Macro Dataset", type="primary"):
                # Reset session state on a new run
                st.session_state.gen_done = False
                st.session_state.gen_count = 0
                st.session_state.gen_output_file = None

                finetuner_dir = Path("finetuner")
                finetuner_dir.mkdir(exist_ok=True)
                output_file = finetuner_dir / "macro_dataset.jsonl"
                
                # --- Faza 1: Parsowanie PDF ---
                st.markdown("#### 1. Parsowanie PDF")
                pdf_progress_bar = st.progress(0)
                pdf_status_text = st.empty()
                
                all_chunks = []
                total_pdfs = len(selected_pdfs)
                for idx, pdf_file in enumerate(selected_pdfs):
                    pdf_status_text.text(f"Parsowanie pliku {idx+1} z {total_pdfs} ({pdf_file.name})...")
                    text = load_pdf(pdf_file)
                    if text:
                        chunks = [text[i:i+chunk_size_chars] for i in range(0, len(text), chunk_size_chars)]
                        all_chunks.extend(chunks)
                    pdf_progress_bar.progress((idx + 1) / total_pdfs)
                
                pdf_status_text.text(f"✅ Parsowanie zakończone. Uzyskano {len(all_chunks)} fragmentów tekstu.")
                
                if not all_chunks:
                    st.error("No valid text could be extracted from the selected PDFs.")
                    st.stop()
                    
                # --- Faza 2: Przetwarzanie przez LLM ---
                st.markdown("#### 2. Klasyfikacja i Generowanie Datasetu (LLM)")
                llm_progress_bar = st.progress(0)
                llm_status_text = st.empty()
                generated_count = 0
                
                # Używamy trybu 'a' (append) jeśli wznawiamy, aby nie nadpisać dotychczasowej pracy!
                file_mode = 'a' if resume_chunk_idx > 0 else 'w'
                
                if resume_chunk_idx > 0 and len(all_chunks) > 0:
                     llm_progress_bar.progress(min(resume_chunk_idx / len(all_chunks), 1.0))
                     llm_status_text.text(f"Wznowiono od chunka {resume_chunk_idx}...")
                
                with open(output_file, file_mode, encoding='utf-8') as f:
                    for i, chunk in enumerate(all_chunks):
                        if i < resume_chunk_idx:
                            continue # Pomijamy fragmenty, które już zostały przerobione
                            
                        llm_status_text.text(f"Przetwarzanie chunka {i+1} z {len(all_chunks)} (zapisano par Q&A w tej sesji: {generated_count})...")
                        
                        # KROK 1: Klasyfikacja LLM (Czy chunk jest o biomechanice?)
                        if filter_non_biomech:
                            class_prompt = "Analyze the following text extract. Does it contain meaningful information related to biomechanics, human kinematics, kinetics, physiotherapy, or sports science? Answer strictly with one word: YES or NO.\n\nTEXT:\n" + chunk[:2000]
                            try:
                                content, thinking = generate_llm_response(
                                    provider=ai_provider,
                                    model=gen_model,
                                    prompt=class_prompt,
                                    api_key=api_key,
                                    temperature=0.1,
                                    max_tokens=250,
                                    context_size=4096
                                )
                                answer = (content + " " + thinking).strip().upper()
                                
                                if "YES" not in answer and "TAK" not in answer:
                                    st.info(f"⏭️ LLM odrzucił chunk {i+1} (brak związku z biomechaniką)")
                                    llm_progress_bar.progress((i + 1) / len(all_chunks))
                                    continue
                            except Exception as e:
                                st.warning(f"Error during classification of chunk {i+1}: {e}")
                            
                        # KROK 2: Generowanie datasetu dla zaakceptowanego chunka
                        prompt = f"{system_prompt}\n\nTEXT EXTRACT:\n{chunk}"
                        
                        try:
                            content, thinking = generate_llm_response(
                                provider=ai_provider,
                                model=gen_model,
                                prompt=prompt,
                                api_key=api_key,
                                temperature=temperature,
                                max_tokens=max_tokens,
                                context_size=ctx_size
                            )
                            start_idx = content.find('[')
                            end_idx = content.rfind(']')
                            
                            if start_idx != -1 and end_idx != -1:
                                json_str = content[start_idx:end_idx+1]
                                pairs = json.loads(json_str)
                                
                                for pair in pairs:
                                    if 'instruction' in pair and 'output' in pair:
                                        if 'input' not in pair:
                                            pair['input'] = ""
                                        json.dump(pair, f, ensure_ascii=False)
                                        f.write('\n')
                                        f.flush()  # Zapisz natychmiast na dysk!
                                        generated_count += 1
                        except Exception as e:
                            st.warning(f"Error processing chunk {i+1}: {e}")
                            
                        llm_progress_bar.progress((i + 1) / len(all_chunks))
                        
                llm_status_text.text("Zakończono procesowanie!")
                st.session_state.gen_done = True
                st.session_state.gen_count = generated_count
                st.session_state.gen_output_file = str(output_file)

            # Render generated dataset download controls if ready
            if st.session_state.gen_done and st.session_state.gen_output_file:
                st.success(f"✅ Macro dataset generated successfully with {st.session_state.gen_count} pairs! Saved directly to {st.session_state.gen_output_file}")
                try:
                    with open(st.session_state.gen_output_file, 'r', encoding='utf-8') as f:
                        file_data = f.read()
                    
                    st.download_button(
                        label="⬇️ Download Macro JSONL Dataset",
                        data=file_data,
                        file_name=Path(st.session_state.gen_output_file).name,
                        mime="application/jsonlines",
                        key="download_macro_dataset_btn"
                    )
                except Exception as e:
                    st.error(f"Error reading dataset for download: {e}")

if 'source_type' in locals() and source_type == "Parse Raw Biodex PDFs":
    df = st.session_state.get('parsed_biodex_df', None)
else:
    df = load_data(selected_data_file)

if df is not None:
    tab1, tab2, tab3 = st.tabs(["📊 Data & Stats", "📈 Dynamic Visualizations", "🤖 AI Biomechanics Analyst"])
    
    # Analyze columns in df
    all_cols = df.columns.tolist()
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(exclude=['number']).columns.tolist()
    
    with tab1:
        st.subheader("📊 Dataset Overview")
        st.dataframe(df.head(10))
        
        st.subheader("🧮 Interactive Descriptive Statistics")
        st.caption("Configure custom groups and variables to aggregate dynamically from your biomechanics dataset.")
        
        col_g1, col_g2 = st.columns(2)
        with col_g1:
            # Default groups: check if Ruch, Prędkość, Strona exist
            def_groups = [c for c in ['Ruch', 'Prędkość (°/s)', 'Strona', 'Zawodnik'] if c in all_cols]
            if not def_groups and categorical_cols:
                def_groups = [categorical_cols[0]]
            selected_groups = st.multiselect("Grouping Variables (Categorical):", all_cols, default=def_groups)
            
        with col_g2:
            # Default metrics: check if Peak Torque, Work, Power, Deficit exist
            def_metrics = [c for c in ['Peak Torque (N-m)', 'Total Work (J)', 'Avg. Power (W)', 'Deficit (%)'] if c in numeric_cols]
            if not def_metrics and numeric_cols:
                def_metrics = [numeric_cols[0]]
            selected_metrics = st.multiselect("Metrics to Analyze (Numerical):", numeric_cols, default=def_metrics)
            
        st.divider()
        
        if not selected_groups or not selected_metrics:
            st.info("Select at least one grouping variable and one numerical metric to compute statistics.")
            # Fallback stats
            st.dataframe(df.describe())
            st.session_state['stats_text'] = df.describe().to_string()
        else:
            try:
                # Dynamic aggregation
                grouped_stats = df.groupby(selected_groups)[selected_metrics].agg(['mean', 'std', 'max', 'min']).round(2)
                st.dataframe(grouped_stats)
                
                # Export to string for AI
                stats_text = f"Biomechanical data grouped by {', '.join(selected_groups)}:\n" + grouped_stats.to_string()
                st.session_state['stats_text'] = stats_text
            except Exception as e:
                st.error(f"Error computing grouped stats: {e}")
                st.dataframe(df.describe())
                st.session_state['stats_text'] = df.describe().to_string()
                
    with tab2:
        st.subheader("📈 Dynamic Visualization Builder")
        st.caption("Select your axes, grouping variables, and plot type to dynamically visualize any biomechanical indicators.")
        
        if len(all_cols) < 2:
            st.warning("Not enough columns to render custom visualizations.")
        else:
            col_v1, col_v2 = st.columns(2)
            with col_v1:
                plot_type = st.selectbox("Chart Type:", ["Box Plot", "Bar Chart", "Scatter Plot", "Line Chart"])
                
                # Pick X axis
                def_x = 'Prędkość (°/s)' if 'Prędkość (°/s)' in all_cols else all_cols[0]
                x_axis = st.selectbox("X-Axis Variable:", all_cols, index=all_cols.index(def_x) if def_x in all_cols else 0)
                
            with col_v2:
                # Pick Y axis
                def_y = 'Peak Torque (N-m)' if 'Peak Torque (N-m)' in numeric_cols else (numeric_cols[0] if numeric_cols else all_cols[-1])
                y_axis = st.selectbox("Y-Axis Variable (Numerical):", numeric_cols, index=numeric_cols.index(def_y) if def_y in numeric_cols else 0)
                
                # Optional color mapping
                color_options = ["None"] + all_cols
                def_color = 'Strona' if 'Strona' in all_cols else 'None'
                color_var = st.selectbox("Color / Grouping Variable (Optional):", color_options, index=color_options.index(def_color) if def_color in color_options else 0)
                
            # Optional facet column
            facet_options = ["None"] + all_cols
            def_facet = 'Ruch' if 'Ruch' in all_cols else 'None'
            facet_var = st.selectbox("Facet / Column Wrap (Optional):", facet_options, index=facet_options.index(def_facet) if def_facet in facet_options else 0)
            
            st.divider()
            
            with st.spinner("Generating custom plot..."):
                try:
                    color_arg = None if color_var == "None" else color_var
                    facet_arg = None if facet_var == "None" else facet_var
                    
                    title_text = f"{y_axis} by {x_axis}"
                    if color_arg:
                        title_text += f" grouped by {color_arg}"
                    
                    if plot_type == "Box Plot":
                        fig = px.box(df, x=x_axis, y=y_axis, color=color_arg, facet_col=facet_arg, title=title_text)
                    elif plot_type == "Bar Chart":
                        # Group by inputs to prevent overlapping bars
                        group_vars = [x_axis]
                        if color_arg: group_vars.append(color_arg)
                        if facet_arg: group_vars.append(facet_arg)
                        grouped_df = df.groupby(group_vars)[y_axis].mean().reset_index()
                        fig = px.bar(grouped_df, x=x_axis, y=y_axis, color=color_arg, facet_col=facet_arg, barmode="group", title=f"Mean {title_text}")
                    elif plot_type == "Scatter Plot":
                        fig = px.scatter(df, x=x_axis, y=y_axis, color=color_arg, facet_col=facet_arg, title=title_text)
                    elif plot_type == "Line Chart":
                        group_vars = [x_axis]
                        if color_arg: group_vars.append(color_arg)
                        if facet_arg: group_vars.append(facet_arg)
                        grouped_df = df.groupby(group_vars)[y_axis].mean().reset_index()
                        fig = px.line(grouped_df, x=x_axis, y=y_axis, color=color_arg, facet_col=facet_arg, title=f"Mean {title_text}")
                        
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.error(f"Error rendering chart: {e}")
                    
    with tab3:
        st.subheader("🤖 AI Biomechanics Analyst")
        st.caption("Generate a professional, detailed analytical report from the dynamically aggregated data.")
        
        col_am1, col_am2 = st.columns(2)
        with col_am1:
            available_models = get_provider_models(ai_provider)
            analyst_model = st.selectbox("Select Model for Analysis:", available_models, key="analyst_model_select")
        
        ai_prompt = st.text_area("System Instruction for the AI:", 
            value="You are a world-class sports science researcher, clinical biomechanist, and athletic performance analyst. "
                  "I will provide you with a descriptive statistical summary of a biomechanical evaluation. "
                  "Analyze the results across different experimental groups/conditions. Identify significant differences, performance imbalances, peak forces/velocities, or potential athletic/clinical risks. "
                  "Provide a professional, highly structured, and concise analysis citing exact numbers, mean values, and standard deviations with concrete actionable recommendations.",
            height=150)
            
        if st.button("🚀 Generate AI Report", type="primary"):
            stats_content = st.session_state.get('stats_text', '')
            if not stats_content:
                st.error("No statistics available. Please check the Data tab.")
            else:
                with st.spinner(f"Analyzing the data using {analyst_model}..."):
                    try:
                        full_prompt = f"{ai_prompt}\n\nHere are the computed statistics:\n\n{stats_content}"
                        
                        content, thinking = generate_llm_response(
                            provider=ai_provider,
                            model=analyst_model,
                            prompt=full_prompt,
                            api_key=api_key,
                            temperature=temperature,
                            max_tokens=max_tokens,
                            context_size=ctx_size
                        )
                        
                        if thinking.strip():
                            with st.expander("🤔 Diagnostic Process"):
                                st.markdown(thinking)
                                
                        if content.strip():
                            st.success("✅ Assessment complete")
                            st.markdown(content)
                        else:
                            st.warning("Empty response from the model.")
                            
                    except Exception as e:
                        st.error(f"Error during AI analysis: {e}")