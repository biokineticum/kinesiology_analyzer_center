# 📊 Kinesiology Analyzer Center
### *Offline-First Biomechanical Extraction, Analytics, & Custom Local AI Fine-Tuning*

[![Python Version](https://img.shields.io/badge/python-3.10%20%7C%203.11-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![HIPAA Compliant](https://img.shields.io/badge/Privacy-GDPR%20%7C%20HIPAA%20Ready-success.svg)](#-privacy--security)

**Kinesiology Analyzer Center** is an elite, open-source, offline-first desktop application designed to bridge the gap between raw clinical biomechanics report data and advanced artificial intelligence. Built specifically for physical therapists, sports scientists, and athletic trainers, it allows laboratories to ingest, analyze, and train custom AI models on their private kinetic data without ever exposing patient files to the cloud.

---

## 🌟 Core Pillars & Features

### 📂 1. Automated Multi-Sensor Biomechanics Ingestion
* **Isokinetic Dynamometry:** Bypasses sluggish OCR by using high-fidelity spatial coordinate text reconstruction (powered by `PyMuPDF`) to directly parse tabular data from standard **Biodex System 4** and **Humac Norm** isokinetic reports.
* **EMG & Kinematics (Noraxon & Others):** Built with an extensible, modular parsing architecture ready to ingest raw TXT/CSV/PDF reports from professional biomechanics hardware—such as **Noraxon myoRESEARCH** suites—extracting electromyography (EMG) muscle activation profiles, joint angles, and synchronized 3D kinematic curves.
* **Force & Plantar Pressure:** Ready for force plate integration (Kistler, AMTI) to parse ground reaction forces (GRF) and plantar pressure matrices.
* **Dataset Consolidation:** Automatically compiles parsed multi-sensor clinical data into unified, structured Excel/CSV datasets ready for downstream bio-analytics and machine learning.

### 📈 2. Dynamic Kinematic Visualizations
* **Interactive Statistics:** Dynamically group, aggregate, and slice your consolidated athletic datasets by player name, velocity, side, or movement type.
* **Plotly Visuals:** Build and export publication-ready Box Plots, Grouped Bar Charts, Scatter Plots, and Line Charts to track athletic rehabilitation and performance progressions.

### 🧠 3. Pluggable AI & Local Fine-Tuning Pipeline
* **Multi-Provider AI Routing:** Supports **Local Ollama** (100% offline), **Google Gemini API**, and **OpenAI API** using standard, ultra-lightweight REST HTTP requests.
* **Dynamic Local Model Selectors:** Select models **independently** inside each individual tab based on the active provider. Run a fast model (like `gemini-1.5-flash`) for search/RAG and a deep model (like `gemini-1.5-pro` or a fine-tuned Ollama model) for full analytics!
* **Hybrid RAG Biomechanics Assistant:** Ask complex research questions directly against scientific PDFs using a custom, high-speed TF-IDF + Numeric Boost Hybrid RAG search.
* **Automated QA Dataset Synthesizer:** Automatically chunk research papers and stream structured, Unsloth-compatible Instruction/Input/Output JSONL training datasets.
* **In-App Training Engine:** Fine-tune Gemma, Llama, or Qwen models directly inside the UI. Features a **Stable HF PEFT** training engine that bypasses standard Windows Triton compilation bugs for 100% stable execution on consumer GPUs.
* **One-Click GGUF Export:** Automates the merging, quantization, GGUF conversion, and direct import of your fine-tuned model back into Ollama with tailor-made Modelfiles.

---

## 🛠️ Technology Stack

* **Core Logic & Extraction:** Python, `PyMuPDF` (Fitz), Regex
* **Data Processing & Analytics:** Pandas, Numpy
* **Visualizations:** Plotly Express
* **UI Framework:** Streamlit (Web Server), PySide6 (Native Qt Desktop Shell wrapper)
* **AI & Deep Learning:** Requests (REST APIs), Ollama (Local LLM), Unsloth (Stable Windows Fine-Tuning)

---

## 🚀 Getting Started

### 📋 Prerequisites
* Windows 10/11
* Python 3.10 or 3.11 installed

### 📥 Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/yourusername/kinesiology-analyzer-center.git
   cd kinesiology-analyzer-center
   ```

2. **Install Core Requirements:**
   ```bash
   pip install -r requirements.txt
   ```

3. *(Optional)* **Deep Learning & Fine-Tuning Engines:**
   * **Windows/Linux (with Nvidia GPUs):** Follow the official [Unsloth Windows installation guide](https://github.com/unslothai/unsloth#windows-installation) to configure PyTorch CUDA and the ultra-fast Unsloth engine.
   * **macOS (Apple Silicon):** Unsloth is optimized for Nvidia CUDA. On macOS, choose the **"Stable HF PEFT"** engine in the training settings! This utilizes Apple Silicon's native GPU acceleration (MPS) for stable local training without any complex compilation setup.

---

## 🏃 Uruchamianie (Execution)

### 🪟 Windows Execution
* **Option A: Native Desktop Mode (Recommended)** - Simply double-click the **`Uruchom_Kinesiology_Analyzer.vbs`** file in the root directory. This launches a silent background Streamlit server and opens a beautiful, native PySide6 desktop window with an elegant loading splash screen.
* **Option B: Debug Mode** - Double-click **`run_debug.bat`** to run the native desktop wrapper in a visible console shell to easily view logs and troubleshoot.

### 🍎 macOS & 🐧 Linux Execution
* **Option A: Native Desktop Mode** - Open your terminal, navigate to the project directory, and run the following commands to make the launcher script executable and boot the native PySide6 desktop application window:
  ```bash
  chmod +x run_mac.sh
  ./run_mac.sh
  ```
* **Option B: Standard Web Mode (Any Platform)** - Run the Streamlit server directly to open the application in your default web browser:
  ```bash
  streamlit run pdf_parser_trainer.py
  ```

---

## 🔒 Privacy & Security

Kinesiology Analyzer Center is designed from the ground up for high-privacy environments:
* **GDPR & HIPAA Ready:** By selecting the **Local Ollama** engine, all AI data parsing, RAG searches, and model training occur 100% locally and offline on your computer. No data ever leaves your laboratory's network.
* **Built-in Safety:** The project includes a robust pre-configured `.gitignore` file that strictly excludes raw patient folders (`inbox/`, `processed/`), large fine-tuned model weights, and local debug logs, ensuring you can never accidentally commit sensitive data to public repositories.

---

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
