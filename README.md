#  IRIS-2026: Static Memory Compiler for Edge-AI Automotive Diagnostics

> **A deterministic post-training quantization and linear tensor arena allocation engine designed to deploy deep learning anomaly detection models onto memory-constrained microcontrollers.**

---

##  Performance & Optimization Summary

| Metric | Baseline (FP32 Unoptimized) | IRIS Compiled (INT8 + Arena) | Efficiency Gain / Impact |
| :--- | :--- | :--- | :--- |
| **Model Classification Accuracy** | — | **90.15%** | High Diagnostic Precision |
| **Static Weight Memory (ROM)** | 2,948 Bytes | **737 Bytes** | **75.00% Reduction** |
| **Transient Activation RAM** | 103 Bytes | **64 Bytes** | **37.86% RAM Savings** |
| **Total Hardware Footprint** | 3,051 Bytes | **801 Bytes** |  **73.75% Total Reduction** |

---

##  Pipeline Architecture & Code Structure

The project repository is structured across four sequential pipeline stages and one hardware export artifact:
### Script Overview

1. **`generate_data.py`** Simulates 10,000 operational OBD-II sensor profiles (RPM, Throttle Position, Coolant Temp, MAF, O2 Voltage) bound by physical fluid dynamics constraints to label vehicular emission anomalies.
   
2. **`train_victim.py`** Trains a 3-layer Dense Neural Network (`5 -> 32 -> 16 -> 1`) in PyTorch. Achieves 90.15% accuracy and exports the universal computational graph to `victim_model.onnx`.

3. **`analyze_graph.py`** Parses the ONNX computational graph blueprint to profile layer-by-layer shapes, operations (`Gemm`, `Relu`), and unoptimized byte footprints.

4. **`compile_engine.py`** Executes Post-Training Symmetric 8-bit Quantization ($INT8$) to reduce static parameter memory by 75%. Simultaneously tracks activation birth/death lifespans to map overlapping tensors to a shared 64-byte static `tensor_arena` memory pool.

5. **`deployed_model.h`** The final generated C++ header file containing quantized integer weight arrays and static workspace parameters ready to compile onto embedded silicon chips.

---

##  How to Reproduce

Execute the complete end-to-end pipeline in order from your terminal:

```bash
# 1. Synthesize OBD-II sensor dataset
python3 generate_data.py

# 2. Train baseline PyTorch model & export ONNX graph
python3 train_victim.py

# 3. Analyze ONNX graph execution lifecycle
python3 analyze_graph.py

# 4. Run quantization, arena allocator, and generate C++ header
python3 compile_engine.py

---

##  Step 2: Create a New Repository on GitHub

1. Go to [GitHub.com](https://github.com) and log into your account.
2. Click the **`+`** icon in the top right corner and select **New repository**.
3. **Repository name:** Type `IRIS-Edge-AI-Compiler`.
4. **Description:** Type `Static Memory Compiler for Deploying Edge-AI Models on Automotive ECUs`.
5. Keep it **Public**.
6. **Important:** Leave "Add a README file" **unchecked** (since we already created one on your Mac).
7. Click **Create repository**.

---

##  Step 3: Push Your Code from Your Mac Terminal

Open your VS Code terminal (or Mac Terminal) and run these exact commands line-by-line:

```bash
# 1. Navigate to your project directory
cd /Users/drkrm/Documents/IRIS

# 2. Initialize local Git repository
git init

# 3. Stage all your files (Python scripts, C++ header, README)
git add .

# 4. Save your initial commit
git commit -m "Initial commit: IRIS Edge AI Compiler Pipeline"

# 5. Rename the default branch to main
git branch -M main

# 6. Link your Mac folder to your GitHub repository
# (Replace YOUR_USERNAME with your actual GitHub username)
git remote add origin https://github.com/YOUR_USERNAME/IRIS-Edge-AI-Compiler.git

# 7. Push your code live!
git push -u origin main
