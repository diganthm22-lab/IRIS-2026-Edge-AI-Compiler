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
   
2. **`train_victim.py`** Trains a 3-layer Dense Neural Network (`5 -> 32 -> 16 -> 1`) in PyTorch. Achieves 94.15% accuracy and exports the universal computational graph to `victim_model.onnx`.

3. **`analyze_graph.py`** Parses the ONNX computational graph blueprint to profile layer-by-layer shapes, operations (`Gemm`, `Relu`), and unoptimized byte footprints.

4. **`compile_engine.py`** Executes Post-Training Symmetric 8-bit Quantization ($INT8$) to reduce static parameter memory by 75%. Simultaneously tracks activation birth/death lifespans to map overlapping tensors to a shared 64-byte static `tensor_arena` memory pool.

5. **`deployed_model.h`** The final generated C++ header file containing quantized integer weight arrays and static workspace parameters ready to compile onto embedded silicon chips.

---

