🧮 LLM-SRBench: Symbolic Regression Benchmarking with LLMs

A comparative evaluation of modern Large Language Models and evolutionary Symbolic Regression techniques

📌 Overview

This project implements a complete symbolic regression benchmarking pipeline, inspired by the research paper:
“LLM-SRBench: A New Benchmark for Evaluating LLMs on Symbolic Regression” (ICML 2025 submission)
The goal is to evaluate how different reasoning methods and LLMs recover underlying mathematical expressions from small numeric samples.

You benchmark:

Models
TinyLlama-1.1B-Chat
Phi-2 (Microsoft)
GPT-2 XL

Methods
Direct prompting
Chain-of-Thought (CoT) prompting
LLMSR (LLM-guided symbolic regression, your custom implementation)
LaSR-v2 (a pure-Python evolutionary symbolic regression engine)

Datasets
Synthetic equations (20 equations auto-generated)
Transform equations (physics & math equations from ltransform dataset)

All evaluations are stored in results.csv, and visualized using multiple analytical plots.

🏆 Key Features

✔ 1. Per-model dynamic quantized loading

FP16 for small models
8-bit quantization + CPU offload (bitsandbytes) for larger models
Automatically frees VRAM after each model run

✔ 2. Fully reproducible symbolic regression evaluation

Each method returns a Python λ-function which is evaluated numerically and symbolically:
NMSE (Normalized MSE)
acc@0.1 (accuracy within 0.1 error)
boolean symbolic equivalence

✔ 3. Custom LLM-SR Implementation

Your pipeline includes a real LLMSR algorithm, combining:
evolutionary search
numerical curve fitting
local optimization
LLM-based prior proposal

✔ 4. LaSR-v2 (your custom genetic symbolic regressor)

Lightweight pure-Python SGA symbolic regression engine with:
mutation
crossover
elitism
numerical constant tuning
expression simplification

✔ 5. End-to-end Benchmarking & Visualization
You generate:
Model × Dataset accuracy
Method × Model accuracy
Method × Dataset accuracy
Method × Dataset × Model (3-axis comparison)
Comparison vs paper benchmark values
All plots are formatted for publication-level clarity.


📊 Final Results Summary

🥇 Best Method (overall)
Method	Accuracy (acc@0.1)
LLMSR	≈ 66%
LaSR-v2	46.6%
Direct	~5%
CoT	~1%

🥇 Best Model (overall)
Model	Accuracy
TinyLlama-1.1B	Highest overall accuracy
Phi-2	moderate
GPT-2 XL	close behind

📌 Synthetic vs Transform Dataset
LLMSR excels on synthetic equations (81%),
Drops significantly on transform equations (6–7%) due to multivariable physics formulas.
LaSR scores 100% on transform for some models (trivial expressions), but lower on synthetic.

🆚 Paper Comparison

Method:LLMSR ----Paper accuracy:~67%  Project Accuracy:~66%
Method:LaSR ----Paper accuracy:~63%  Project Accuracy:~47%

Implementation slightly matches paper results because:
Simpler datasets was used
Synthetic equations are easier to recover
Applied numerical tuning + simplification


📚 Reference

This work is inspired by:
LLM-SRBench: A New Benchmark for Scientific Equation Discovery with Large Language Models by
Parshin Shojaee, Ngoc-Hieu Nguyen, Kazem Meidani, Amir Barati Farimani, Khoa D Doan, Chandan K. Reddy
ICML 2025
