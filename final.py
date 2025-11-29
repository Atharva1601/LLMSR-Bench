# final.py
"""
SRBench — Final quantized-safe runner (per-model load)
- Models:
    TinyLlama/TinyLlama-1.1B-Chat-v1.0 -> fp16
    microsoft/phi-2                    -> 8-bit with fp32 CPU offload
    gpt2-xl                            -> 8-bit with fp32 CPU offload
- Loads each model, runs all equations for that model, then unloads model to free VRAM.
- Writes CSV only: results.csv
- Robust: method init/run failures fallback to "lambda x: 0"
- Deterministic: random.seed(42)
"""

import os
import sys
import csv
import time
import random
import traceback

import sympy as sp
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from methods.direct.direct import DirectMethod
from methods.cot.cot import CoTMethod
from methods.llmsr.llmsr import LLMSRMethod
from methods.lasr.lasr import simple_lasr
from bench.evaluator import evaluate_equation_from_text

# Deterministic behavior
random.seed(42)

# ---------------------------
# Models and quant preferences
# ---------------------------
MODEL_LIST = [
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "microsoft/phi-2",
    "gpt2-xl",
]

MODEL_QUANT_PREF = {
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0": "fp16",
    "microsoft/phi-2": "8bit_offload",
    "gpt2-xl": "8bit_offload",
}

# BitsAndBytes configs
BNB_CFG_8BIT_OFFLOAD = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
    llm_int8_enable_fp32_cpu_offload=True,
)


# ---------------------------------------------------------
# Utilities
# ---------------------------------------------------------
def progress_bar(current, total, bar_length=30):
    if total <= 0:
        return
    percent = float(current) / float(total)
    arrow = "=" * int(round(percent * bar_length))
    spaces = " " * (bar_length - len(arrow))
    sys.stdout.write(f"\r[{arrow}{spaces}] {int(percent * 100)}%")
    sys.stdout.flush()


def load_transform(path, n=5):
    if not os.path.exists(path):
        print(f"[WARN] Missing transform dataset: {path}")
        return []

    import json

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    final = []
    for d in data[:n]:
        expr = d.get("expression", "0")
        symbols = d.get("symbols", ["x"])
        main = symbols[0]
        sym = sp.symbols(main)

        try:
            expr_sp = sp.sympify(expr, locals={main: sym})
        except:
            expr_sp = sym * 0

        samples = []
        for x in [1, 2, 3]:
            try:
                y = float(expr_sp.subs(sym, x))
            except:
                y = 0.0
            samples.append({main: float(x), "y": y})

        final.append({"expr": expr, "samples": samples})

    return final


def generate_synth(n=20):
    eqs = []
    for _ in range(n):
        a = random.randint(1, 5)
        b = random.randint(-4, 4)
        c = random.randint(-5, 5)

        form = random.choice(["lin", "quad", "mix"])

        if form == "lin":
            expr = f"{a}*x + {b}"
        elif form == "quad":
            expr = f"{a}*x**2 + {b}*x + {c}"
        else:
            expr = f"{a}*x**3 + {b}*x + {c}"

        sym = sp.symbols("x")
        expr_sp = sp.sympify(expr)

        samples = []
        for x in [1, 2, 3]:
            y = float(expr_sp.subs(sym, x))
            samples.append({"x": float(x), "y": y})

        eqs.append({"expr": expr, "samples": samples})

    return eqs


# ---------------------------------------------------------
# Load individual model safely
# ---------------------------------------------------------
def load_model_for_benchmark(model_name):
    tok = None
    model = None
    err = None
    loaded = False
    pref = MODEL_QUANT_PREF.get(model_name, "fp16")

    print(f"\n[LOAD] Loading {model_name} with pref={pref}")

    try:
        tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    except Exception as e:
        err = f"Tokenizer failed: {e}"
        print(err)
        return tok, model, loaded, err

    try:
        if pref == "fp16":
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True,
            )
        else:
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto",
                quantization_config=BNB_CFG_8BIT_OFFLOAD,
                trust_remote_code=True,
            )

        loaded = True
        print(f"[LOAD] Loaded {model_name}")

    except Exception as e:
        err = f"Model failed: {e}"
        print(err)

    return tok, model, loaded, err


# ---------------------------------------------------------
# Initialize methods (using preloaded model if possible)
# ---------------------------------------------------------
def try_init_method(cls, model_name, model, tok):
    try:
        inst = cls(model_name, model=model, tokenizer=tok)
        return inst
    except TypeError:
        return cls(model_name)
    except Exception:
        return None


# ---------------------------------------------------------
# Run all equations for one model
# ---------------------------------------------------------
def run_all_for_model(model_name, tok, model, equations, csv_rows):
    print(f"\n===== RUNNING {model_name} =====")

    D = try_init_method(DirectMethod, model_name, model, tok)
    C = try_init_method(CoTMethod, model_name, model, tok)
    L = try_init_method(LLMSRMethod, model_name, model, tok)

    total = len(equations)
    for idx, eq in enumerate(equations, 1):
        expr = eq["expr"]
        samples = eq["samples"]

        vars_ = list(samples[0].keys())
        vars_.remove("y")

        print(f"\n--- Equation {idx}/{total}: {expr} ---")
        progress_bar(idx, total)

        # DIRECT
        try:
            p1 = D.run(vars_, samples) if D else "lambda x: 0"
        except:
            p1 = "lambda x: 0"
        s1 = evaluate_equation_from_text(p1, expr, vars_, samples)

        # COT
        try:
            p2 = C.run(vars_, samples) if C else "lambda x: 0"
        except:
            p2 = "lambda x: 0"
        s2 = evaluate_equation_from_text(p2, expr, vars_, samples)

        # LLMSR
        try:
            p3 = L.run(vars_, samples) if L else "lambda x: 0"
        except:
            p3 = "lambda x: 0"
        s3 = evaluate_equation_from_text(p3, expr, vars_, samples)

        # LASR
        try:
            p4 = simple_lasr(vars_, samples)
        except:
            p4 = "lambda x: 0"
        s4 = evaluate_equation_from_text(p4, expr, vars_, samples)

        # Append to CSV
        csv_rows.extend(
            [
                [
                    model_name,
                    f"eq_{idx}",
                    expr,
                    "direct",
                    s1["numeric"],
                    s1["symbolic"],
                    s1["nmse"],
                    s1["acc01"],
                ],
                [
                    model_name,
                    f"eq_{idx}",
                    expr,
                    "cot",
                    s2["numeric"],
                    s2["symbolic"],
                    s2["nmse"],
                    s2["acc01"],
                ],
                [
                    model_name,
                    f"eq_{idx}",
                    expr,
                    "llmsr",
                    s3["numeric"],
                    s3["symbolic"],
                    s3["nmse"],
                    s3["acc01"],
                ],
                [
                    model_name,
                    f"eq_{idx}",
                    expr,
                    "lasr",
                    s4["numeric"],
                    s4["symbolic"],
                    s4["nmse"],
                    s4["acc01"],
                ],
            ]
        )


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
if __name__ == "__main__":
    synth = generate_synth(20)
    transform = load_transform("datasets/ltransform_dataset.json", 5)
    all_eq = synth + transform

    csv_rows = []

    for model_name in MODEL_LIST:
        tok, model, loaded, err = load_model_for_benchmark(model_name)

        if not loaded:
            print(f"[WARN] Model {model_name} did not preload correctly: {err}")
            tok = None
            model = None

        run_all_for_model(model_name, tok, model, all_eq, csv_rows)

        # Free VRAM
        try:
            del model
        except:
            pass
        torch.cuda.empty_cache()

    # Write CSV
    with open("results.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "model",
                "equation",
                "expr",
                "method",
                "numeric",
                "symbolic",
                "nmse",
                "acc01",
            ]
        )
        writer.writerows(csv_rows)

    print("\n===== DONE — CSV saved → results.csv =====")
