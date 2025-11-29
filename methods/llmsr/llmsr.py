# methods/llmsr/llmsr.py
"""
LLMSRMethod (iterative LLM-driven symbolic regression)

Implements an iterative search similar to the paper:
 - Generate initial candidate equations with the LLM
 - Evaluate numeric error (NMSE) over samples
 - Keep top-k candidates
 - Mutate candidates using the LLM (prompting for variations)
 - Optimize numeric constants (BFGS/curve_fit if available, else linear fit)
 - Iterate for max_iters and return best candidate as "lambda var: expr"

Constructor:
    LLMSRMethod(model_name, model=None, tokenizer=None, max_iters=5, init_k=8, keep_k=4)

Method:
    run(variables, samples) -> returns string like "lambda x: x**2 + 3"
"""

from typing import List, Tuple
import math
import random
import time
import traceback

import sympy as sp
import numpy as np


try:
    from scipy.optimize import curve_fit

    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def extract_lambda(text: str) -> str:
    lines = [l.strip() for l in text.splitlines() if "lambda" in l]
    if lines:
        return lines[-1]

    if "return" in text:
        parts = text.split("return")
        return "lambda x: " + parts[-1].strip().strip("`'\"")
    return ""


def prompt_initial_candidates(tokenizer, model, variables, samples, n=8, device="cpu"):
    """
    Ask LLM to generate n candidate lambdas. Returns list of lambda strings.
    Uses a compact prompt; deterministic (do_sample=False).
    """
    var = variables[0]
    sample_str = "\n".join([f"{s[var]} -> {s['y']}" for s in samples])
    system = (
        "You are an assistant that returns only Python lambda functions mapping the variable to y.\n"
        "Return only expressions like: lambda x: x**2 + 3*x + 1\n"
        "Do not output extra text."
    )
    prompt = (
        system + "\n\n"
        f"Data:\n{sample_str}\n\n"
        f"Generate {n} diverse candidate lambdas (one per line)."
    )

    inp = tokenizer(prompt, return_tensors="pt", truncation=True)
    inp = {k: v.to(model.device) for k, v in inp.items()}
    out = model.generate(
        **inp, max_new_tokens=256, do_sample=False, num_return_sequences=1
    )
    text = tokenizer.decode(out[0], skip_special_tokens=True)

    lines = [l.strip() for l in text.splitlines() if "lambda" in l]

    if len(lines) < n:
        tokens = text.replace(";", "\n").splitlines()
        lines = [l.strip() for l in tokens if "lambda" in l]

    seen = set()
    candidates = []
    for l in lines:
        if l not in seen:
            seen.add(l)
            candidates.append(l)

    while len(candidates) < n:
        deg = random.choice([1, 2, 3])
        a = random.randint(1, 5)
        b = random.randint(-3, 3)
        if deg == 1:
            s = f"lambda {var}: {a}*{var} + {b}"
        elif deg == 2:
            s = f"lambda {var}: {a}*{var}**2 + {b}*{var} + {random.randint(-3, 3)}"
        else:
            s = f"lambda {var}: {a}*{var}**3 + {b}*{var} + {random.randint(-3, 3)}"
        if s not in seen:
            seen.add(s)
            candidates.append(s)
    return candidates[:n]


def mutate_candidate_with_llm(
    tokenizer, model, candidate: str, variables, samples, n_variants=3
):
    """
    Ask the LLM to produce mutated variants of 'candidate'.
    Returns list of lambda strings.
    """
    var = variables[0]
    sample_str = "\n".join([f"{s[var]} -> {s['y']}" for s in samples])
    prompt = (
        "You are an assistant that suggests improved symbolic expressions.\n"
        "Input candidate:\n" + candidate + "\n\n"
        f"Data:\n{sample_str}\n\n"
        "Produce up to {n} mutated candidate lambdas (one per line) that fix errors or simplify.\n"
        "Return only lambdas like: lambda x: x**2 + 3*x + 1"
    ).replace("{n}", str(n_variants))
    inp = tokenizer(prompt, return_tensors="pt", truncation=True)
    inp = {k: v.to(model.device) for k, v in inp.items()}
    try:
        out = model.generate(**inp, max_new_tokens=200, do_sample=False)
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        lines = [l.strip() for l in text.splitlines() if "lambda" in l]

        if not lines:
            tokens = text.replace(";", "\n").splitlines()
            lines = [l.strip() for l in tokens if "lambda" in l]

        seen = set()
        res = []
        for l in lines:
            if l not in seen:
                seen.add(l)
                res.append(l)
        return res
    except Exception:
        return []


def eval_candidate_numeric(expr_str: str, variables, samples) -> float:
    """
    Compute NMSE (normalized MSE) of candidate expression on samples.
    Return +inf on failure.
    """
    var = variables[0]
    try:
        if "lambda" in expr_str:
            parts = expr_str.split(":", 1)
            expr = parts[1].strip()
        else:
            expr = expr_str

        sym = sp.symbols(var)
        expr_sp = sp.sympify(expr, locals={var: sym})
        f = sp.lambdify(sym, expr_sp, modules=["numpy", "math"])
        xs = np.array([s[var] for s in samples], dtype=float)
        ys = np.array([s["y"] for s in samples], dtype=float)
        ys_pred = f(xs)
        ys_pred = np.array(ys_pred, dtype=float)
        mse = np.mean((ys - ys_pred) ** 2)
        denom = np.mean((ys - np.mean(ys)) ** 2)
        nmse = float(mse / denom) if denom != 0 else (0.0 if mse == 0 else float("inf"))
        return nmse
    except Exception:
        return float("inf")


def numeric_tune_constants(expr_str: str, variables, samples):
    """
    Try to optimize numeric constants in expr_str.
    If scipy.curve_fit available, use it.
    Otherwise try simple linear solve for linear-in-params forms.
    Returns tuned expression string (may be same).
    """
    var = variables[0]

    try:
        if "lambda" in expr_str:
            expr = expr_str.split(":", 1)[1].strip()
        else:
            expr = expr_str
        sym = sp.symbols(var)
        expr_sp = sp.sympify(expr, evaluate=False)
    except Exception:
        return expr_str

    tokens = []
    repl_map = {}
    expr_text = str(expr_sp)

    def replace_numbers(e, params):
        if e.is_Number:
            idx = len(params)
            pname = f"p{idx}"
            params.append(float(e))
            return sp.Symbol(pname), params
        elif e.args:
            new_args = []
            for a in e.args:
                na, params = replace_numbers(a, params)
                new_args.append(na)
            return e.func(*new_args), params
        else:
            return e, params

    try:
        new_expr, params = replace_numbers(expr_sp, [])
    except Exception:
        return expr_str

    if len(params) == 0:
        # nothing to tune
        return expr_str

    # Build numeric function for curve_fit if available
    xs = np.array([s[var] for s in samples], dtype=float)
    ys = np.array([s["y"] for s in samples], dtype=float)

    if SCIPY_AVAILABLE:
        params_syms = [sp.Symbol(f"p{i}") for i in range(len(params))]
        free_syms = [sym] + params_syms
        try:
            f_lambda = sp.lambdify(free_syms, new_expr, modules=["numpy", "math"])
        except Exception:
            return expr_str

        def curvefun(x, *pvals):
            return np.array(f_lambda(x, *pvals), dtype=float)

        try:
            p0 = params
            popt, _ = curve_fit(curvefun, xs, ys, p0=p0, maxfev=5000)

            tuned_expr = new_expr
            for i, v in enumerate(popt):
                tuned_expr = tuned_expr.xreplace({sp.Symbol(f"p{i}"): sp.N(v)})
            # return lambda string
            return f"lambda {var}: {str(sp.simplify(tuned_expr))}"
        except Exception:
            return expr_str
    else:
        # simple fallback: attempt linear regression if expression is linear in params
        # Attempt to convert expression to linear combination of params: expr = A(x)*p0 + B(x)*p1 + C(x)
        # This fallback is conservative and only handles linear paramization
        try:
            p_syms = [sp.Symbol(f"p{i}") for i in range(len(params))]
            coeffs = []

            A = []
            for x_val in xs:
                vals = []
                for p in p_syms:
                    deriv = sp.diff(new_expr, p)
                    deriv_f = sp.lambdify(sym, deriv, modules=["math", "numpy"])
                    vals.append(float(deriv_f(x_val)))
                const_part = float(
                    sp.lambdify(
                        sym,
                        new_expr.subs({p_syms[i]: 0 for i in range(len(p_syms))}),
                        modules=["math", "numpy"],
                    )(x_val)
                )
                A.append(vals + [const_part])
            A = np.array(A)

            Y = ys

            if A.shape[0] < A.shape[1]:
                return expr_str
            sol, *_ = np.linalg.lstsq(A, Y, rcond=None)
            tuned_expr = new_expr
            for i, v in enumerate(sol[:-1]):
                tuned_expr = tuned_expr.xreplace({p_syms[i]: sp.N(v)})
            tuned_expr = (
                tuned_expr.xreplace({p_syms[-1]: sp.N(sol[-1])})
                if len(sol) > 0
                else tuned_expr
            )
            return f"lambda {var}: {str(sp.simplify(tuned_expr))}"
        except Exception:
            return expr_str


class LLMSRMethod:
    def __init__(
        self, model_name, model=None, tokenizer=None, max_iters=5, init_k=8, keep_k=4
    ):
        self.model_name = model_name
        self.model = model
        self.tokenizer = tokenizer
        self.max_iters = max_iters
        self.init_k = init_k
        self.keep_k = keep_k

        if self.tokenizer is None or self.model is None:
            try:
                print(f"[LLMSR] Loading model/tokenizer fallback for {model_name}")
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name, trust_remote_code=True
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    trust_remote_code=True,
                )
            except Exception as e:
                print(f"[LLMSR][ERROR] fallback load failed: {e}")
                self.model = None
                self.tokenizer = None

    def run(self, variables, samples) -> str:
        var = variables[0]
        if self.model is None or self.tokenizer is None:
            return "lambda x: 0"

        candidates = prompt_initial_candidates(
            self.tokenizer, self.model, variables, samples, n=self.init_k
        )

        scored = []
        for c in candidates:
            nmse = eval_candidate_numeric(c, variables, samples)
            scored.append((c, nmse))
        scored.sort(key=lambda x: x[1])

        best_expr, best_score = scored[0]
        print(f"[LLMSR] Initial best: {best_expr} (nmse={best_score})")

        for it in range(self.max_iters):
            topk = [c for c, s in scored[: self.keep_k]]

            tuned = []
            for c in topk:
                tuned_c = numeric_tune_constants(c, variables, samples)
                nm = eval_candidate_numeric(tuned_c, variables, samples)
                tuned.append((tuned_c, nm))

            combined = scored + tuned

            variants = []
            for c in topk:
                muts = mutate_candidate_with_llm(
                    self.tokenizer, self.model, c, variables, samples, n_variants=3
                )
                for m in muts:
                    nm = eval_candidate_numeric(m, variables, samples)
                    variants.append((m, nm))

            all_candidates = combined + variants

            unique = {}
            for expr, nmse in all_candidates:
                if expr not in unique or nmse < unique[expr]:
                    unique[expr] = nmse
            scored = sorted(list(unique.items()), key=lambda x: x[1])
            current_best, current_best_score = scored[0]
            print(
                f"[LLMSR] Iter {it + 1}: best={current_best} (nmse={current_best_score})"
            )
            if current_best_score < best_score:
                best_expr, best_score = current_best, current_best_score
            # early stop
            if best_score == 0 or best_score < 1e-6:
                break

        # final numeric tuning
        final_tuned = numeric_tune_constants(best_expr, variables, samples)
        final_nmse = eval_candidate_numeric(final_tuned, variables, samples)
        print(f"[LLMSR] Final: {final_tuned} (nmse={final_nmse})")

        if "lambda" not in final_tuned:
            final_tuned = f"lambda {var}: {final_tuned}"
        return final_tuned
