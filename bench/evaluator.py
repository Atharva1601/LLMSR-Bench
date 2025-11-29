# bench/evaluator.py

import sympy as sp
import numpy as np
from .numeric_eval import compute_nmse, compute_acc01
from .symbolic_eval import symbolic_equivalent


def strip_lambda(expr):
    """
    Converts:
        'lambda x: x**2 + 2*x + 1'
    into:
        'x**2 + 2*x + 1'
    """
    if expr.startswith("lambda"):
        # Remove "lambda"
        expr = expr.replace("lambda", "")
        # Remove variable list
        if ":" in expr:
            expr = expr.split(":", 1)[1]
        return expr.strip()
    return expr.strip()


def evaluate_equation_from_text(pred_text, true_text, variables, samples):
    """
    pred_text: predicted lambda or expression
    true_text: ground truth expression (no lambda prefix)
    variables: ["x"] or ["x","y"]
    samples: [{"x":1,"y":4}, ...]
    """

    # --- CLEAN UP EXPRESSIONS ---
    pred_expr = strip_lambda(pred_text)
    true_expr = strip_lambda(true_text)

    # Convert variables to SymPy symbols
    syms = [sp.symbols(v) for v in variables]

    try:
        # Sympify expressions
        pred_sym = sp.sympify(pred_expr, locals={v: sp.symbols(v) for v in variables})
        true_sym = sp.sympify(true_expr, locals={v: sp.symbols(v) for v in variables})

        # Convert to numeric functions
        f_pred = sp.lambdify(syms, pred_sym, "numpy")
        f_true = sp.lambdify(syms, true_sym, "numpy")

        # Prepare sample points
        xs = np.array([[s[v] for v in variables] for s in samples], dtype=float)
        ys_true = np.array([s["y"] for s in samples], dtype=float)

        ys_pred = []
        for row in xs:
            if len(variables) == 1:
                ys_pred.append(float(f_pred(row[0])))
            else:
                ys_pred.append(float(f_pred(*row)))

        ys_pred = np.array(ys_pred)

    except Exception as e:
        # print("Eval error:", e)   # optional debug
        return {"numeric": False, "symbolic": False, "nmse": float("inf"), "acc01": 0.0}

    # --- METRICS ---
    nmse = compute_nmse(ys_true, ys_pred)
    acc01 = compute_acc01(ys_true, ys_pred)

    numeric_pass = (nmse < 1.0) and (acc01 > 0.5)
    symbolic_pass = symbolic_equivalent(pred_expr, true_expr, variables)

    return {
        "numeric": numeric_pass,
        "symbolic": symbolic_pass,
        "nmse": nmse,
        "acc01": acc01,
    }
