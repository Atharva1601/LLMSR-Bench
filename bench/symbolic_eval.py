# bench/symbolic_eval.py
import sympy as sp


def _to_sympy(expr_text, variables):
    """
    Accept expression as string and convert to a sympy expression.
    Returns sympy expression or raises.
    """
    # sympify with local symbols to avoid creating new names
    syms = {v: sp.symbols(v) for v in variables}
    return sp.sympify(expr_text, locals=syms)


def symbolic_equivalent(pred_text, true_text, variables):
    """
    Compare two expressions provided as strings.
    Returns True if algebraically equivalent (according to sympy simplify).
    """
    try:
        pred = _to_sympy(pred_text, variables)
        true = _to_sympy(true_text, variables)
        diff = sp.simplify(pred - true)
        # if difference simplifies to 0, they are equivalent
        return bool(diff == 0)
    except Exception:
        # fallback: try numeric checks on some random points
        try:
            import random
            import math

            syms = {v: sp.symbols(v) for v in variables}
            pred = _to_sympy(pred_text, variables)
            true = _to_sympy(true_text, variables)
            f_pred = sp.lambdify(list(syms.values()), pred, "math")
            f_true = sp.lambdify(list(syms.values()), true, "math")
            for _ in range(6):
                pts = [random.uniform(1.0, 5.0) for _ in variables]
                p = f_pred(*pts)
                t = f_true(*pts)
                if math.isfinite(p) and math.isfinite(t):
                    if abs(p - t) > 1e-4 * (abs(t) + 1e-6):
                        return False
            return True
        except Exception:
            return False
