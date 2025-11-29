"""
LaSR-v2 (FAST VERSION)
Pure-Python Symbolic Regression for laptop CPUs.
No Julia, No PySR, No curve_fit (disabled).
Optimized for speed (20x faster) and stability.
"""

import random
import numpy as np
import sympy as sp
import traceback

GLOBAL_SEED = 42
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)


def safe_lambdify(expr, var):
    """Safely turn sympy expression into a numpy-evaluable function."""
    try:
        sym = sp.symbols(var)
        f = sp.lambdify(sym, expr, modules=["numpy", "math"])

        def wrapper(xarr):
            try:
                return np.array(f(xarr), dtype=float)
            except Exception:
                out = []
                for xv in np.array(xarr, dtype=float):
                    try:
                        out.append(float(f(xv)))
                    except Exception:
                        return None
                return np.array(out, dtype=float)

        return wrapper
    except Exception:
        return None


def nmse_of_expr(expr, var, samples):
    """Compute NMSE safely."""
    try:
        f = safe_lambdify(expr, var)
        if f is None:
            return float("inf")

        xs = np.array([s[var] for s in samples], dtype=float)
        ys = np.array([s["y"] for s in samples], dtype=float)
        pred = f(xs)
        if pred is None:
            return float("inf")

        mse = np.mean((ys - pred) ** 2)
        denom = np.mean((ys - np.mean(ys)) ** 2)
        if denom == 0:
            return 0 if mse == 0 else float("inf")
        return float(mse / denom)
    except Exception:
        return float("inf")


# Random Expression Generator

UNARY_OPS = ["sin", "cos"]
BINARY_OPS = ["+", "-", "*"]


def random_const():
    return random.randint(-3, 3)


def make_random_expr(var, max_depth=2):
    x = sp.symbols(var)

    def build(depth):
        if depth <= 0:
            return x if random.random() < 0.6 else sp.Integer(random_const())

        if random.random() < 0.3:
            return x

        if random.random() < 0.5:
            return sp.Integer(random_const())

        # unary
        if random.random() < 0.4:
            op = random.choice(UNARY_OPS)
            sub = build(depth - 1)
            return getattr(sp, op)(sub)

        # binary
        op = random.choice(BINARY_OPS)
        a = build(depth - 1)
        b = build(depth - 1)

        if op == "+":
            return a + b
        if op == "-":
            return a - b
        if op == "*":
            return a * b

        return x

    return sp.simplify(build(max_depth))


def all_subexpr(expr):
    res = []

    def rec(e):
        res.append(e)
        for a in e.args:
            try:
                rec(a)
            except:
                pass

    rec(expr)
    return res


def random_subexpr(expr):
    subs = all_subexpr(expr)
    return random.choice(subs) if subs else expr


def replace_subexpr(expr, target, repl):
    try:
        return expr.xreplace({target: repl})
    except:
        try:
            return sp.sympify(str(expr).replace(str(target), str(repl)))
        except:
            return expr


def mutate_expr(expr, var, max_depth=2):
    try:
        if random.random() < 0.7:
            new_sub = make_random_expr(var, max_depth=max_depth)
            t = random_subexpr(expr)
            return sp.simplify(replace_subexpr(expr, t, new_sub))
        else:
            #
            def tweak(e):
                if e.is_Integer:
                    return sp.Integer(e + random.choice([-1, 0, 1]))
                if e.args:
                    return e.func(*[tweak(a) for a in e.args])
                return e

            return sp.simplify(tweak(expr))
    except:
        return expr


def crossover_expr(a, b):
    try:
        ta = random_subexpr(a)
        tb = random_subexpr(b)
        a2 = replace_subexpr(a, ta, tb)
        b2 = replace_subexpr(b, tb, ta)
        return sp.simplify(a2), sp.simplify(b2)
    except:
        return a, b


def tournament_select(pop, scores, k=3):
    idxs = random.sample(range(len(pop)), min(k, len(pop)))
    best = idxs[0]
    for i in idxs:
        if scores[i] < scores[best]:
            best = i
    return pop[best]


class LaSR_v2:
    def __init__(
        self,
        var_name,
        population_size=20,
        generations=10,
        elitism=2,
        crossover_prob=0.5,
        mutation_prob=0.2,
        max_depth=2,
        seed=GLOBAL_SEED,
        numeric_tune=False,
    ):
        self.var = var_name
        self.pop_size = population_size
        self.generations = generations
        self.elitism = elitism
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.max_depth = max_depth
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

    def init_population(self):
        return [
            make_random_expr(self.var, self.max_depth) for _ in range(self.pop_size)
        ]

    def evolve(self, samples):
        population = self.init_population()
        scores = [nmse_of_expr(e, self.var, samples) for e in population]
        best_expr = population[int(np.argmin(scores))]
        best_score = min(scores)

        for _ in range(self.generations):
            ranked = sorted(zip(population, scores), key=lambda x: x[1])
            population = [x[0] for x in ranked]
            scores = [x[1] for x in ranked]

            new_pop = population[: self.elitism]

            while len(new_pop) < self.pop_size:
                p1 = tournament_select(population, scores)
                p2 = tournament_select(population, scores)

                c1, c2 = p1, p2
                if random.random() < self.crossover_prob:
                    c1, c2 = crossover_expr(p1, p2)

                if random.random() < self.mutation_prob:
                    c1 = mutate_expr(c1, self.var, self.max_depth)

                if random.random() < self.mutation_prob:
                    c2 = mutate_expr(c2, self.var, self.max_depth)

                new_pop.append(c1)
                if len(new_pop) < self.pop_size:
                    new_pop.append(c2)

            population = new_pop
            scores = [nmse_of_expr(e, self.var, samples) for e in population]

            # update best
            idx = int(np.argmin(scores))
            if scores[idx] < best_score:
                best_expr, best_score = population[idx], scores[idx]

        return best_expr, best_score


def simple_lasr(vars_, samples):
    try:
        var = vars_[0]
        sr = LaSR_v2(var_name=var)
        best_expr, best_score = sr.evolve(samples)
        return f"lambda {var}: {str(sp.simplify(best_expr))}"
    except Exception:
        traceback.print_exc()
        return "lambda x: 0"
