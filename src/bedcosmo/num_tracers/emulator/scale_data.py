"""Scale training data outputs by specified input variables.

Creates a copy of all .npz files in a directory with y values multiplied
by the product of the specified scale expressions.

Scale expressions are parsed as a single infix language: ``+ - * / ^`` or ``**``,
``()`` , ``exp(...)``, ``log(...)``, variables, and floats.

Variables: N_tracers, Om, Ok, w0, wa, hrdrag.
Precedence: ``^`` / ``**`` (right) > ``*`` / ``/`` > ``+`` ``-`` ; unary ``+`` / ``-``.
Use parentheses when needed; e.g. ``(hrdrag/exp(Om))^2`` differs from
``hrdrag/exp(Om)^2`` (the latter is ``hrdrag / (exp(Om)^2)``).

Usage:
    python scale_data.py <data_dir> <expr1> [expr2 ...]

Example:
    python scale_data.py /path/to/v3 N_tracers
    python scale_data.py /path/to/v3 1/exp(Om)
    python scale_data.py /path/to/v3 'log(N_tracers)' '1/exp(Om)' 'hrdrag^2'
    python scale_data.py /path/to/v3 'log(N_tracers)' '1/exp(Om)' 'hrdrag/log(hrdrag)'

Output directory is created alongside the input directory with suffix
_{expr1}_{expr2}_scaled. e.g.:
    /path/to/v3 -> /path/to/v3_N_tracers_scaled
    /path/to/v3 -> /path/to/v3_inv_exp_Om__scaled
"""

from __future__ import annotations

import os
import re
import sys
from typing import Any

import numpy as np

# --- scale expression parsing / evaluation (also used by test_cov_scaling.py) ---

ALLOWED_VARS = frozenset({"N_tracers", "Om", "Ok", "w0", "wa", "hrdrag"})
ALLOWED_FUNCS = frozenset({"exp", "log"})

_TOKEN_RE = re.compile(
    r"\s*(\*\*|\d+\.?\d*(?:[eE][-+]?\d+)?|[a-zA-Z_]\w*|[+\-*/^()])"
)


def _power_suffix_token(power: float) -> str:
    if power == 1.0:
        return ""
    if abs(power - round(power)) < 1e-9:
        return f"pow{int(round(power))}"
    return "pow" + f"{power:.6g}".replace("-", "m").replace(".", "p").replace("+", "")


def _tokenize(s: str) -> list[Any]:
    s = s.strip()
    if not s:
        raise ValueError("empty scale expression")
    tokens: list[Any] = []
    pos = 0
    while pos < len(s):
        m = _TOKEN_RE.match(s, pos)
        if not m:
            raise ValueError(
                f"Invalid token at position {pos} in scale expression {s!r}"
            )
        g = m.group(1)
        pos = m.end()
        if g == "**":
            tokens.append("**")
        elif g[0].isdigit() or (g[0] == "." and len(g) > 1 and g[1].isdigit()):
            tokens.append(float(g))
        else:
            tokens.append(g)
    tokens.append("EOF")
    return tokens


class _Parser:
    """Recursive-descent parser for infix scale expressions."""

    def __init__(self, tokens: list[Any]) -> None:
        self.toks = tokens
        self.i = 0

    def peek(self) -> Any:
        return self.toks[self.i]

    def consume(self, expected: str | None = None) -> Any:
        t = self.toks[self.i]
        if expected is not None and t != expected:
            raise ValueError(f"expected {expected!r}, got {t!r} in scale expression")
        self.i += 1
        return t

    def parse(self) -> tuple:
        node = self.parse_expr()
        if self.peek() != "EOF":
            raise ValueError(f"trailing junk in scale expression: {self.peek()!r}")
        return node

    def parse_expr(self) -> tuple:
        return self.parse_additive()

    def parse_additive(self) -> tuple:
        node = self.parse_multiplicative()
        while self.peek() in ("+", "-"):
            op = self.consume()
            right = self.parse_multiplicative()
            node = ("binop", op, node, right)
        return node

    def parse_multiplicative(self) -> tuple:
        node = self.parse_power()
        while self.peek() in ("*", "/"):
            op = self.consume()
            right = self.parse_power()
            node = ("binop", op, node, right)
        return node

    def parse_power(self) -> tuple:
        node = self.parse_unary()
        if self.peek() in ("**", "^"):
            self.consume()
            right = self.parse_power()
            node = ("binop", "**", node, right)
        return node

    def parse_unary(self) -> tuple:
        if self.peek() == "+":
            self.consume()
            return self.parse_unary()
        if self.peek() == "-":
            self.consume()
            return ("unary", "-", self.parse_unary())
        return self.parse_primary()

    def parse_primary(self) -> tuple:
        t = self.peek()
        if t == "(":
            self.consume("(")
            node = self.parse_expr()
            self.consume(")")
            return node
        if isinstance(t, float):
            self.consume()
            return ("lit", t)
        if isinstance(t, str) and t not in ("EOF",):
            self.consume()
            if t not in ALLOWED_VARS and t not in ALLOWED_FUNCS:
                raise ValueError(
                    f"unknown identifier {t!r} in scale expression "
                    f"(allowed vars: {sorted(ALLOWED_VARS)}, funcs: {sorted(ALLOWED_FUNCS)})"
                )
            if self.peek() == "(":
                if t not in ALLOWED_FUNCS:
                    raise ValueError(f"expected variable, got function call {t!r}")
                self.consume("(")
                arg = self.parse_expr()
                self.consume(")")
                return ("call", t, arg)
            if t in ALLOWED_FUNCS:
                raise ValueError(f"missing '(' after function {t!r}")
            return ("var", t)
        raise ValueError(f"unexpected token {t!r} in scale expression")


def _parse_infix(expr: str) -> tuple:
    return _Parser(_tokenize(expr)).parse()


def parse_scale_expression(expr: str) -> tuple:
    """Parse *expr* into an AST (tuple tree)."""
    e = expr.strip()
    if not e:
        raise ValueError("empty scale expression")
    return _parse_infix(e)


def _eval_ast(node: tuple, env: dict[str, np.ndarray | float]) -> np.ndarray | float:
    tag = node[0]
    if tag == "lit":
        return float(node[1])
    if tag == "var":
        name = node[1]
        if name not in env:
            raise KeyError(name)
        return np.asarray(env[name], dtype=float)
    if tag == "call":
        fn, child = node[1], node[2]
        arg = _eval_ast(child, env)
        if fn == "exp":
            return np.exp(arg)
        if fn == "log":
            return np.log(arg)
        raise ValueError(fn)
    if tag == "unary":
        if node[1] == "-":
            return -_eval_ast(node[2], env)
        raise ValueError(node[1])
    if tag == "binop":
        op, left, right = node[1], _eval_ast(node[2], env), _eval_ast(node[3], env)
        if op == "+":
            return left + right
        if op == "-":
            return left - right
        if op == "*":
            return left * right
        if op == "/":
            return left / right
        if op == "**":
            return np.power(left, right)
        raise ValueError(op)
    raise ValueError(f"bad AST node {node!r}")


def _vars_in_ast(node: tuple) -> set[str]:
    tag = node[0]
    if tag == "lit":
        return set()
    if tag == "var":
        return {node[1]}
    if tag == "call":
        return _vars_in_ast(node[2])
    if tag == "unary":
        return _vars_in_ast(node[2])
    if tag == "binop":
        return _vars_in_ast(node[2]) | _vars_in_ast(node[3])
    return set()


def _validate_vars(vs: set[str], expr: str) -> None:
    unknown = vs - ALLOWED_VARS
    if unknown:
        raise ValueError(f"unknown variable(s) {unknown} in {expr!r}")


def variables_in_scale_expression(expr: str) -> set[str]:
    """Names of input variables referenced by this expression."""
    ast = parse_scale_expression(expr)
    vs = _vars_in_ast(ast)
    _validate_vars(vs, expr)
    return vs


def eval_scale_expression(
    expr: str, env: dict[str, np.ndarray | float]
) -> np.ndarray | float:
    """Evaluate expression with env[var] = scalar or 1d array (per training row)."""
    ast = parse_scale_expression(expr)
    vs = _vars_in_ast(ast)
    _validate_vars(vs, expr)
    for v in vs:
        if v not in env:
            raise KeyError(f"missing value for variable {v!r} in scale expression")
    return _eval_ast(ast, env)


def _ast_to_suffix(node: tuple) -> str:
    tag = node[0]
    if tag == "var":
        return node[1]
    if tag == "lit":
        v = float(node[1])
        if abs(v - round(v)) < 1e-9:
            return str(int(round(v)))
        return f"{v:.6g}".replace("-", "m").replace(".", "p")
    if tag == "call":
        return f"{node[1]}_{_ast_to_suffix(node[2])}"
    if tag == "unary":
        if node[1] == "-":
            return "minus_" + _ast_to_suffix(node[2])
        raise ValueError(node[1])
    if tag == "binop":
        op, a, b = node[1], node[2], node[3]
        if op == "/" and a[0] == "lit" and abs(float(a[1]) - 1.0) < 1e-15:
            return "inv_" + _ast_to_suffix(b)
        if op == "**":
            if b[0] == "lit":
                tok = _power_suffix_token(float(b[1]))
                base = _ast_to_suffix(a)
                return f"{base}_{tok}" if tok else base
            return f"{_ast_to_suffix(a)}_pow_{_ast_to_suffix(b)}"
        if op == "/":
            return f"{_ast_to_suffix(a)}_div_{_ast_to_suffix(b)}"
        if op == "*":
            return f"{_ast_to_suffix(a)}_mul_{_ast_to_suffix(b)}"
        if op == "+":
            return f"{_ast_to_suffix(a)}_plus_{_ast_to_suffix(b)}"
        if op == "-":
            return f"{_ast_to_suffix(a)}_minus_{_ast_to_suffix(b)}"
        raise ValueError(f"unhandled binop {op!r} for suffix")
    raise ValueError(node)


def scale_expression_suffix(expr: str) -> str:
    """Filesystem-safe token for one scale factor (for output paths)."""
    return _ast_to_suffix(parse_scale_expression(expr))


def _ast_to_latex(node: tuple, param_latex: dict[str, str]) -> str:
    tag = node[0]
    if tag == "lit":
        v = float(node[1])
        return str(int(v)) if abs(v - round(v)) < 1e-9 else str(v)
    if tag == "var":
        name = node[1]
        if name not in param_latex:
            return r"\mathrm{" + name.replace("_", r"\_") + "}"
        return param_latex[name][1:-1]
    if tag == "call":
        fn, ch = node[1], node[2]
        inner = _ast_to_latex(ch, param_latex)
        return rf"\{fn}\left({inner}\right)"
    if tag == "unary":
        ch = node[2]
        inner = _ast_to_latex(ch, param_latex)
        if ch[0] == "binop" and ch[1] in ("+", "-"):
            inner = r"\left(" + inner + r"\right)"
        return "-" + inner
    if tag == "binop":
        op, a, b = node[1], node[2], node[3]
        la, lb = _ast_to_latex(a, param_latex), _ast_to_latex(b, param_latex)
        if op == "/":
            return rf"\frac{{{la}}}{{{lb}}}"
        if op == "**":
            return rf"\left({la}\right)^{{{lb}}}"
        sym = {"+": "+", "-": "-", "*": r"\cdot "}.get(op, op)
        return rf"\left({la}\,{sym}\,{lb}\right)"
    raise ValueError(node)


def scale_expression_latex(expr: str, param_latex: dict[str, str]) -> str:
    """Math-mode LaTeX for plot labels."""
    ast = parse_scale_expression(expr)
    body = _ast_to_latex(ast, param_latex)
    return f"${body}$"


# --- batch scaling of .npz training data ---


def scale_dataset(data_dir: str, scale_exprs: list[str]) -> str:
    data_dir = data_dir.rstrip("/")
    suffix = "_".join(scale_expression_suffix(e) for e in scale_exprs) + "_scaled"
    out_dir = f"{data_dir}_{suffix}"
    os.makedirs(out_dir, exist_ok=True)

    npz_files = sorted(f for f in os.listdir(data_dir) if f.endswith(".npz"))
    if not npz_files:
        print(f"No .npz files found in {data_dir}")
        sys.exit(1)

    print(f"Input:  {data_dir}")
    print(f"Output: {out_dir}")
    print(f"Scale expressions: {scale_exprs}")
    print(f"Files: {len(npz_files)}")

    for fname in npz_files:
        d = np.load(os.path.join(data_dir, fname))
        x = d["x"]
        y = d["y"]
        param_names = list(d["param_names"])

        scale_factor = np.ones(len(x), dtype=np.float64)
        for expr in scale_exprs:
            needed = variables_in_scale_expression(expr)
            for v in needed:
                if v not in param_names:
                    print(f"Error: '{v}' not in param_names {param_names} (in {expr!r})")
                    sys.exit(1)
            env = {name: x[:, param_names.index(name)].astype(np.float64) for name in param_names}
            scale_factor *= np.asarray(
                eval_scale_expression(expr, env), dtype=np.float64
            ).reshape(-1)

        y_scaled = y * scale_factor[:, None]

        out_path = os.path.join(out_dir, fname)
        np.savez(
            out_path,
            x=x,
            y=y_scaled,
            param_names=d["param_names"],
            target_names=d["target_names"],
        )
        print(f"  {fname}: y * {' * '.join(scale_exprs)}, shape {y.shape}")

    print(f"\nDone. Scaled data saved to {out_dir}")
    return out_dir


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(f"Usage: {sys.argv[0]} <data_dir> <expr1> [expr2 ...]")
        print(
            "Expressions: infix with + - * / ^ ** ( ) exp() log() and variables. "
            "Quote tokens in the shell if needed (e.g. 'hrdrag/log(hrdrag)')."
        )
        sys.exit(1)

    data_dir = sys.argv[1]
    scale_exprs = sys.argv[2:]

    if not os.path.isdir(data_dir):
        print(f"Error: {data_dir} is not a directory")
        sys.exit(1)

    scale_dataset(data_dir, scale_exprs)
