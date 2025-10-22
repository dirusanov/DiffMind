from __future__ import annotations

import ast
import math
from typing import Any

from ..base import Tool, ToolContext


class CalculatorTool(Tool):
    name = "calculator"
    description = "Evaluate a simple arithmetic expression. Supported: +, -, *, /, ^, parentheses, sqrt, log, sin, cos."

    def run(self, ctx: ToolContext, tool_input: str) -> str:  # noqa: ARG002
        try:
            value = _safe_eval(tool_input)
            return f"{value}"
        except Exception as e:
            return f"error: {e}"


_ALLOWED_FUNCS = {
    "sqrt": math.sqrt,
    "log": math.log,
    "sin": math.sin,
    "cos": math.cos,
}


class _Eval(ast.NodeVisitor):
    def visit_Expr(self, node: ast.Expr) -> Any:  # pragma: no cover - trivial
        return self.visit(node.value)

    def visit_BinOp(self, node: ast.BinOp) -> Any:
        left = self.visit(node.left)
        right = self.visit(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Pow):
            return left ** right
        raise ValueError("unsupported operator")

    def visit_UnaryOp(self, node: ast.UnaryOp) -> Any:
        operand = self.visit(node.operand)
        if isinstance(node.op, ast.USub):
            return -operand
        if isinstance(node.op, ast.UAdd):
            return +operand
        raise ValueError("unsupported unary operator")

    def visit_Call(self, node: ast.Call) -> Any:
        if not isinstance(node.func, ast.Name):
            raise ValueError("unsupported function")
        name = node.func.id
        if name not in _ALLOWED_FUNCS:
            raise ValueError("unknown function")
        args = [self.visit(a) for a in node.args]
        return _ALLOWED_FUNCS[name](*args)

    def visit_Name(self, node: ast.Name) -> Any:
        if node.id in {"pi", "e"}:
            return getattr(math, node.id)
        raise ValueError("name not allowed")

    def visit_Constant(self, node: ast.Constant) -> Any:  # noqa: D401
        if isinstance(node.value, (int, float)):
            return node.value
        raise ValueError("constant not allowed")

    def generic_visit(self, node: ast.AST) -> Any:  # pragma: no cover - safety
        raise ValueError("syntax not allowed")


def _safe_eval(expr: str) -> Any:
    tree = ast.parse(expr.replace("^", "**"), mode="eval")
    return _Eval().visit(tree.body)

