import streamlit as st
import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)

TRANSFORMATIONS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)

TRIGONOMETRIC = {
    sp.sin, sp.cos, sp.tan, sp.cot, sp.sec, sp.csc,
    sp.asin, sp.acos, sp.atan, sp.acot, sp.asec, sp.acsc,
}

HYPERBOLIC = {
    sp.sinh, sp.cosh, sp.tanh, sp.coth, sp.sech, sp.csch,
    sp.asinh, sp.acosh, sp.atanh, sp.acoth, sp.asech, sp.acsch,
}

ROUNDING = {sp.floor, sp.ceiling}

LOCAL_DICT = {
    name: getattr(sp, name)
    for name in dir(sp)
    if not name.startswith("_")
}

LOCAL_DICT.update({
    "ln": sp.log,
    "ceiling": sp.ceiling,
    "ceil": sp.ceiling,
    "Abs": sp.Abs,
    "abs": sp.Abs,
    "sqrt": sp.sqrt,
    "I": sp.I,
    "pi": sp.pi,
    "E": sp.E,
    "oo": sp.oo,
    "zoo": sp.zoo,
    "nan": sp.nan,
})


def parse_expression(expression_str):
    if not isinstance(expression_str, str) or not expression_str.strip():
        raise ValueError("Please enter a mathematical expression.")

    return parse_expr(
        expression_str.strip(),
        transformations=TRANSFORMATIONS,
        local_dict=LOCAL_DICT.copy(),
    )


def classify_algebraic(expr):
    symbols = sorted(expr.free_symbols, key=str)

    try:
        is_poly = (
            expr.is_polynomial(*symbols)
            if symbols else expr.is_polynomial()
        )

        if is_poly:
            if not symbols:
                return "Constant"
            if len(symbols) > 1:
                return "Multivariable Polynomial"

            degree = sp.degree(expr, symbols[0])
            if degree == 1:
                return "Linear"
            if degree == 2:
                return "Quadratic"
            if degree == 3:
                return "Cubic"
            return "Polynomial"

        is_rational = (
            expr.is_rational_function(*symbols)
            if symbols else expr.is_rational_function()
        )
        if is_rational:
            return "Rational"

    except (TypeError, ValueError, NotImplementedError):
        pass

    return None


def identify_function_types(expression_str):
    try:
        expr = parse_expression(expression_str)
        categories = set()
        nodes = list(sp.preorder_traversal(expr))

        if any(node.func in TRIGONOMETRIC for node in nodes):
            categories.add("Trigonometric")

        if any(node.func in HYPERBOLIC for node in nodes):
            categories.add("Hyperbolic")

        if expr.has(sp.log):
            categories.add("Logarithmic")

        if expr.has(sp.exp):
            categories.add("Exponential")

        for node in nodes:
            if isinstance(node, sp.Pow) and node.exp.free_symbols:
                categories.add("Exponential")

        if any(node.func in ROUNDING for node in nodes):
            categories.add("Floor/Ceiling")

        if expr.has(sp.Abs):
            categories.add("Absolute Value")

        if expr.has(sp.Piecewise):
            categories.add("Piecewise")

        for node in nodes:
            if isinstance(node, sp.Pow):
                exponent = node.exp
                if exponent.is_Rational and exponent.q != 1:
                    categories.add("Radical/Root")

        if expr.has(sp.factorial):
            categories.add("Factorial")

        if expr.has(sp.sign):
            categories.add("Sign Function")

        if expr.has(sp.DiracDelta, sp.Heaviside):
            categories.add("Distribution/Step Function")

        for node in nodes:
            if isinstance(node, sp.Function):
                module = getattr(node.func, "__module__", "")
                if module.startswith("sympy.functions.special"):
                    categories.add("Special Function")

        if expr.has(sp.I):
            categories.add("Complex Expression")

        algebraic = classify_algebraic(expr)
        if algebraic:
            categories.add(algebraic)

        if not categories:
            categories.add("Other")

        return sorted(categories)

    except Exception as error:
        return [f"Invalid Syntax: {error}"]


def render():
    st.subheader("🔎 Function Identifier")
    st.write(
        "Enter a mathematical expression to identify its function types."
    )

    expression = st.text_input(
        "Mathematical expression",
        value="sin(x) + x^2",
        help="Examples: 2*x+5, x^2, sin(x), log(x), sqrt(x)",
    )

    if st.button("Identify Function", type="primary"):
        if not expression.strip():
            st.warning("Please enter an expression first.")
            return

        results = identify_function_types(expression)

        if results and results[0].startswith("Invalid Syntax:"):
            st.error(results[0])
        else:
            st.success("Classification complete!")
            st.write("**Expression:**")
            st.code(expression)

            st.write("**Identified function types:**")
            for category in results:
                st.write(f"• {category}")

            st.caption(
                "This tool classifies expressions; it does not solve equations."
            )