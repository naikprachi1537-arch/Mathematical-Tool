import streamlit as st
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import warnings

from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
    function_exponentiation,
    rationalize,
)

x, y, t = sp.symbols("x y t", real=True)

TRANSFORMATIONS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
    function_exponentiation,
    rationalize,
)

LOCAL_DICT = {
    "x": x,
    "y": y,
    "t": t,
    "pi": sp.pi,
    "e": sp.E,
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "asin": sp.asin,
    "acos": sp.acos,
    "atan": sp.atan,
    "sinh": sp.sinh,
    "cosh": sp.cosh,
    "tanh": sp.tanh,
    "sqrt": sp.sqrt,
    "abs": sp.Abs,
    "exp": sp.exp,
    "ln": sp.log,
    "log": sp.log,
    "floor": sp.floor,
    "ceil": sp.ceiling,
}

def parse_math(expression):
    return parse_expr(
        expression,
        local_dict=LOCAL_DICT,
        transformations=TRANSFORMATIONS,
    )

def render():
    st.subheader("📈 Function Plotter")
    st.write("Enter a mathematical function and generate its graph.")

    mode = st.selectbox(
        "Plot type",
        [
            "Function y = f(x)",
            "Implicit equation F(x, y) = 0",
            "Parametric curve",
        ],
    )

    expression = st.text_input(
        "Enter your function",
        value="sin(x)",
        help="Examples: x^2, sin(x), exp(x), sqrt(x)",
    )

    if mode == "Parametric curve":
        x_expression = st.text_input(
            "Enter x(t)", value="cos(t)"
        )
        y_expression = st.text_input(
            "Enter y(t)", value="sin(t)"
        )

    col1, col2 = st.columns(2)
    with col1:
        lower = st.number_input(
            "Range minimum", value=-10.0
        )
    with col2:
        upper = st.number_input(
            "Range maximum", value=10.0
        )

    if st.button("Generate Graph", type="primary"):
        if lower >= upper:
            st.error("The minimum must be smaller than the maximum.")
            return

        fig, ax = plt.subplots(figsize=(9, 5))

        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                if mode == "Function y = f(x)":
                    expr = parse_math(expression)

                    if not expr.free_symbols.issubset({x}):
                        raise ValueError(
                            "For this plot type, use x as the variable."
                        )

                    function = sp.lambdify(x, expr, modules=["numpy"])
                    xs = np.linspace(lower, upper, 2000)
                    values = np.asarray(function(xs))

                    if values.ndim == 0:
                        values = np.full(xs.shape, values.item())

                    if np.iscomplexobj(values):
                        if np.any(np.abs(np.imag(values)) > 1e-10):
                            raise ValueError(
                                "This function produces complex values. "
                                "Try a different expression."
                            )
                        values = np.real(values)

                    values = np.asarray(values, dtype=float)
                    values[~np.isfinite(values)] = np.nan
                    values[np.abs(values) > 1e6] = np.nan

                    ax.plot(xs, values, linewidth=2, label=f"y = {expr}")
                    ax.set_ylabel("y")

                elif mode == "Implicit equation F(x, y) = 0":
                    if "=" in expression:
                        left, right = expression.split("=", 1)
                        expr = parse_math(f"({left}) - ({right})")
                    else:
                        expr = parse_math(expression)

                    if not expr.free_symbols.issubset({x, y}):
                        raise ValueError(
                            "Use only x and y in an implicit equation."
                        )

                    function = sp.lambdify((x, y), expr, modules=["numpy"])
                    xs = np.linspace(lower, upper, 400)
                    ys = np.linspace(lower, upper, 400)
                    X, Y = np.meshgrid(xs, ys)
                    Z = np.asarray(function(X, Y), dtype=float)
                    Z[~np.isfinite(Z)] = np.nan

                    ax.contour(X, Y, Z, levels=[0], linewidths=2)
                    ax.set_ylabel("y")

                else:
                    expr_x = parse_math(x_expression)
                    expr_y = parse_math(y_expression)

                    if not expr_x.free_symbols.issubset({t}):
                        raise ValueError("x(t) must use only t.")
                    if not expr_y.free_symbols.issubset({t}):
                        raise ValueError("y(t) must use only t.")

                    fx = sp.lambdify(t, expr_x, modules=["numpy"])
                    fy = sp.lambdify(t, expr_y, modules=["numpy"])
                    ts = np.linspace(lower, upper, 2000)

                    xv = np.asarray(fx(ts), dtype=float)
                    yv = np.asarray(fy(ts), dtype=float)
                    xv[~np.isfinite(xv)] = np.nan
                    yv[~np.isfinite(yv)] = np.nan

                    ax.plot(xv, yv, linewidth=2)
                    ax.set_ylabel("y(t)")
                    ax.set_xlabel("x(t)")
                    ax.set_aspect("equal", adjustable="datalim")

                if mode != "Parametric curve":
                    ax.set_xlabel("x")

                ax.axhline(0, color="gray", linewidth=0.8)
                ax.axvline(0, color="gray", linewidth=0.8)
                ax.grid(True, alpha=0.3)
                ax.legend() if mode == "Function y = f(x)" else None
                fig.tight_layout()

                st.pyplot(fig)
                plt.close(fig)
                st.success("Graph generated successfully!")

        except Exception as error:
            plt.close(fig)
            st.error(f"Could not plot this expression: {error}")

