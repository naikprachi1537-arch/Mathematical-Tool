import warnings

import numpy as np
import matplotlib.pyplot as plt
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

x = sp.Symbol("x", real=True)
z = sp.Symbol("z")
t = sp.Symbol("t", real=True)

SAFE_LOCALS = {
    "x": x, "z": z, "t": t,
    "pi": sp.pi, "E": sp.E, "I": sp.I, "oo": sp.oo,
    "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
    "asin": sp.asin, "acos": sp.acos, "atan": sp.atan,
    "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh,
    "asinh": sp.asinh, "acosh": sp.acosh, "atanh": sp.atanh,
    "log": sp.log, "ln": sp.log, "exp": sp.exp,
    "sqrt": sp.sqrt, "Abs": sp.Abs, "abs": sp.Abs,
    "sign": sp.sign, "floor": sp.floor, "ceiling": sp.ceiling,
    "Piecewise": sp.Piecewise, "Eq": sp.Eq,
    "Integral": sp.Integral, "Min": sp.Min, "Max": sp.Max,
    "re": sp.re, "im": sp.im, "conjugate": sp.conjugate,
}


def parse_math(text, extra_locals=None):
    """Parse common mathematical notation."""
    text = (
        text.strip()
        .replace("π", "pi")
        .replace("∞", "oo")
        .replace("−", "-")
        .replace("×", "*")
        .replace("÷", "/")
    )

    if not text:
        raise ValueError("Please enter a mathematical expression.")

    local_dict = dict(SAFE_LOCALS)
    if extra_locals:
        local_dict.update(extra_locals)

    return parse_expr(
        text,
        local_dict=local_dict,
        transformations=TRANSFORMATIONS,
        evaluate=True,
    )


def safe_simplify(expr):
    try:
        return sp.simplify(expr)
    except Exception:
        return expr


def safe_substitute(expr, variable, point):
    try:
        value = expr.subs(variable, point)
        value = value.doit() if hasattr(value, "doit") else value
        return safe_simplify(value)
    except Exception:
        return None


def is_undefined(value):
    if value is None:
        return True

    try:
        if value.has(sp.nan, sp.zoo, sp.S.ComplexInfinity):
            return True
    except Exception:
        pass

    return value in (sp.nan, sp.zoo, sp.S.ComplexInfinity)


def is_finite_value(value):
    if value is None or is_undefined(value):
        return False

    if value in (sp.oo, -sp.oo):
        return False

    try:
        if value.is_finite is False:
            return False
    except Exception:
        pass

    return True


def compute_limit(expr, variable, point, direction="+-"):
    try:
        result = sp.limit(expr, variable, point, dir=direction)
        return safe_simplify(result)
    except Exception:
        return None


def equality_status(a, b):
    """Return True, False, or None if equality is undecidable."""
    if a is None or b is None:
        return None

    try:
        difference = safe_simplify(a - b)

        if difference == 0 or difference.is_zero is True:
            return True

        if difference.is_zero is False:
            return False

        answer = sp.ask(sp.Q.zero(difference))

        if answer is True:
            return True
        if answer is False:
            return False
    except Exception:
        pass

    try:
        answer = a.equals(b)
        return answer if answer in (True, False) else None
    except Exception:
        return None


def analyze_real_variable(expr, point):
    """Analyze x approaching a real point from both sides."""
    left = compute_limit(expr, x, point, "-")
    right = compute_limit(expr, x, point, "+")
    two_sided = compute_limit(expr, x, point, "+-")
    value = safe_substitute(expr, x, point)

    sides_equal = equality_status(left, right)

    if sides_equal is True:
        limit_value = two_sided if two_sided is not None else left
        limit_exists = not is_undefined(limit_value)
    elif sides_equal is False:
        limit_value, limit_exists = None, False
    else:
        limit_value = two_sided
        limit_exists = None

        if two_sided is not None and not is_undefined(two_sided):
            limit_exists = True

    continuity = "Unable to determine"

    if limit_exists is False:
        continuity = "Discontinuous: two-sided limit does not exist"

    elif limit_exists is True and is_finite_value(limit_value):
        same_value = equality_status(value, limit_value)

        if same_value is True:
            continuity = "Continuous"
        elif same_value is False or is_undefined(value):
            continuity = "Discontinuous: removable (or undefined point)"
        else:
            continuity = "Unable to determine continuity"

    elif limit_exists is True:
        continuity = "Discontinuous: non-finite limit"

    if sides_equal is False:
        classification = "Jump / unequal one-sided limits"
    elif (
        left in (sp.oo, -sp.oo, sp.zoo)
        or right in (sp.oo, -sp.oo, sp.zoo)
    ):
        classification = "Infinite behavior (check one-sided limits)"
    elif limit_exists is None:
        classification = "Limit could not be determined reliably"
    elif limit_exists is False:
        classification = "Limit does not exist"
    else:
        classification = "Finite two-sided limit"

    return {
        "variable": x,
        "point": point,
        "value": value,
        "left": left,
        "right": right,
        "limit": limit_value,
        "limit_exists": limit_exists,
        "continuity": continuity,
        "classification": classification,
    }


def analyze_complex_variable(expr, point):
    """Analyze a candidate limit as z approaches a complex point."""
    value = safe_substitute(expr, z, point)
    result = compute_limit(expr, z, point, "+-")

    continuity = "Unable to determine"

    if result is not None and is_finite_value(result):
        same_value = equality_status(value, result)

        if same_value is True:
            continuity = "Continuous (symbolic substitution agrees)"
        elif same_value is False or is_undefined(value):
            continuity = (
                "Discontinuous at the point "
                "(candidate removable singularity)"
            )
    elif result is not None:
        continuity = "Not continuous with a finite limit"

    return {
        "variable": z,
        "point": point,
        "value": value,
        "left": None,
        "right": None,
        "limit": result,
        "limit_exists": (
            None if result is None else not is_undefined(result)
        ),
        "continuity": continuity,
        "classification": (
            "Complex-plane limit: verify path independence "
            "for rigorous confirmation"
        ),
    }


def display_math(label, value):
    st.markdown(f"**{label}**")

    if value is None:
        st.info("Unable to determine symbolically.")
    else:
        st.latex(sp.latex(value))


def show_results(result, complex_mode=False):
    st.subheader("Results")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Approach point**")
        st.latex(
            rf"{sp.latex(result['variable'])}"
            rf"\to {sp.latex(result['point'])}"
        )

    with col2:
        st.markdown("**Function value at the point**")
        value = result["value"]

        if value is None:
            st.write("Undefined or unable to evaluate")
        else:
            st.latex(sp.latex(value))

    st.divider()

    if not complex_mode:
        col1, col2 = st.columns(2)

        with col1:
            display_math("Left-hand limit", result["left"])

        with col2:
            display_math("Right-hand limit", result["right"])

    display_math("Limit", result["limit"])

    st.markdown("**Continuity**")
    if result["continuity"] == "Continuous":
        st.success(result["continuity"])
    elif result["continuity"].startswith("Discontinuous"):
        st.warning(result["continuity"])
    else:
        st.info(result["continuity"])

    st.markdown("**Classification**")
    st.write(result["classification"])

    if complex_mode:
        st.info(
            "For complex-variable limits, a symbolic result alone "
            "does not prove that the limit is independent of the "
            "path taken through the complex plane."
        )


def plot_real_function(expr, point, result):
    """Plot a real-valued function around a real approach point."""
    try:
        numeric_point = complex(sp.N(point))

        if abs(numeric_point.imag) > 1e-10:
            st.info("Graph skipped: the approach point is not real.")
            return

        p = numeric_point.real
        function = sp.lambdify(x, expr, modules=["numpy"])
        xs = np.linspace(p - 3, p + 3, 1600)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ys = function(xs)

        ys = np.asarray(ys)

        if np.iscomplexobj(ys):
            if np.nanmax(np.abs(np.imag(ys))) > 1e-8:
                st.info(
                    "Graph skipped: the function has complex values "
                    "on the plotted interval."
                )
                return
            ys = np.real(ys)

        if ys.ndim == 0:
            ys = np.full_like(xs, ys, dtype=float)

        ys = np.asarray(ys, dtype=float)
        ys[~np.isfinite(ys)] = np.nan
        ys[np.abs(ys) > 1e4] = np.nan

        fig, ax = plt.subplots(figsize=(9, 5))

        ax.plot(
            xs,
            ys,
            label=f"f(x) = {sp.sstr(expr)}",
            linewidth=2,
        )
        ax.axvline(
            p,
            linestyle="--",
            alpha=0.7,
            label=f"x = {point}",
        )

        limit_value = result["limit"]

        if limit_value is not None and is_finite_value(limit_value):
            try:
                lv = complex(sp.N(limit_value))

                if abs(lv.imag) < 1e-8 and abs(lv.real) < 1e4:
                    ax.plot(
                        p,
                        lv.real,
                        "o",
                        markerfacecolor="white",
                        markersize=8,
                        markeredgewidth=2,
                        label="Limit value",
                    )
            except Exception:
                pass

        try:
            fv = complex(sp.N(result["value"]))

            if abs(fv.imag) < 1e-8 and abs(fv.real) < 1e4:
                ax.plot(p, fv.real, "o", label="Function value")
        except Exception:
            pass

        ax.set_title("Limit and Continuity")
        ax.set_xlabel("x")
        ax.set_ylabel("f(x)")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend()
        fig.tight_layout()

        st.pyplot(fig)
        plt.close(fig)

    except Exception as exc:
        st.warning(f"Graph could not be generated: {exc}")


def render():
    """Streamlit entry point for the Limit & Continuity Calculator."""
    st.title("Limit & Continuity Calculator")

    st.write(
        "Explore symbolic limits, one-sided limits, continuity, "
        "and function behavior near a point."
    )

    mode = st.radio(
        "Choose calculation mode",
        [
            "Real variable (x)",
            "Complex variable (z)",
        ],
        horizontal=True,
        key="limit_mode",
    )

    complex_mode = mode == "Complex variable (z)"
    variable = z if complex_mode else x
    variable_name = "z" if complex_mode else "x"

    with st.expander("Expression examples"):
        st.markdown(
            """
            - `sin(x)/x`
            - `(x^2-4)/(x-2)`
            - `abs(x)/x`
            - `Piecewise((x^2, x < 0), (x, True))`
            - Complex mode: `(z^2+1)/(z-I)`

            Powers can be entered as `x^2` or `x**2`.
            You can also use `pi`, `oo` for infinity, and `I` for
            the imaginary unit.
            """
        )

    expression_text = st.text_input(
        f"Enter f({variable_name})",
        value="sin(x)/x" if not complex_mode else "(z**2+1)/(z-I)",
        key="limit_expression",
    )

    point_text = st.text_input(
        "Approach point",
        value="0",
        key="limit_point",
        help="Examples: 0, 2, pi/2. For complex mode, try I.",
    )

    show_graph = False

    if not complex_mode:
        show_graph = st.checkbox(
            "Plot the function near the approach point",
            value=True,
            key="limit_plot_graph",
        )

    if st.button(
        "Calculate Limit",
        type="primary",
        use_container_width=True,
        key="calculate_limit",
    ):
        try:
            expr = parse_math(expression_text)
            point = parse_math(point_text)

            if not complex_mode and point.is_real is False:
                st.error(
                    "Real-variable mode requires a real approach point. "
                    "Choose complex-variable mode for complex points."
                )
                return

            if not expr.has(variable):
                st.caption(
                    f"The expression does not contain {variable_name}; "
                    "it is treated as a constant with respect to that "
                    "variable."
                )

            with st.spinner("Analyzing limit and continuity..."):
                result = (
                    analyze_complex_variable(expr, point)
                    if complex_mode
                    else analyze_real_variable(expr, point)
                )

            st.markdown("**Parsed expression**")
            st.latex(sp.latex(expr))

            show_results(result, complex_mode=complex_mode)

            if show_graph and not complex_mode:
                st.divider()
                st.subheader("Function Graph")
                plot_real_function(expr, point, result)

        except Exception as exc:
            st.error(f"Could not calculate the limit: {exc}")
            st.caption(
                "Check your expression syntax. Examples: sin(x), "
                "(x^2-1)/(x-1), Piecewise((x, x < 0), (x**2, True))."
            )

    st.divider()
    st.caption(
        "Educational tool: symbolic results may be inconclusive for "
        "some expressions. Complex-variable limits may require "
        "additional path analysis."
    )


if __name__ == "__main__":
    render()