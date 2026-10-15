import warnings

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import sympy as sp

from sympy import (
    Abs, E, I, S, Symbol, pi, oo, exp, log, sqrt, Piecewise,
    floor, ceiling, sign, Min, Max, sinc, Heaviside, DiracDelta,
    sin, cos, tan, cot, sec, csc,
    asin, acos, atan, acot, asec, acsc,
    sinh, cosh, tanh, coth, sech, csch,
)
from sympy.calculus.util import continuous_domain, function_range
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)

x = Symbol("x", real=True)

SAFE_FUNCTIONS = {
    "sin": sin, "cos": cos, "tan": tan,
    "cot": cot, "sec": sec, "csc": csc,
    "asin": asin, "acos": acos, "atan": atan,
    "acot": acot, "asec": asec, "acsc": acsc,
    "sinh": sinh, "cosh": cosh, "tanh": tanh,
    "coth": coth, "sech": sech, "csch": csch,
    "sqrt": sqrt, "Abs": Abs, "abs": Abs,
    "exp": exp, "log": log, "ln": log,
    "floor": floor, "ceiling": ceiling, "ceil": ceiling,
    "sign": sign, "Min": Min, "Max": Max,
    "Piecewise": Piecewise, "Heaviside": Heaviside,
    "DiracDelta": DiracDelta, "sinc": sinc,
    "pi": pi, "E": E, "e": E, "I": I,
    "oo": oo, "inf": oo,
    "True": True, "False": False,
    "x": x,
}

TRANSFORMATIONS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)


def parse_function(text):
    """Parse a supported mathematical expression."""
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Please enter a mathematical expression.")

    text = text.strip()
    text = text.replace("π", "pi").replace("∞", "oo")
    text = text.replace("−", "-").replace("×", "*").replace("÷", "/")
    text = text.replace("√", "sqrt")

    expr = parse_expr(
        text,
        local_dict=SAFE_FUNCTIONS.copy(),
        global_dict={
            "__builtins__": {},
            "Integer": sp.Integer,
            "Float": sp.Float,
            "Rational": sp.Rational,
            "Symbol": sp.Symbol,
            "Add": sp.Add,
            "Mul": sp.Mul,
            "Pow": sp.Pow,
        },
        transformations=TRANSFORMATIONS,
        evaluate=True,
    )

    if not isinstance(expr, sp.Expr):
        raise ValueError("Input must be a mathematical expression in x.")

    # This calculator supports one real variable: x.
    if expr.free_symbols - {x}:
        raise ValueError("Please use only x as the variable.")

    return expr


def safe_domain(expr, variable):
    try:
        return continuous_domain(expr, variable, S.Reals)
    except Exception:
        return None


def safe_range(expr, variable, domain):
    if domain is None:
        return None

    try:
        return function_range(expr, variable, domain)
    except Exception:
        return None


def domain_contains(domain, value):
    if domain is None:
        return True

    try:
        result = domain.contains(sp.Float(value))
        return result is S.true or result is True
    except Exception:
        return False


def real_numeric_value(expr, variable, value):
    try:
        result = sp.N(expr.subs(variable, sp.Float(value)))

        if result.has(sp.zoo, sp.nan, sp.oo, -sp.oo):
            return None

        if result.is_real is False:
            return None

        numeric = complex(result)

        if abs(numeric.imag) > 1e-8:
            return None

        if not np.isfinite(numeric.real):
            return None

        return float(numeric.real)

    except Exception:
        return None


def numerical_injectivity_check(expr, variable, domain):
    """A sampled collision can disprove injectivity, not prove it."""
    sample_x = np.linspace(-10, 10, 401)
    outputs = []

    for value in sample_x:
        if not domain_contains(domain, value):
            continue

        y_value = real_numeric_value(expr, variable, value)

        if y_value is not None:
            outputs.append((value, y_value))

    outputs.sort(key=lambda item: item[1])

    for i in range(len(outputs)):
        for j in range(i + 1, min(i + 8, len(outputs))):
            x1, y1 = outputs[i]
            x2, y2 = outputs[j]

            if abs(y1 - y2) < 1e-7 and abs(x1 - x2) > 1e-4:
                return "Many-to-One (sample collision found)"

    return "Unknown (cannot prove injectivity)"


def classify_function(expr, variable, domain, value_range,
                      codomain=S.Reals):
    injective = None

    if not expr.has(variable):
        injective = False

    if injective is None:
        if expr == variable or expr == -variable:
            injective = True
        elif expr == variable**3 or expr == -variable**3:
            injective = True
        elif expr == sp.exp(variable):
            injective = True

    if injective is None:
        heuristic = numerical_injectivity_check(
            expr, variable, domain
        )

        if heuristic.startswith("Many-to-One"):
            injective_label = heuristic
        else:
            injective_label = "Unknown (cannot prove injectivity)"

    elif injective:
        injective_label = "One-to-One (Injective)"
    else:
        injective_label = "Many-to-One"

    if value_range is None:
        onto_label = "Unknown (range could not be calculated)"
    else:
        try:
            equal = sp.simplify(value_range == codomain)

            if equal is True or equal is S.true:
                onto_label = "Onto (Surjective)"
            else:
                unequal = sp.simplify(value_range != codomain)

                if unequal is True or unequal is S.true:
                    onto_label = "Into (range is not the full codomain)"
                else:
                    onto_label = "Unknown (range/codomain unresolved)"
        except Exception:
            onto_label = "Unknown (range/codomain unresolved)"

    if (
        injective is True
        and onto_label == "Onto (Surjective)"
    ):
        bijective_label = "Bijective"
    elif (
        injective is False
        or onto_label.startswith("Into")
        or injective_label.startswith("Many-to-One")
    ):
        bijective_label = "Not Bijective"
    else:
        bijective_label = "Unknown"

    return injective_label, onto_label, bijective_label


def sample_mapping(expr, variable, domain, max_points=7):
    candidates = [-3, -2, -1, 0, 1, 2, 3]
    pairs = []

    for value in candidates:
        if not domain_contains(domain, value):
            continue

        result = real_numeric_value(expr, variable, value)

        if result is not None:
            pairs.append((float(value), result))

        if len(pairs) >= max_points:
            break

    return pairs


def draw_plots(expr, variable, domain):
    fig, (ax_map, ax_graph) = plt.subplots(
        1, 2, figsize=(14, 5)
    )

    # Illustrative mapping diagram
    ax_map.set_title("Mapping Diagram (sample values)")
    ax_map.axis("off")

    pairs = sample_mapping(expr, variable, domain)

    if not pairs:
        ax_map.text(
            0.5, 0.5,
            "No suitable real sample points found.",
            ha="center", va="center",
            transform=ax_map.transAxes,
        )
    else:
        input_x, output_x = 0.18, 0.78
        input_positions = np.linspace(0.85, 0.15, len(pairs))

        output_values = []
        for _, y_value in pairs:
            if not any(
                abs(y_value - old) < 1e-7
                for old in output_values
            ):
                output_values.append(y_value)

        output_positions = np.linspace(
            0.85, 0.15, max(1, len(output_values))
        )

        output_pos = {
            value: output_positions[i]
            for i, value in enumerate(output_values)
        }

        for (x_value, y_value), y_pos in zip(
            pairs, input_positions
        ):
            ax_map.scatter(
                [input_x], [y_pos],
                s=55, color="dodgerblue"
            )
            ax_map.text(
                input_x - 0.05, y_pos, f"{x_value:g}",
                ha="right", va="center"
            )

            target = min(
                output_values,
                key=lambda val: abs(val - y_value)
            )

            ax_map.annotate(
                "",
                xy=(output_x - 0.04, output_pos[target]),
                xytext=(input_x + 0.04, y_pos),
                arrowprops=dict(
                    arrowstyle="->", color="gray", lw=1.2
                ),
            )

        for y_value, y_pos in zip(
            output_values, output_positions
        ):
            ax_map.scatter(
                [output_x], [y_pos],
                s=55, color="forestgreen"
            )
            ax_map.text(
                output_x + 0.05, y_pos, f"{y_value:.4g}",
                ha="left", va="center"
            )

        ax_map.text(
            input_x, 0.96, "Inputs x",
            ha="center", weight="bold"
        )
        ax_map.text(
            output_x, 0.96, "Outputs f(x)",
            ha="center", weight="bold"
        )

        ax_map.set_xlim(0, 1)
        ax_map.set_ylim(0, 1)

    # Cartesian graph
    ax_graph.set_title(f"f(x) = {sp.sstr(expr)}")
    ax_graph.set_xlabel("x")
    ax_graph.set_ylabel("f(x)")
    ax_graph.axhline(0, color="black", lw=0.8)
    ax_graph.axvline(0, color="black", lw=0.8)
    ax_graph.grid(True, linestyle=":", alpha=0.5)

    try:
        numeric_func = sp.lambdify(
            variable, expr, modules=["numpy", "scipy"]
        )

        x_values = np.linspace(-10, 10, 2000)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            y_values = numeric_func(x_values)

        if np.isscalar(y_values):
            y_values = np.full_like(
                x_values, y_values, dtype=float
            )

        y_values = np.asarray(y_values)

        if np.iscomplexobj(y_values):
            real_mask = np.abs(np.imag(y_values)) < 1e-8
            y_values = np.where(
                real_mask, np.real(y_values), np.nan
            )

        y_values = np.asarray(y_values, dtype=float)
        y_values[~np.isfinite(y_values)] = np.nan
        y_values[np.abs(y_values) > 100] = np.nan

        ax_graph.plot(
            x_values, y_values,
            color="crimson", lw=1.8
        )
        ax_graph.set_xlim(-10, 10)
        ax_graph.set_ylim(-10, 10)

    except Exception:
        ax_graph.text(
            0.5, 0.5,
            "Could not plot this expression.",
            ha="center", va="center",
            transform=ax_graph.transAxes,
        )

    fig.tight_layout()
    return fig


def render():
    """Streamlit entry point for the main application."""

    st.title("Function Mapping, Domain & Range")

    st.write(
        "Analyze a mathematical function, calculate its domain "
        "and range, and explore its mapping diagram and graph."
    )

    with st.form("function_mapping_domain_range_form"):
        expression_text = st.text_input(
            "Enter f(x)",
            value="x^2 - 4",
            help=(
                "Examples: x^2 - 4, (x+1)/(x-2), "
                "sqrt(x-1), sin(x), exp(x)"
            ),
        )

        submitted = st.form_submit_button(
            "Analyze Function",
            type="primary",
            use_container_width=True,
        )

    if not submitted:
        st.info(
            "Enter a function and click Analyze Function to begin."
        )
        st.caption(
            "Use x as the variable. Both ^ and ** can represent powers."
        )
        return

    try:
        expr = parse_function(expression_text)
    except Exception as exc:
        st.error(f"Invalid expression: {exc}")
        st.info(
            "Check your parentheses and function names. "
            "Examples: x^2, sin(x), sqrt(x-1), (x+1)/(x-2)."
        )
        return

    with st.spinner("Analyzing the function..."):
        domain = safe_domain(expr, x)
        value_range = safe_range(expr, x, domain)

        injective, onto, bijective = classify_function(
            expr, x, domain, value_range
        )

    st.subheader("Function")
    st.latex(r"f(x) = " + sp.latex(expr))

    st.subheader("Domain and Range")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Domain**")
        if domain is not None:
            st.latex(sp.latex(domain))
        else:
            st.warning(
                "Could not determine the exact domain symbolically."
            )

    with col2:
        st.markdown("**Range**")
        if value_range is not None:
            st.latex(sp.latex(value_range))
        else:
            st.warning(
                "Could not determine the exact range symbolically."
            )

    st.subheader("Function Classification")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Injectivity**")
        st.write(injective)

    with col2:
        st.markdown("**Onto / Into**")
        st.write(onto)

    with col3:
        st.markdown("**Bijectivity**")
        st.write(bijective)

    st.caption(
        "Onto/into analysis assumes the codomain is the real numbers. "
        "Unknown means the program could not establish the result."
    )

    st.subheader("Mapping Diagram and Graph")

    try:
        fig = draw_plots(expr, x, domain)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    except Exception as exc:
        st.error(f"Could not generate the plots: {exc}")

    with st.expander("Sample function values"):
        pairs = sample_mapping(expr, x, domain)

        if pairs:
            st.dataframe(
                [
                    {
                        "x": round(x_value, 5),
                        "f(x)": round(y_value, 5),
                    }
                    for x_value, y_value in pairs
                ],
                use_container_width=True,
                hide_index=True,
            )
        else:
            st.write("No suitable real sample values were found.")

    with st.expander("How to interpret these results"):
        st.markdown(
            """
            - **One-to-one:** Different inputs have different outputs.
            - **Many-to-one:** At least two different inputs share an output.
            - **Onto:** The range equals the specified codomain.
            - **Into:** The range does not cover the entire codomain.
            - **Bijective:** The function is both one-to-one and onto.
            """
        )

        st.warning(
            "Numerical sampling is illustrative, not a mathematical "
            "proof. Some classifications may remain unknown."
        )


# Standalone testing is optional. The main app imports the function above.
def main():
    render()


if __name__ == "__main__":
    main()