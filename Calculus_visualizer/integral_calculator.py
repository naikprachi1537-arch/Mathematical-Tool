import warnings

import numpy as np
import scipy.integrate as scipy_integrate
import streamlit as st
import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)

# Accept expressions such as 2x and x^2.
TRANSFORMATIONS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)

LOCAL_DICT = {
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "asin": sp.asin,
    "acos": sp.acos,
    "atan": sp.atan,
    "sinh": sp.sinh,
    "cosh": sp.cosh,
    "tanh": sp.tanh,
    "exp": sp.exp,
    "log": sp.log,
    "ln": sp.log,
    "sqrt": sp.sqrt,
    "abs": sp.Abs,
    "Abs": sp.Abs,
    "pi": sp.pi,
    "E": sp.E,
    "e": sp.E,
    "oo": sp.oo,
    "inf": sp.oo,
    "I": sp.I,
    "gamma": sp.gamma,
    "erf": sp.erf,
    "floor": sp.floor,
    "ceiling": sp.ceiling,
}


def parse_math(text):
    """Parse a mathematical expression."""
    text = text.strip().replace("[", "(").replace("]", ")")

    if not text:
        raise ValueError("Expression cannot be empty.")

    return parse_expr(
        text,
        local_dict=LOCAL_DICT.copy(),
        transformations=TRANSFORMATIONS,
        evaluate=True,
    )


def get_variable(expr, requested=None):
    """Get the integration variable."""
    symbols = sorted(expr.free_symbols, key=lambda s: s.name)

    if requested:
        name = requested.strip()

        if not name.isidentifier():
            raise ValueError("Enter a valid variable name, such as x or t.")

        match = next(
            (symbol for symbol in symbols if symbol.name == name),
            None,
        )

        return match if match is not None else sp.Symbol(name)

    if len(symbols) == 1:
        return symbols[0]

    if not symbols:
        return sp.Symbol("x")

    return symbols[0]


def numeric_integral(expr, var, lower, upper):
    """Try numerical integration for a real-valued function."""
    numeric_fn = sp.lambdify(var, expr, modules=["numpy", "scipy"])

    def function(value):
        result = numeric_fn(value)

        if hasattr(result, "item"):
            result = result.item()

        result = complex(result)

        if abs(result.imag) > 1e-10:
            raise ValueError(
                "The integrand became complex; real numerical "
                "integration cannot be used."
            )

        return float(result.real)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter(
            "always",
            scipy_integrate.IntegrationWarning,
        )

        value, error = scipy_integrate.quad(
            function,
            float(sp.N(lower)),
            float(sp.N(upper)),
            limit=300,
        )

    messages = [str(item.message) for item in caught]
    return value, error, messages


def show_indefinite(expr, var):
    """Display an indefinite integral."""
    st.subheader("Indefinite Integral")

    with st.spinner("Calculating symbolic integral..."):
        try:
            result = sp.simplify(sp.integrate(expr, var))

            st.markdown("**Your integral**")
            st.latex(
                rf"\int {sp.latex(expr)}\,d{sp.latex(var)}"
            )

            if isinstance(result, sp.Integral) or result.has(sp.Integral):
                st.warning(
                    "SymPy could not find a complete symbolic "
                    "antiderivative. The result below is unevaluated."
                )
                st.latex(sp.latex(result))
                return

            st.markdown("**Antiderivative**")
            st.latex(
                rf"{sp.latex(result)} + C"
            )

            # Verify by differentiating the answer.
            try:
                check = sp.simplify(sp.diff(result, var) - expr)

                if check == 0:
                    st.success(
                        "Verification passed: differentiating the "
                        "answer reproduces the integrand."
                    )
                else:
                    st.info(
                        "Automatic verification did not simplify "
                        "to zero."
                    )
            except Exception:
                st.info("Automatic verification was unavailable.")

        except Exception as exc:
            st.error(f"Symbolic integration failed: {exc}")


def show_definite(expr, var, lower, upper):
    """Display a definite integral and numerical fallback if needed."""
    st.subheader("Definite Integral")

    st.markdown("**Your integral**")
    st.latex(
        rf"\int_{{{sp.latex(lower)}}}^{{{sp.latex(upper)}}}"
        rf"{sp.latex(expr)}\,d{sp.latex(var)}"
    )

    with st.spinner("Calculating definite integral..."):
        try:
            exact = sp.integrate(expr, (var, lower, upper))
        except Exception:
            exact = sp.Integral(expr, (var, lower, upper))

    if not isinstance(exact, sp.Integral) and not exact.has(sp.Integral):
        st.markdown("**Symbolic result**")
        st.latex(sp.latex(exact))

        try:
            approx = sp.N(exact, 15)

            if approx.is_finite:
                st.metric("Decimal approximation", str(approx))
            elif approx in (sp.oo, -sp.oo, sp.zoo, sp.nan):
                st.warning(
                    "The result is infinite or undefined. "
                    "Check convergence and singularities."
                )
        except Exception:
            pass

        return

    st.warning(
        "SymPy could not fully evaluate this integral symbolically."
    )
    st.markdown("**Unevaluated integral**")
    st.latex(sp.latex(exact))

    # Numerical quadrature requires finite real bounds.
    if lower.free_symbols or upper.free_symbols:
        st.info("Numerical fallback requires constant bounds.")
        return

    if lower.is_real is False or upper.is_real is False:
        st.info("Numerical fallback requires real bounds.")
        return

    if lower in (sp.oo, -sp.oo) or upper in (sp.oo, -sp.oo):
        st.info(
            "Numerical fallback in this calculator requires finite "
            "bounds. The symbolic result is shown above."
        )
        return

    st.markdown("**Numerical approximation**")

    try:
        value, error, notes = numeric_integral(
            expr, var, lower, upper
        )

        st.metric("Estimated integral value", f"{value:.12g}")
        st.caption(f"Estimated absolute error: {error:.3g}")

        if notes:
            st.warning(
                "Numerical integration reported a warning. "
                "The estimate may be unreliable."
            )

            for note in notes:
                st.write(f"- {note}")

    except Exception as exc:
        st.error(
            f"Numerical integration could not produce a result: {exc}"
        )
        st.caption(
            "The integral may be divergent, complex-valued, singular, "
            "or too difficult for numerical integration."
        )


def render():
    """Render the Integral Calculator inside the Streamlit app."""
    st.title("∫ Integral Calculator")
    st.write(
        "Calculate indefinite and definite integrals using symbolic "
        "mathematics, with numerical integration as a fallback."
    )

    with st.expander("Expression syntax and examples"):
        st.markdown(
            """
            - Polynomial: `x^2 + 3x + 2`
            - Trigonometric: `sin(x)`, `cos(x^2)`
            - Exponential: `exp(-x^2)`
            - Logarithmic: `log(x)` or `ln(x)`
            - Square root: `sqrt(x)`
            - Rational: `1/(1+x^2)`
            - Special functions: `gamma(x)`, `erf(x)`

            You can use `x^2` or `x**2`, and implicit multiplication
            such as `2x`.
            """
        )

    expression_text = st.text_input(
        "Enter the function to integrate",
        value="x^2",
        key="integral_expression",
        placeholder="Example: x^2 + sin(x)",
    )

    try:
        parsed_expr = parse_math(expression_text)
        available_variables = sorted(
            symbol.name for symbol in parsed_expr.free_symbols
        )
    except Exception:
        parsed_expr = None
        available_variables = []

    default_variable = (
        available_variables[0] if available_variables else "x"
    )

    variable_options = list(
        dict.fromkeys(available_variables + [default_variable, "x", "y", "t"])
    )

    variable = st.selectbox(
        "Integrate with respect to",
        options=variable_options,
        index=variable_options.index(default_variable),
        key="integral_variable",
    )

    integral_type = st.radio(
        "Integral type",
        options=["Indefinite", "Definite"],
        horizontal=True,
        key="integral_type",
    )

    lower_text = "0"
    upper_text = "1"

    if integral_type == "Definite":
        col1, col2 = st.columns(2)

        with col1:
            lower_text = st.text_input(
                "Lower bound",
                value="0",
                key="integral_lower_bound",
            )

        with col2:
            upper_text = st.text_input(
                "Upper bound",
                value="1",
                key="integral_upper_bound",
            )

    calculate = st.button(
        "Calculate Integral",
        type="primary",
        key="calculate_integral",
        use_container_width=True,
    )

    if calculate:
        try:
            expr = parse_math(expression_text)
            var = get_variable(expr, variable)

            if integral_type == "Indefinite":
                show_indefinite(expr, var)

            else:
                lower = parse_math(lower_text)
                upper = parse_math(upper_text)

                if lower.free_symbols or upper.free_symbols:
                    st.error(
                        "Bounds must be constants, such as 0, 1, pi, "
                        "or -2.5. Variables are not allowed in bounds."
                    )
                    return

                if lower.is_real is False or upper.is_real is False:
                    st.error("Please enter real-valued bounds.")
                    return

                show_definite(expr, var, lower, upper)

        except Exception as exc:
            st.error(f"Could not calculate the integral: {exc}")

    st.divider()
    st.caption(
        "Note: symbolic integration may not find a closed-form answer "
        "for every function. Numerical results are estimates, not proofs."
    )


# Allows app.py to load this calculator as a module.
if __name__ == "__main__":
    render()