import numpy as np
import streamlit as st
import sympy as sp
from sympy import (
    Symbol, pi, E, I, FiniteSet, Interval, solveset,
)
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

FUNCTIONS = {
    "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
    "cot": sp.cot, "sec": sp.sec, "csc": sp.csc,
    "cosec": sp.csc,
    "asin": sp.asin, "acos": sp.acos, "atan": sp.atan,
    "acot": sp.acot, "asec": sp.asec, "acsc": sp.acsc,
    "sqrt": sp.sqrt, "abs": sp.Abs, "log": sp.log,
    "ln": sp.log, "exp": sp.exp, "pi": pi, "E": E,
    "I": I, "e": E, "oo": sp.oo,
    "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh,
    "asinh": sp.asinh, "acosh": sp.acosh, "atanh": sp.atanh,
    "floor": sp.floor, "ceiling": sp.ceiling,
}


def parse_user_expression(text, var):
    """Parse common calculator notation into a SymPy expression."""
    text = text.strip()
    text = text.replace("[", "(").replace("]", ")")
    text = text.replace("{", "(").replace("}", ")")
    text = text.replace("π", "pi").replace("∞", "oo")
    text = text.replace("°", "*(pi/180)")
    text = text.replace("−", "-").replace("×", "*").replace("÷", "/")

    local = dict(FUNCTIONS)
    local[str(var)] = var

    return parse_expr(
        text,
        local_dict=local,
        transformations=TRANSFORMATIONS,
        evaluate=True,
    )


def simplify_candidates(expr):
    """Try multiple simplification strategies."""
    candidates = [expr]

    strategies = [
        sp.simplify,
        sp.trigsimp,
        lambda e: sp.trigsimp(sp.expand_trig(e)),
        lambda e: sp.trigsimp(e.rewrite(sp.sin)),
        lambda e: sp.trigsimp(e.rewrite(sp.cos)),
        lambda e: sp.cancel(sp.together(e)),
        lambda e: sp.factor(sp.cancel(sp.together(e))),
    ]

    try:
        from sympy.simplify.fu import fu
        strategies.append(fu)
    except Exception:
        pass

    for strategy in strategies:
        try:
            result = strategy(expr)
            if result is not None:
                candidates.append(result)
        except Exception:
            continue

    return min(
        candidates,
        key=lambda item: (sp.count_ops(item), len(str(item))),
    )


def denominator_conditions(expr):
    """Collect denominators that must not be zero."""
    conditions = []

    try:
        denominator = sp.denom(sp.together(expr))
        if denominator != 1:
            conditions.append(denominator)
    except Exception:
        pass

    for node in sp.preorder_traversal(expr):
        if isinstance(node, sp.Pow) and node.exp.is_negative:
            conditions.append(node.base)

    return conditions


def is_numerically_valid(original_diff, var, value, tol=1e-8):
    """Check a candidate against the original equation."""
    try:
        substituted = original_diff.subs(var, value)

        if substituted.has(sp.zoo, sp.nan, sp.oo, -sp.oo):
            return False

        numeric = complex(sp.N(substituted, 30))
        return abs(numeric) <= tol
    except Exception:
        return False


def denominator_is_nonzero(expr, var, value, tol=1e-10):
    for denominator in denominator_conditions(expr):
        try:
            result = complex(sp.N(denominator.subs(var, value), 30))
            if abs(result) < tol:
                return False
        except Exception:
            continue

    return True


def numeric_roots(expr, var, lo, hi, samples=1600, tol=1e-8):
    """Approximate real roots over a finite interval."""
    import scipy.optimize

    roots = []
    function = sp.lambdify(var, expr, modules=["numpy"])

    lo_num = float(sp.N(lo))
    hi_num = float(sp.N(hi))
    grid = np.linspace(lo_num, hi_num, samples + 1)

    def add_root(value):
        if value < lo_num - tol or value >= hi_num - tol:
            return

        if any(abs(value - old) < 1e-6 for old in roots):
            return

        if is_numerically_valid(expr, var, sp.Float(value), tol=1e-6):
            roots.append(float(value))

    previous_x = None
    previous_y = None

    for current_x in grid:
        try:
            current_y = complex(function(current_x))

            if (
                abs(current_y.imag) >= 1e-7
                or abs(current_y.real) >= 1e12
            ):
                previous_x, previous_y = None, None
                continue

            current_y = current_y.real

            if abs(current_y) < 1e-7:
                add_root(current_x)

            if (
                previous_x is not None
                and previous_y * current_y < 0
            ):
                try:
                    root = scipy.optimize.brentq(
                        lambda value: float(np.real(function(value))),
                        previous_x,
                        current_x,
                    )
                    add_root(root)
                except Exception:
                    pass

            previous_x, previous_y = current_x, current_y

        except Exception:
            previous_x, previous_y = None, None

    # Additional guesses can find some roots that touch zero
    # without changing sign.
    for guess in np.linspace(lo_num, hi_num, 100, endpoint=False):
        try:
            root = sp.nsolve(
                expr,
                var,
                float(guess),
                tol=1e-14,
                maxsteps=100,
                verify=False,
            )

            numeric = complex(sp.N(root, 20))

            if abs(numeric.imag) < 1e-7:
                add_root(numeric.real)

        except Exception:
            pass

    return sorted(roots)


def solve_equation(lhs, rhs, var, lo=0, hi=2 * sp.pi):
    """Solve an equation and return solutions and diagnostic notes."""
    original_diff = lhs - rhs
    simplified = simplify_candidates(original_diff)

    domain = Interval(
        sp.sympify(lo),
        sp.sympify(hi),
        left_open=False,
        right_open=True,
    )

    solutions = []
    solver_notes = []

    attempts = [
        (
            "Symbolic solver",
            lambda expr: solveset(expr, var, domain=domain),
        ),
        (
            "Trig-expanded solver",
            lambda expr: solveset(
                sp.trigsimp(sp.expand_trig(expr)),
                var,
                domain=domain,
            ),
        ),
        (
            "Rational simplification solver",
            lambda expr: solveset(
                sp.cancel(sp.together(expr)),
                var,
                domain=domain,
            ),
        ),
    ]

    for label, attempt in attempts:
        try:
            result = attempt(simplified)
            solver_notes.append(f"{label}: {result}")

            if isinstance(result, FiniteSet):
                for solution in result:
                    if solution.is_real is False:
                        continue

                    if not denominator_is_nonzero(
                        original_diff, var, solution
                    ):
                        continue

                    if is_numerically_valid(
                        original_diff, var, solution
                    ):
                        solutions.append(sp.simplify(solution))

            if solutions:
                break

        except Exception as exc:
            solver_notes.append(f"{label} failed: {exc}")

    # Numerical fallback.
    if not solutions:
        try:
            numeric = numeric_roots(simplified, var, lo, hi)

            for root in numeric:
                value = sp.Float(root, 12)

                if (
                    denominator_is_nonzero(original_diff, var, value)
                    and is_numerically_valid(
                        original_diff, var, value, tol=1e-6
                    )
                ):
                    solutions.append(value)

        except Exception as exc:
            solver_notes.append(f"Numerical fallback failed: {exc}")

    # Deduplicate solutions.
    unique = []

    for solution in solutions:
        try:
            numeric_solution = float(sp.N(solution))

            if not any(
                abs(numeric_solution - float(sp.N(old))) < 1e-7
                for old in unique
            ):
                unique.append(solution)

        except Exception:
            if solution not in unique:
                unique.append(solution)

    unique.sort(key=lambda solution: float(sp.N(solution)))

    return {
        "original": sp.Eq(lhs, rhs),
        "reduced": sp.Eq(simplified, 0),
        "solutions": unique,
        "notes": solver_notes,
        "lower": lo,
        "upper": hi,
    }


def show_equation_results(result):
    st.subheader("Equation Results")

    st.markdown("**Original equation**")
    st.latex(sp.latex(result["original"]))

    st.markdown("**Reduced equation**")
    st.latex(sp.latex(result["reduced"]))

    st.markdown(
        f"Search interval: "
        f"[{sp.sstr(result['lower'])}, {sp.sstr(result['upper'])})"
    )

    solutions = result["solutions"]

    if solutions:
        st.success(f"Found {len(solutions)} distinct solution(s).")

        for index, solution in enumerate(solutions, start=1):
            degrees = sp.N(solution * 180 / pi, 10)
            radians = sp.N(solution, 10)

            with st.container(border=True):
                st.markdown(f"**Solution {index}**")
                st.latex(sp.latex(solution))
                st.write(f"Radians: `{radians}`")
                st.write(f"Degrees: `{degrees}°`")

        st.info(
            "Numerical fallback can miss roots. These results are "
            "not a universal proof that every solution was found."
        )

    else:
        st.warning(
            "No verified solutions were found in this interval. "
            "This does not prove that no solutions exist."
        )

        if result["notes"]:
            with st.expander("Solver details"):
                for note in result["notes"][-3:]:
                    st.write(note)


def render():
    """Streamlit entry point for the Trigonometric Calculator."""
    st.title("Trigonometric Calculator")

    st.write(
        "Simplify trigonometric expressions or solve equations "
        "using symbolic and numerical methods."
    )

    x = Symbol("x", real=True)

    mode = st.radio(
        "Choose operation",
        ["Simplify Expression", "Solve Equation"],
        horizontal=True,
        key="trig_mode",
    )

    with st.expander("Examples and supported notation"):
        st.markdown(
            """
            **Simplify expressions**
            - `sin(x)^2 + cos(x)^2`
            - `tan(x) * cos(x)`
            - `sin(2*x) / sin(x)`

            **Solve equations**
            - `sin(x) = 1/2`
            - `tan(2*x) = sqrt(3)`
            - `cos(x)^2 = 1/4`

            Use radians by default. Degrees are supported with the
            degree symbol, such as `30°`. Powers can use `^` or `**`.
            """
        )

    expression = st.text_input(
        "Enter expression or equation",
        value="sin(x)^2 + cos(x)^2"
        if mode == "Simplify Expression"
        else "sin(x) = 1/2",
        key="trig_expression",
        placeholder="Example: sin(x) = 1/2",
    )

    if mode == "Solve Equation":
        col1, col2 = st.columns(2)

        with col1:
            lower_text = st.text_input(
                "Interval lower bound",
                value="0",
                key="trig_lower",
            )

        with col2:
            upper_text = st.text_input(
                "Interval upper bound",
                value="2*pi",
                key="trig_upper",
            )

    if st.button(
        "Calculate",
        type="primary",
        use_container_width=True,
        key="trig_calculate",
    ):
        try:
            if not expression.strip():
                st.error("Please enter an expression or equation.")
                return

            if mode == "Simplify Expression":
                expr = parse_user_expression(expression, x)
                best = simplify_candidates(expr)

                st.subheader("Simplification Results")

                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("**Original expression**")
                    st.latex(sp.latex(expr))

                with col2:
                    st.markdown("**Simplified expression**")
                    st.latex(sp.latex(best))

                st.divider()

                st.markdown("**Sine/cosine form**")
                st.latex(
                    sp.latex(
                        sp.trigsimp(expr.rewrite(sp.sin))
                    )
                )

                st.markdown("**Factored form**")
                st.latex(sp.latex(sp.factor(sp.together(expr))))

                if not expr.free_symbols:
                    st.markdown("**Exact value**")
                    st.latex(sp.latex(sp.simplify(expr)))
                    st.write(f"Numerical value: `{sp.N(expr, 12)}`")

            else:
                if expression.count("=") != 1:
                    st.error(
                        "Enter exactly one equals sign, for example "
                        "`sin(x) = 1/2`."
                    )
                    return

                left_text, right_text = expression.split("=", 1)

                lhs = parse_user_expression(left_text, x)
                rhs = parse_user_expression(right_text, x)

                lower = parse_user_expression(lower_text, x)
                upper = parse_user_expression(upper_text, x)

                if lower.free_symbols or upper.free_symbols:
                    st.error(
                        "Interval bounds must be constants, such as "
                        "`0`, `pi`, or `2*pi`."
                    )
                    return

                if lower.is_real is False or upper.is_real is False:
                    st.error("Please use real-valued interval bounds.")
                    return

                if lower.is_finite is False or upper.is_finite is False:
                    st.error("Interval bounds must be finite.")
                    return

                if float(sp.N(lower)) >= float(sp.N(upper)):
                    st.error(
                        "The lower bound must be less than the upper bound."
                    )
                    return

                with st.spinner("Solving equation..."):
                    result = solve_equation(
                        lhs, rhs, x, lo=lower, hi=upper
                    )

                show_equation_results(result)

        except Exception as exc:
            st.error(f"Could not process this input: {exc}")
            st.caption(
                "Check brackets, function names, and operators. "
                "Examples: sin(x), sin(x)=1/2, tan(2*x)=sqrt(3)."
            )

    st.divider()
    st.caption(
        "Educational calculator: symbolic simplification and numerical "
        "root-finding may not resolve every possible expression."
    )


if __name__ == "__main__":
    render()