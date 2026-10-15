
import streamlit as st
import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    convert_xor,
    implicit_multiplication_application,
)

TRANSFORMATIONS = standard_transformations + (
    convert_xor,
    implicit_multiplication_application,
)

x, y, z = sp.symbols("x y z")

LOCAL_DICT = {
    "x": x,
    "y": y,
    "z": z,
    "I": sp.I,
    "i": sp.I,
    "j": sp.I,
    "pi": sp.pi,
    "E": sp.E,
    "e": sp.E,
    "sqrt": sp.sqrt,
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "log": sp.log,
    "exp": sp.exp,
    "Abs": sp.Abs,
    "abs": sp.Abs,
}


def parse_entry(value):
    """Parse and validate one numerical or symbolic matrix entry."""
    value = value.strip()

    if not value:
        raise ValueError("Matrix entries cannot be empty.")

    try:
        expression = parse_expr(
            value,
            local_dict=LOCAL_DICT.copy(),
            transformations=TRANSFORMATIONS,
            evaluate=True,
        )

        if expression.has(sp.zoo, sp.nan, sp.oo, -sp.oo, sp.S.ComplexInfinity):
            raise ValueError("Entry cannot contain infinity or undefined values.")

        if expression.is_commutative is False:
            raise ValueError("Matrix entries must be scalar expressions.")

        return expression

    except (SyntaxError, TypeError, ValueError, NameError) as exc:
        raise ValueError(
            f"Cannot interpret '{value}'. Check the expression."
        ) from exc


def matrix_to_latex(matrix):
    """Convert a SymPy matrix to LaTeX for display."""
    return sp.latex(matrix)


def render():
    st.title("Matrix Determinant Calculator")
    st.write(
        "Calculate the determinant of square matrices using symbolic "
        "mathematics. Supports real numbers, complex numbers, fractions, "
        "and symbolic expressions."
    )

    st.info(
        "Examples: `2`, `-3`, `1/2`, `2+3*I`, `sqrt(2)`, `x+1`, `x^2`. "
        "Use `I` or `i` for the imaginary unit."
    )

    total = st.number_input(
        "Number of matrices",
        min_value=1,
        max_value=5,
        value=1,
        step=1,
        key="determinant_total",
    )

    total = int(total)
    matrix_inputs = []

    for m in range(total):
        st.markdown(f"### Matrix {m + 1}")

        size = st.number_input(
            f"Matrix {m + 1} size (N × N)",
            min_value=1,
            max_value=6,
            value=2,
            step=1,
            key=f"determinant_size_{m}",
        )

        size = int(size)
        entries = []

        for r in range(size):
            columns = st.columns(size)
            row = []

            for c in range(size):
                value = columns[c].text_input(
                    f"M{m + 1}[{r + 1},{c + 1}]",
                    value="1" if r == c else "0",
                    key=f"determinant_entry_{m}_{r}_{c}",
                )
                row.append(value)

            entries.append(row)

        matrix_inputs.append(entries)

    calculate = st.button(
        "Calculate Determinants",
        type="primary",
        use_container_width=True,
    )

    if not calculate:
        return

    st.divider()
    st.subheader("Results")

    for index, entries in enumerate(matrix_inputs, start=1):
        st.markdown(f"### Matrix {index}")

        matrix_rows = []
        errors = []

        for r, row in enumerate(entries):
            parsed_row = []

            for c, value in enumerate(row):
                try:
                    parsed_row.append(parse_entry(value))
                except ValueError as exc:
                    errors.append(
                        f"Row {r + 1}, column {c + 1}: {exc}"
                    )

            matrix_rows.append(parsed_row)

        if errors:
            st.error("Please correct the following entries:")
            for error in errors:
                st.write(f"- {error}")
            continue

        try:
            matrix = sp.Matrix(matrix_rows)

            st.markdown("**Entered Matrix**")
            st.latex(matrix_to_latex(matrix))

            determinant = sp.simplify(
                matrix.det(method="domain-ge")
            )

            st.markdown("**Determinant**")
            st.latex(
                r"\det(A_{" + str(index) + r"}) = "
                + sp.latex(determinant)
            )

            if determinant == 0 or determinant.is_zero is True:
                st.error(
                    "This matrix is singular because its determinant is zero."
                )
            elif determinant.is_zero is False:
                st.success(
                    "This matrix is nonsingular because its determinant "
                    "is nonzero."
                )
            else:
                st.info(
                    "The determinant depends on the symbols or assumptions. "
                    "Singularity cannot be determined conclusively."
                )

            with st.expander("View result details"):
                st.write("Exact result:")
                st.code(str(determinant))

                if not determinant.free_symbols:
                    try:
                        st.write("Numerical approximation:")
                        st.code(str(sp.N(determinant, 12)))
                    except (TypeError, ValueError):
                        pass

        except (ValueError, TypeError, ArithmeticError) as exc:
            st.error(f"Could not calculate the determinant: {exc}")

        st.divider()


if __name__ == "__main__":
    render()
