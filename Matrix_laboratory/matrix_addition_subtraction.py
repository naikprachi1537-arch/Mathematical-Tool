
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

x, y, z = sp.symbols("x y z")

LOCAL_DICT = {
    "x": x,
    "y": y,
    "z": z,
    "i": sp.I,
    "j": sp.I,
    "I": sp.I,
    "pi": sp.pi,
    "e": sp.E,
    "E": sp.E,
    "sqrt": sp.sqrt,
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "log": sp.log,
    "exp": sp.exp,
    "abs": sp.Abs,
    "Abs": sp.Abs,
}


def parse_entry(text):
    """Parse and validate one matrix entry."""
    text = str(text).strip()

    if not text:
        raise ValueError("Matrix entries cannot be empty.")

    try:
        value = parse_expr(
            text,
            local_dict=LOCAL_DICT.copy(),
            transformations=TRANSFORMATIONS,
            evaluate=True,
        )

        if value.has(sp.zoo, sp.nan, sp.S.ComplexInfinity):
            raise ValueError("Entries must not be undefined or infinite.")

        if value.is_finite is False:
            raise ValueError("Entries must be finite.")

        return value

    except (SyntaxError, TypeError, ValueError, NameError) as exc:
        raise ValueError(
            f"Cannot interpret '{text}'. Check the expression."
        ) from exc


def render_matrix(matrix, title):
    """Display a SymPy matrix using Streamlit."""
    st.markdown(f"#### {title}")
    st.latex(sp.latex(matrix))


def render():
    st.title("Matrix Addition & Subtraction")
    st.write(
        "Add or subtract two or more matrices with matching dimensions. "
        "Supports fractions, complex numbers, and symbolic expressions."
    )

    st.info(
        "Examples: `2`, `1/3`, `2+3*i`, `x+2`, `sqrt(2)`, `x^2`. "
        "Use `*` for multiplication, such as `2*x`."
    )

    operation = st.radio(
        "Choose operation",
        ["Addition", "Subtraction"],
        horizontal=True,
        key="matrix_add_sub_operation",
    )

    num_matrices = st.number_input(
        "Number of matrices",
        min_value=2,
        max_value=6,
        value=2,
        step=1,
        key="matrix_add_sub_count",
    )

    col1, col2 = st.columns(2)

    with col1:
        rows = st.number_input(
            "Number of rows",
            min_value=1,
            max_value=6,
            value=2,
            step=1,
            key="matrix_add_sub_rows",
        )

    with col2:
        cols = st.number_input(
            "Number of columns",
            min_value=1,
            max_value=6,
            value=2,
            step=1,
            key="matrix_add_sub_cols",
        )

    rows = int(rows)
    cols = int(cols)
    num_matrices = int(num_matrices)

    st.caption(
        f"Each matrix must have {rows} row(s) and {cols} column(s)."
    )

    with st.form("matrix_add_sub_form"):
        matrix_inputs = []

        for m in range(num_matrices):
            st.markdown(f"### Matrix {m + 1}")
            entries = []

            for r in range(rows):
                row_entries = st.columns(cols)
                current_row = []

                for c in range(cols):
                    value = row_entries[c].text_input(
                        f"M{m + 1}[{r + 1},{c + 1}]",
                        value="0",
                        key=f"mas_{m}_{r}_{c}",
                    )
                    current_row.append(value)

                entries.append(current_row)

            matrix_inputs.append(entries)

        show_decimal = st.checkbox(
            "Also show decimal approximation",
            value=False,
        )

        submitted = st.form_submit_button(
            "Calculate",
            type="primary",
            use_container_width=True,
        )

    if submitted:
        matrices = []
        errors = []

        for m, entries in enumerate(matrix_inputs):
            parsed_rows = []

            for r, entry_row in enumerate(entries):
                parsed_row = []

                for c, entry in enumerate(entry_row):
                    try:
                        parsed_row.append(parse_entry(entry))
                    except ValueError as exc:
                        errors.append(
                            f"Matrix {m + 1}, row {r + 1}, "
                            f"column {c + 1}: {exc}"
                        )

                parsed_rows.append(parsed_row)

            if not errors:
                matrices.append(sp.Matrix(parsed_rows))

        if errors:
            st.error("Please correct these entries:")
            for error in errors:
                st.write(f"- {error}")
            return

        result = matrices[0]

        for matrix in matrices[1:]:
            if operation == "Addition":
                result = result + matrix
            else:
                result = result - matrix

        result = result.applyfunc(sp.simplify)

        st.divider()
        st.subheader("Calculation Results")

        with st.expander("View input matrices", expanded=False):
            for index, matrix in enumerate(matrices, start=1):
                render_matrix(matrix, f"Matrix {index}")

        symbols_found = set()
        for entry in result:
            symbols_found.update(entry.free_symbols)

        if operation == "Addition":
            expression = " + ".join(
                f"M{i + 1}" for i in range(num_matrices)
            )
        else:
            expression = " - ".join(
                [f"M1"] + [
                    f"M{i + 1}" for i in range(1, num_matrices)
                ]
            )

        st.markdown(f"**Operation:** {expression}")
        render_matrix(result, "Final Result")

        st.success("Calculation completed successfully!")

        if show_decimal:
            if symbols_found:
                st.warning(
                    "The result contains symbols. Decimal approximation "
                    "requires numerical values for those symbols."
                )
            else:
                try:
                    approximate = result.evalf()
                    render_matrix(approximate, "Decimal Approximation")
                except (TypeError, ValueError, NotImplementedError):
                    st.warning(
                        "A decimal approximation could not be calculated."
                    )


if __name__ == "__main__":
    render()
