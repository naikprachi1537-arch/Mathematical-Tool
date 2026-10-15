
import streamlit as st


def parse_number(value):
    """Parse real or complex numbers, accepting i or j."""
    value = value.strip().replace("i", "j")

    if not value:
        raise ValueError("Please enter a number.")

    return complex(value)


def format_number(value):
    """Format real and complex results cleanly."""
    real = value.real
    imag = value.imag

    if abs(real) < 1e-10:
        real = 0.0
    if abs(imag) < 1e-10:
        imag = 0.0

    def fmt(number):
        return f"{number:.4f}".rstrip("0").rstrip(".")

    if imag == 0:
        return fmt(real)

    if real == 0:
        if abs(imag - 1) < 1e-10:
            return "j"
        if abs(imag + 1) < 1e-10:
            return "-j"
        return f"{fmt(imag)}j"

    sign = "+" if imag > 0 else "-"
    imag_abs = abs(imag)

    imag_text = (
        "j" if abs(imag_abs - 1) < 1e-10
        else f"{fmt(imag_abs)}j"
    )

    return f"{fmt(real)}{sign}{imag_text}"


def multiply_matrices(A, B):
    """Multiply two compatible matrices."""
    if len(A[0]) != len(B):
        raise ValueError(
            f"Multiplication is not possible: the first matrix has "
            f"{len(A[0])} columns, but the second matrix has "
            f"{len(B)} rows."
        )

    rows_A = len(A)
    cols_A = len(A[0])
    cols_B = len(B[0])

    result = [
        [0j for _ in range(cols_B)]
        for _ in range(rows_A)
    ]

    for i in range(rows_A):
        for j in range(cols_B):
            for k in range(cols_A):
                result[i][j] += A[i][k] * B[k][j]

    return result


def matrix_to_latex(matrix):
    """Convert a matrix to readable LaTeX."""
    rows = []

    for row in matrix:
        rows.append(
            " & ".join(
                r"\left(" + format_number(value) + r"\right)"
                for value in row
            )
        )

    return r"\begin{bmatrix}" + r" \\ ".join(rows) + r"\end{bmatrix}"


def render():
    st.title("Matrix Multiplication")
    st.write(
        "Multiply two or more matrices in sequence. "
        "Supports real and complex numbers."
    )

    st.info(
        "Enter values such as `2`, `3.5`, `2+3j`, `-4j`, or `1+2i`. "
        "Use `i` or `j` for the imaginary unit."
    )

    num_matrices = st.number_input(
        "Number of matrices",
        min_value=2,
        max_value=6,
        value=2,
        step=1,
        key="matrix_mult_count",
    )

    num_matrices = int(num_matrices)

    st.markdown("### Matrix dimensions")

    dimensions = []

    for index in range(num_matrices):
        st.markdown(f"**Matrix {index + 1}**")
        col1, col2 = st.columns(2)

        with col1:
            rows = st.number_input(
                f"Rows — Matrix {index + 1}",
                min_value=1,
                max_value=6,
                value=2,
                step=1,
                key=f"matrix_mult_rows_{index}",
            )

        with col2:
            cols = st.number_input(
                f"Columns — Matrix {index + 1}",
                min_value=1,
                max_value=6,
                value=2,
                step=1,
                key=f"matrix_mult_cols_{index}",
            )

        dimensions.append((int(rows), int(cols)))

    for index in range(1, num_matrices):
        previous_cols = dimensions[index - 1][1]
        current_rows = dimensions[index][0]

        if previous_cols != current_rows:
            st.error(
                f"Matrix {index + 1} must have {previous_cols} rows "
                f"because Matrix {index} has {previous_cols} columns. "
                "Please adjust the dimensions before calculating."
            )
            return

    with st.form("matrix_multiplication_form"):
        matrix_inputs = []

        for m, (rows, cols) in enumerate(dimensions):
            st.markdown(f"### Enter Matrix {m + 1}")
            entries = []

            for r in range(rows):
                row_columns = st.columns(cols)
                current_row = []

                for c in range(cols):
                    value = row_columns[c].text_input(
                        f"M{m + 1}[{r + 1},{c + 1}]",
                        value="0",
                        key=f"matrix_mult_entry_{m}_{r}_{c}",
                    )
                    current_row.append(value)

                entries.append(current_row)

            matrix_inputs.append(entries)

        submitted = st.form_submit_button(
            "Multiply Matrices",
            type="primary",
            use_container_width=True,
        )

    if not submitted:
        return

    matrices = []
    errors = []

    for m, entries in enumerate(matrix_inputs):
        matrix = []

        for r, row in enumerate(entries):
            parsed_row = []

            for c, value in enumerate(row):
                try:
                    parsed_row.append(parse_number(value))
                except ValueError:
                    errors.append(
                        f"Matrix {m + 1}, row {r + 1}, column {c + 1}: "
                        f"'{value}' is not a valid number."
                    )

            matrix.append(parsed_row)

        matrices.append(matrix)

    if errors:
        st.error("Please correct the following entries:")
        for error in errors:
            st.write(f"- {error}")
        return

    result = matrices[0]

    for index, matrix in enumerate(matrices[1:], start=2):
        try:
            result = multiply_matrices(result, matrix)
        except ValueError as exc:
            st.error(str(exc))
            return

    st.divider()
    st.subheader("Results")

    with st.expander("View input matrices", expanded=False):
        for index, matrix in enumerate(matrices, start=1):
            st.markdown(f"**Matrix {index}**")
            st.latex(matrix_to_latex(matrix))

    st.markdown("### Final Product Matrix")
    st.latex(matrix_to_latex(result))
    st.success("Matrix multiplication completed successfully!")

    st.caption(
        f"Result dimensions: {len(result)} × {len(result[0])}"
    )

    result_text = "\n".join(
        ["\t".join(format_number(value) for value in row)
         for row in result]
    )

    st.download_button(
        "Download result as TXT",
        data=result_text,
        file_name="matrix_multiplication_result.txt",
        mime="text/plain",
        use_container_width=True,
    )


if __name__ == "__main__":
    render()
