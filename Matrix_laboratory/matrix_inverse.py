
import streamlit as st
import numpy as np


def parse_number(value):
    """Parse finite real or complex numbers, accepting i or j."""
    value = value.strip().replace("I", "j").replace("i", "j")

    if not value:
        raise ValueError("Enter a number.")

    number = complex(value)

    if not np.isfinite(number.real) or not np.isfinite(number.imag):
        raise ValueError("NaN and infinity are not allowed.")

    return number


def format_number(value, decimals=6):
    """Format a complex number without unnecessary imaginary parts."""
    real = float(np.real(value))
    imag = float(np.imag(value))

    if abs(real) < 1e-12:
        real = 0.0
    if abs(imag) < 1e-12:
        imag = 0.0

    def fmt(number):
        return f"{number:.{decimals}f}".rstrip("0").rstrip(".") or "0"

    if imag == 0:
        return fmt(real)

    if real == 0:
        if abs(imag - 1) < 1e-12:
            return "j"
        if abs(imag + 1) < 1e-12:
            return "-j"
        return f"{fmt(imag)}j"

    sign = "+" if imag > 0 else "-"
    imag_text = (
        "j" if abs(abs(imag) - 1) < 1e-12
        else f"{fmt(abs(imag))}j"
    )
    return f"{fmt(real)}{sign}{imag_text}"


def matrix_to_latex(matrix):
    """Format a NumPy matrix for Streamlit's LaTeX renderer."""
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
    st.title("Matrix Inverse Calculator")
    st.write(
        "Calculate the inverse of square matrices containing real or "
        "complex numbers. Check numerical stability and verify the result."
    )

    st.info(
        "Examples: `2`, `-2.5`, `3+4j`, `2-7j`, `4j`, or `3+4i`. "
        "Use `i` or `j` for the imaginary unit."
    )

    total_matrices = st.number_input(
        "How many matrices do you want to process?",
        min_value=1,
        max_value=5,
        value=1,
        step=1,
        key="matrix_inverse_count",
    )

    total_matrices = int(total_matrices)
    all_inputs = []

    for m in range(total_matrices):
        st.markdown(f"### Matrix {m + 1}")

        dimension_col1, dimension_col2 = st.columns(2)

        with dimension_col1:
            rows = st.number_input(
                f"Rows — Matrix {m + 1}",
                min_value=1,
                max_value=6,
                value=2,
                step=1,
                key=f"matrix_inverse_rows_{m}",
            )

        with dimension_col2:
            cols = st.number_input(
                f"Columns — Matrix {m + 1}",
                min_value=1,
                max_value=6,
                value=2,
                step=1,
                key=f"matrix_inverse_cols_{m}",
            )

        rows = int(rows)
        cols = int(cols)

        if rows != cols:
            st.warning(
                "This matrix is not square, so its two-sided inverse "
                "does not exist. You can still submit it to see the result."
            )

        entries = []

        for r in range(rows):
            entry_cols = st.columns(cols)
            row_entries = []

            for c in range(cols):
                value = entry_cols[c].text_input(
                    f"M{m + 1}[{r + 1},{c + 1}]",
                    value="1" if r == c else "0",
                    key=f"matrix_inverse_entry_{m}_{r}_{c}",
                )
                row_entries.append(value)

            entries.append(row_entries)

        all_inputs.append((rows, cols, entries))

    calculate = st.button(
        "Calculate Inverse",
        type="primary",
        use_container_width=True,
    )

    if not calculate:
        return

    st.divider()
    st.subheader("Results")

    for index, (rows, cols, entries) in enumerate(all_inputs, start=1):
        st.markdown(f"### Matrix {index}")

        parsed_rows = []
        errors = []

        for r, row in enumerate(entries):
            parsed_row = []

            for c, value in enumerate(row):
                try:
                    parsed_row.append(parse_number(value))
                except (ValueError, OverflowError):
                    errors.append(
                        f"Row {r + 1}, column {c + 1}: "
                        f"'{value}' is not a valid finite number."
                    )

            parsed_rows.append(parsed_row)

        if errors:
            st.error("Please correct the following entries:")
            for error in errors:
                st.write(f"- {error}")
            continue

        matrix = np.array(parsed_rows, dtype=np.complex128)

        st.markdown("**Original Matrix**")
        st.latex(matrix_to_latex(matrix))

        if rows != cols:
            st.error(
                "An inverse does not exist for a non-square matrix."
            )
            continue

        try:
            if not np.all(np.isfinite(matrix)):
                st.error("The matrix contains non-finite values.")
                continue

            singular_values = np.linalg.svd(
                matrix, compute_uv=False
            )

            if len(singular_values) == 0 or singular_values[0] == 0:
                st.error(
                    "This is a singular matrix. Its inverse does not exist."
                )
                continue

            condition_number = np.linalg.cond(matrix)

            if (
                not np.isfinite(condition_number)
                or condition_number > 1e12
            ):
                st.warning(
                    "This matrix is singular or nearly singular. "
                    "Any computed inverse may be numerically unreliable."
                )

            inverse = np.linalg.solve(
                matrix,
                np.eye(rows, dtype=np.complex128),
            )

            if not np.all(np.isfinite(inverse)):
                st.error(
                    "Numerical overflow occurred while computing the inverse."
                )
                continue

            verification_error = np.linalg.norm(
                matrix @ inverse - np.eye(rows, dtype=np.complex128),
                ord=np.inf,
            )

            col1, col2 = st.columns(2)

            with col1:
                st.metric(
                    "Condition number",
                    f"{condition_number:.6g}",
                )

            with col2:
                st.metric(
                    "Verification error",
                    f"{verification_error:.3e}",
                )

            st.markdown("**Inverse Matrix**")
            st.latex(matrix_to_latex(inverse))

            if verification_error < 1e-8:
                st.success("Verification passed: A × A⁻¹ is approximately I.")
            else:
                st.warning(
                    "Verification accuracy may be insufficient. "
                    "Check the matrix's numerical conditioning."
                )

            result_text = "\n".join(
                [
                    "\t".join(format_number(value) for value in row)
                    for row in inverse
                ]
            )

            st.download_button(
                f"Download inverse of Matrix {index}",
                data=result_text,
                file_name=f"matrix_{index}_inverse.txt",
                mime="text/plain",
                key=f"matrix_inverse_download_{index}",
            )

        except np.linalg.LinAlgError:
            st.error(
                "The inverse could not be computed. "
                "The matrix may be singular or numerically unstable."
            )
        except (ValueError, FloatingPointError, OverflowError) as exc:
            st.error(f"Calculation error: {exc}")


if __name__ == "__main__":
    render()
