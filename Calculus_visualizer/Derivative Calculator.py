import ast
import streamlit as st
import sympy as sp

FUNCTIONS = {
    "sin": sp.sin, "cos": sp.cos, "tan": sp.tan,
    "sec": sp.sec, "csc": sp.csc, "cot": sp.cot,
    "asin": sp.asin, "acos": sp.acos, "atan": sp.atan,
    "sinh": sp.sinh, "cosh": sp.cosh, "tanh": sp.tanh,
    "asinh": sp.asinh, "acosh": sp.acosh, "atanh": sp.atanh,
    "log": sp.log, "ln": sp.log, "exp": sp.exp,
    "sqrt": sp.sqrt, "Abs": sp.Abs, "abs": sp.Abs,
    "floor": sp.floor, "ceiling": sp.ceiling,
    "factorial": sp.factorial,
}

CONSTANTS = {
    "pi": sp.pi,
    "E": sp.E,
    "I": sp.I,
    "oo": sp.oo,
    "EulerGamma": sp.EulerGamma,
}


def parse_expression(expression, symbols):
    """Parse a restricted mathematical expression without eval()."""
    if not expression or not expression.strip():
        raise ValueError("Expression cannot be empty.")

    if len(expression) > 500:
        raise ValueError("Expression is too long (maximum 500 characters).")

    try:
        tree = ast.parse(expression.replace("^", "**"), mode="eval")
    except SyntaxError as error:
        raise ValueError("Invalid mathematical syntax.") from error

    def convert(node):
        if isinstance(node, ast.Constant):
            if isinstance(node.value, bool) or not isinstance(
                node.value, (int, float)
            ):
                raise ValueError("Only numeric constants are allowed.")

            if isinstance(node.value, int):
                return sp.Integer(node.value)
            return sp.Float(node.value)

        if isinstance(node, ast.Name):
            if node.id in CONSTANTS:
                return CONSTANTS[node.id]

            if node.id in FUNCTIONS:
                raise ValueError(
                    f"Write {node.id} with parentheses, e.g. {node.id}(x)."
                )

            if node.id.startswith("__"):
                raise ValueError("Invalid variable name.")

            return symbols.setdefault(node.id, sp.Symbol(node.id))

        if isinstance(node, ast.BinOp):
            left = convert(node.left)
            right = convert(node.right)

            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                return left / right
            if isinstance(node.op, ast.Pow):
                return left ** right
            if isinstance(node.op, ast.Mod):
                return sp.Mod(left, right)

            raise ValueError("Unsupported arithmetic operator.")

        if isinstance(node, ast.UnaryOp):
            value = convert(node.operand)

            if isinstance(node.op, ast.UAdd):
                return value
            if isinstance(node.op, ast.USub):
                return -value

            raise ValueError("Unsupported unary operator.")

        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Only approved math functions are allowed.")

            name = node.func.id

            if name not in FUNCTIONS:
                raise ValueError(f"Unsupported function: {name}")

            if node.keywords:
                raise ValueError("Keyword arguments are not supported.")

            args = [convert(arg) for arg in node.args]

            if name == "factorial" and len(args) != 1:
                raise ValueError("factorial() requires one argument.")

            if name in ("log", "ln"):
                if len(args) not in (1, 2):
                    raise ValueError("log() accepts one argument and an optional base.")
            elif len(args) != 1:
                raise ValueError(f"{name}() requires one argument.")

            return FUNCTIONS[name](*args)

        raise ValueError(
            "Unsupported expression. Use numbers, variables, "
            "arithmetic operators, and approved functions."
        )

    return convert(tree.body)


def get_variable(name, symbols):
    name = name.strip()

    if not name or not name.isidentifier():
        raise ValueError("Enter a valid variable name, such as x or theta.")

    if name in FUNCTIONS or name in CONSTANTS:
        raise ValueError("That name is reserved for a function or constant.")

    return symbols.setdefault(name, sp.Symbol(name))


def render():
    """Main Streamlit entry point called by app.py."""

    st.title("∫ Derivative Calculator")
    st.write(
        "Calculate symbolic derivatives using multiple differentiation methods."
    )

    st.caption(
        "Use * for multiplication, ** or ^ for powers, "
        "and parentheses for function arguments."
    )

    operation = st.selectbox(
        "Choose a differentiation method",
        [
            "Standard Derivative",
            "Implicit Differentiation",
            "Higher-Order Derivative",
            "Logarithmic Differentiation",
            "Parametric Differentiation",
            "Partial Derivatives",
            "Derivative at a Point",
            "Input Syntax Guide",
        ],
    )

    if operation == "Input Syntax Guide":
        st.subheader("Input Syntax Guide")
        st.markdown(
            """
            | Operation | Example |
            |---|---|
            | Multiplication | `3*x`, `x*sin(x)` |
            | Powers | `x**2`, `sec(x)**2` |
            | Division | `x/(x+1)` |
            | Trigonometry | `sin(x)`, `cos(x)`, `tan(x)` |
            | Inverse trigonometry | `asin(x)`, `acos(x)`, `atan(x)` |
            | Hyperbolic | `sinh(x)`, `cosh(x)` |
            | Logarithms | `log(x)`, `ln(x)`, `log(x, 2)` |
            | Exponential | `exp(x)` |
            | Square root | `sqrt(x)` |
            | Absolute value | `Abs(x)` |
            | Constants | `pi`, `E`, `I` |
            | Multiple variables | `x**2 + y**2` |
            """
        )
        return

    with st.form("derivative_calculator_form"):
        if operation == "Implicit Differentiation":
            expression_text = st.text_input(
                "Enter f(x, y) = 0 as an expression",
                value="x**2 + y**2 - 25",
            )
            variable_name = "x"
            order = 1
            point_text = ""
            second_expression_text = ""

        elif operation == "Parametric Differentiation":
            expression_text = st.text_input(
                "Enter x(t)", value="cos(t)"
            )
            second_expression_text = st.text_input(
                "Enter y(t)", value="sin(t)"
            )
            variable_name = "t"
            order = 1
            point_text = ""

        else:
            default_expression = (
                "x**2 + 3*x"
                if operation != "Partial Derivatives"
                else "x**2 + x*y + y**2"
            )

            expression_text = st.text_input(
                "Enter mathematical expression",
                value=default_expression,
            )

            second_expression_text = ""

            if operation == "Partial Derivatives":
                variable_name = st.text_input(
                    "Variables (separate with commas)",
                    value="x,y",
                )
            else:
                variable_name = st.text_input(
                    "Differentiate with respect to",
                    value="x",
                )

            if operation == "Higher-Order Derivative":
                order = st.number_input(
                    "Derivative order",
                    min_value=1,
                    max_value=20,
                    value=2,
                    step=1,
                )
            else:
                order = 1

            if operation == "Derivative at a Point":
                point_text = st.text_input(
                    "Point at which to evaluate the derivative",
                    value="2",
                )
            else:
                point_text = ""

        submitted = st.form_submit_button(
            "Calculate Derivative",
            type="primary",
            use_container_width=True,
        )

    if not submitted:
        st.info("Enter your expression and click Calculate Derivative.")
        return

    symbols = {
        "x": sp.Symbol("x"),
        "y": sp.Symbol("y"),
        "t": sp.Symbol("t"),
    }

    try:
        if operation == "Parametric Differentiation":
            t = symbols["t"]
            x_expr = parse_expression(expression_text, symbols)
            y_expr = parse_expression(second_expression_text, symbols)

            if x_expr.free_symbols - {t} or y_expr.free_symbols - {t}:
                raise ValueError(
                    "Parametric expressions must use only t."
                )

            dx_dt = sp.simplify(sp.diff(x_expr, t))
            dy_dt = sp.simplify(sp.diff(y_expr, t))

            st.subheader("Parametric Differentiation")
            st.latex(r"x(t) = " + sp.latex(x_expr))
            st.latex(r"y(t) = " + sp.latex(y_expr))
            st.latex(r"\frac{dx}{dt} = " + sp.latex(dx_dt))
            st.latex(r"\frac{dy}{dt} = " + sp.latex(dy_dt))

            if dx_dt == 0:
                st.warning(
                    "dx/dt is identically zero; the usual dy/dx ratio "
                    "cannot be calculated."
                )
            else:
                result = sp.simplify(dy_dt / dx_dt)
                st.success("Parametric derivative calculated.")
                st.latex(
                    r"\frac{dy}{dx} = "
                    + sp.latex(result)
                )
                st.caption(
                    "The ratio requires dx/dt to be nonzero at the point considered."
                )

        elif operation == "Partial Derivatives":
            expression = parse_expression(expression_text, symbols)
            names = [
                name.strip()
                for name in variable_name.split(",")
                if name.strip()
            ]

            if not names:
                raise ValueError("Enter at least one variable.")

            if len(set(names)) != len(names):
                raise ValueError("Enter distinct variable names.")

            st.subheader("Partial Derivatives")
            st.latex(r"f = " + sp.latex(expression))

            for name in names:
                variable = get_variable(name, symbols)
                result = sp.simplify(sp.diff(expression, variable))
                st.markdown(f"**Partial derivative with respect to {name}:**")
                st.latex(
                    r"\frac{\partial f}{\partial "
                    + sp.latex(variable)
                    + "} = "
                    + sp.latex(result)
                )

        elif operation == "Implicit Differentiation":
            expression = parse_expression(expression_text, symbols)
            x = symbols["x"]
            y = symbols["y"]

            if y not in expression.free_symbols:
                raise ValueError("The expression must contain y.")

            result = sp.simplify(sp.idiff(expression, y, x))

            st.subheader("Implicit Differentiation")
            st.latex(sp.latex(expression) + " = 0")
            st.success("Implicit derivative calculated.")
            st.latex(r"\frac{dy}{dx} = " + sp.latex(result))

        else:
            expression = parse_expression(expression_text, symbols)
            variable = get_variable(variable_name, symbols)

            if operation == "Higher-Order Derivative":
                result = sp.simplify(
                    sp.diff(expression, variable, int(order))
                )
                derivative_label = (
                    r"\frac{d^{" + str(int(order)) + r"}}{d"
                    + sp.latex(variable)
                    + r"^{" + str(int(order)) + "}}"
                )

            else:
                result = sp.simplify(sp.diff(expression, variable))
                derivative_label = (
                    r"\frac{d}{d" + sp.latex(variable) + "}"
                )

            st.subheader("Derivative Result")
            st.markdown("**Original expression**")
            st.latex(sp.latex(expression))

            st.markdown("**Derivative**")
            st.latex(
                derivative_label + r"\left(" + sp.latex(expression)
                + r"\right) = " + sp.latex(result)
            )

            if operation == "Logarithmic Differentiation":
                st.markdown("**Logarithmic differentiation details**")

                if expression == 0:
                    st.info(
                        "The logarithmic method cannot be applied to "
                        "the zero function. The direct derivative is shown above."
                    )
                else:
                    try:
                        log_derivative = sp.simplify(
                            sp.diff(sp.log(expression), variable)
                        )
                        st.latex(
                            r"\frac{d}{d" + sp.latex(variable)
                            + r"}\ln(f) = "
                            + sp.latex(log_derivative)
                        )
                        st.caption(
                            "This logarithmic identity requires a domain "
                            "where the logarithm is defined and f is nonzero."
                        )
                    except Exception:
                        st.info(
                            "A logarithmic form could not be established; "
                            "the direct derivative is still available above."
                        )

            if operation == "Derivative at a Point":
                point_symbols = symbols.copy()
                point = parse_expression(point_text, point_symbols)
                value = sp.simplify(result.subs(variable, point))

                st.markdown("**Derivative evaluated at the point**")
                st.latex(
                    sp.latex(variable) + " = " + sp.latex(point)
                )
                st.latex(
                    r"f'\left(" + sp.latex(point) + r"\right) = "
                    + sp.latex(value)
                )

        with st.expander("Calculation details"):
            st.write("Input:", expression_text)
            st.write("Method:", operation)
            st.caption(
                "Symbolic results may require domain restrictions "
                "for particular functions or points."
            )

    except Exception as error:
        st.error(f"Could not calculate the derivative: {error}")
        st.info(
            "Check the expression, parentheses, variable names, "
            "and function syntax. Use * for multiplication."
        )


# Optional standalone Streamlit entry point.
if __name__ == "__main__":
    render()