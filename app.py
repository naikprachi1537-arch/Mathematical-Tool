
import importlib.util
import re
import sys
from pathlib import Path

import streamlit as st


# --------------------------------------------------
# PAGE SETTINGS
# --------------------------------------------------

st.set_page_config(
    page_title="Mathematical Tool",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).resolve().parent


# --------------------------------------------------
# CALCULATOR REGISTRY
# Labels are shown in the sidebar.
# Filenames are searched for inside the project folder.
# --------------------------------------------------

TOOLS = {
    "Function Plotter": [
        "Function Identifier",
        "Function Plotter",
        "Function Mapping & Domain & Range",
    ],
    "Calculus Visualizer": [
        "Derivative Calculator",
        "Integral Calculator",
        "Trigonometric Calculator",
        "Limit & Continuity",
    ],
    "Matrix Laboratory": [
        "Matrix Addition & Subtraction",
        "Matrix Multiplication",
        "Matrix Determinant",
        "Matrix Inverse",
    ],
    "Statistics": [
        "Central Tendency",
        "Standard Deviation & CV",
        "Frequency & Cumulative Frequency Distribution",
    ],
}

CALCULATOR_FILES = {
    "Function Identifier": [
        "function_identifier.py",
        "Function_identifier.py",
    ],
    "Function Plotter": [
        "function_plotter.py",
        "Function_plotter.py",
    ],
    "Function Mapping & Domain & Range": [
        "function_mapping_domain_range.py",
        "function_mapping.py",
        "domain_range.py",
    ],
    "Derivative Calculator": [
        "derivative_calculator.py",
    ],
    "Integral Calculator": [
        "integral_calculator.py",
    ],
    "Trigonometric Calculator": [
        "trigonometric_calculator.py",
    ],
    "Limit & Continuity": [
        "limit_continuity.py",
    ],
    "Matrix Addition & Subtraction": [
        "matrix_addition_subtraction.py",
    ],
    "Matrix Multiplication": [
        "matrix_multiplication.py",
    ],
    "Matrix Determinant": [
        "matrix_determinant.py",
    ],
    "Matrix Inverse": [
        "matrix_inverse.py",
    ],
    "Central Tendency": [
        "central_tendency.py",
    ],
    "Standard Deviation & CV": [
        "standard_deviation_cv.py",
    ],
    "Frequency & Cumulative Frequency Distribution": [
        "frequency_cumulative.py",
        "frequency_cumulative_frequency.py",
    ],
}

# Supports both the new render() function and older function names.
FUNCTION_NAMES = {
    "Function Identifier": [
        "render",
        "show_function_identifier",
    ],
    "Function Plotter": [
        "render",
        "show_function_plotter",
    ],
}


# --------------------------------------------------
# FIND AND LOAD A CALCULATOR
# --------------------------------------------------

def normalize_filename(filename):
    return re.sub(r"[^a-z0-9]", "", filename.lower())


def find_calculator_file(tool_name):
    filenames = CALCULATOR_FILES.get(tool_name, [])
    wanted_names = [normalize_filename(name) for name in filenames]

    # Search the project and its subfolders.
    for wanted in wanted_names:
        for path in BASE_DIR.rglob("*.py"):
            if path.name.startswith("."):
                continue

            if normalize_filename(path.name) == wanted:
                # Avoid accidentally loading this app itself.
                if path.resolve() != Path(__file__).resolve():
                    return path

    return None


def load_calculator(tool_name):
    file_path = find_calculator_file(tool_name)

    if file_path is None:
        return None, (
            f"Could not find the Python file for **{tool_name}**. "
            "Check that the file exists and its filename matches "
            "the expected name."
        )

    module_name = (
        "calculator_"
        + re.sub(r"[^a-zA-Z0-9_]", "_", tool_name)
    )

    try:
        spec = importlib.util.spec_from_file_location(
            module_name, file_path
        )

        if spec is None or spec.loader is None:
            return None, f"Could not load `{file_path.name}`."

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        function_names = FUNCTION_NAMES.get(
            tool_name, ["render"]
        )

        for function_name in function_names:
            calculator_function = getattr(
                module, function_name, None
            )

            if callable(calculator_function):
                return calculator_function, None

        return None, (
            f"`{file_path.name}` loaded, but it needs a "
            "`render()` function to display in the website."
        )

    except Exception as error:
        return None, (
            f"An error occurred while loading `{file_path.name}`: "
            f"{error}"
        )


# --------------------------------------------------
# WEBSITE DESIGN
# --------------------------------------------------

st.markdown(
    """
    <style>
    .stApp {
        background-color: #0b1020;
        color: #f3f5ff;
    }

    h1, h2, h3, p, label {
        color: #f3f5ff !important;
    }

    div[data-testid="stMetric"] {
        background: #171f35;
        border: 1px solid #303b59;
        padding: 18px;
        border-radius: 14px;
    }

    div.stButton > button {
        border-radius: 10px;
        min-height: 45px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# NAVIGATION CALLBACKS
# --------------------------------------------------

def select_section(name):
    st.session_state["sidebar_section"] = name


def go_home():
    st.session_state["sidebar_section"] = None


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("📐 Mathematical Tool")
st.sidebar.caption("Your all-in-one mathematics workspace")

selected_section = st.sidebar.radio(
    "Choose a section",
    list(TOOLS.keys()),
    index=None,
    key="sidebar_section",
)

st.sidebar.divider()

if selected_section:
    selected_tool = st.sidebar.radio(
        "Choose a calculator",
        TOOLS[selected_section],
        key=f"tool_{selected_section}",
    )
else:
    selected_tool = None


# --------------------------------------------------
# HOMEPAGE
# --------------------------------------------------

st.title("📐 Mathematical Tool")
st.subheader("Explore mathematics. Calculate with confidence.")
st.write(
    "One workspace for functions, calculus, matrices, and statistics."
)

st.divider()

section_details = [
    (
        "📈",
        "Function Plotter",
        "Explore functions, graphs, mappings, domain, and range.",
    ),
    (
        "∫",
        "Calculus Visualizer",
        "Work with derivatives, integrals, limits, and trigonometry.",
    ),
    (
        "▦",
        "Matrix Laboratory",
        "Perform matrix operations, determinants, and inverses.",
    ),
    (
        "📊",
        "Statistics",
        "Calculate central tendency, dispersion, and frequency tables.",
    ),
]

columns = st.columns(2)

for index, (emoji, name, description) in enumerate(section_details):
    with columns[index % 2]:
        with st.container(border=True):
            st.markdown(f"## {emoji} {name}")
            st.write(description)
            st.caption(f"{len(TOOLS[name])} calculators")

            st.button(
                f"Explore {name}",
                key=f"open_{name}",
                use_container_width=True,
                on_click=select_section,
                args=(name,),
            )


# --------------------------------------------------
# CALCULATOR DISPLAY
# --------------------------------------------------

st.divider()

if selected_section and selected_tool:
    st.header(f"🧮 {selected_tool}")
    st.caption(f"Section: {selected_section}")

    calculator_function, error_message = load_calculator(
        selected_tool
    )

    if error_message:
        st.warning(error_message)
    else:
        try:
            calculator_function()
        except Exception as error:
            st.error(
                f"Something went wrong while displaying "
                f"{selected_tool}: {error}"
            )

    st.divider()

    st.button(
        "← Back to Home",
        on_click=go_home,
    )

else:
    st.subheader("Your mathematics workspace")
    st.write(
        "Choose a section from the sidebar or click a card above "
        "to explore its calculators."
    )
