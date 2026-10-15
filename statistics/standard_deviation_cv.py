
import math
import re
from collections import Counter

import pandas as pd
import streamlit as st


# ============================================================
# NUMBER HANDLING
# ============================================================

def parse_number(text):
    """Parse finite real or complex numbers, such as 3+4j."""
    try:
        value = complex(text.strip())
    except ValueError as exc:
        raise ValueError(f"Invalid number: {text}") from exc

    if not math.isfinite(value.real) or not math.isfinite(value.imag):
        raise ValueError("Numbers must be finite.")

    if value.imag == 0:
        return value.real

    return value


def clean_number(value, tolerance=1e-12):
    if isinstance(value, complex) and abs(value.imag) < tolerance:
        return value.real
    return value


def format_number(value, digits=6):
    if isinstance(value, complex):
        real = 0.0 if abs(value.real) < 1e-12 else value.real
        imag = 0.0 if abs(value.imag) < 1e-12 else value.imag

        if imag == 0:
            return f"{real:.{digits}g}"
        if real == 0:
            return f"{imag:.{digits}g}j"

        sign = "+" if imag >= 0 else "-"
        return f"{real:.{digits}g}{sign}{abs(imag):.{digits}g}j"

    if isinstance(value, float) and math.isnan(value):
        return "Undefined"

    if isinstance(value, float) and math.isinf(value):
        return "Undefined (infinity)"

    if isinstance(value, float):
        return f"{value:.{digits}g}"

    return str(value)


def parse_number_list(text):
    if not text.strip():
        raise ValueError("Please enter at least one value.")

    return [parse_number(item) for item in text.replace(",", " ").split()]


def parse_frequencies(text):
    tokens = text.replace(",", " ").split()

    if not tokens:
        raise ValueError("Please enter frequencies.")

    frequencies = []

    for token in tokens:
        if not re.fullmatch(r"\+?\d+", token):
            raise ValueError(
                "Frequencies must be non-negative integers."
            )
        frequencies.append(int(token))

    if sum(frequencies) <= 0:
        raise ValueError("At least one frequency must be positive.")

    return frequencies


# ============================================================
# STATISTICAL CALCULATIONS
# ============================================================

def calculate_mode(values, frequencies=None):
    counts = Counter()

    if frequencies is None:
        counts.update(values)
    else:
        for value, frequency in zip(values, frequencies):
            if frequency > 0:
                counts[value] += frequency

    if not counts:
        return "Undefined (no observations)"

    highest = max(counts.values())

    if highest == 1 and len(counts) > 1:
        return "No repeated mode"

    modes = [value for value, count in counts.items() if count == highest]

    return modes[0] if len(modes) == 1 else modes


def calculate_statistics(values, frequencies=None):
    if not values:
        raise ValueError("At least one value is required.")

    if frequencies is None:
        frequencies = [1] * len(values)
    elif len(values) != len(frequencies):
        raise ValueError("Values and frequencies must have equal lengths.")

    if any(
        not isinstance(frequency, int) or frequency < 0
        for frequency in frequencies
    ):
        raise ValueError("Frequencies must be non-negative integers.")

    total = sum(frequencies)

    if total <= 0:
        raise ValueError("Total frequency must be positive.")

    mean = sum(
        (frequency * value for value, frequency in zip(values, frequencies)),
        0,
    ) / total

    squared_deviations = [
        frequency * abs(value - mean) ** 2
        for value, frequency in zip(values, frequencies)
    ]

    sum_squared_deviations = math.fsum(squared_deviations)

    population_variance = sum_squared_deviations / total
    population_sd = math.sqrt(max(0.0, population_variance))

    if total > 1:
        sample_variance = sum_squared_deviations / (total - 1)
        sample_sd = math.sqrt(max(0.0, sample_variance))
    else:
        sample_variance = float("nan")
        sample_sd = float("nan")

    mean_magnitude = abs(mean)

    population_cv = (
        population_sd / mean_magnitude * 100
        if mean_magnitude != 0
        else float("nan")
    )

    sample_cv = (
        sample_sd / mean_magnitude * 100
        if mean_magnitude != 0 and total > 1
        else float("nan")
    )

    if any(isinstance(value, complex) for value in values):
        median = "Undefined for complex-valued observations"
    else:
        expanded = []

        for value, frequency in zip(values, frequencies):
            expanded.extend([value] * frequency)

        expanded.sort()
        middle = total // 2

        if total % 2:
            median = expanded[middle]
        else:
            median = (expanded[middle - 1] + expanded[middle]) / 2

    return {
        "Count": total,
        "Mean": clean_number(mean),
        "Median": median,
        "Mode": calculate_mode(values, frequencies),
        "Population Variance": population_variance,
        "Population SD": population_sd,
        "Sample Variance": sample_variance,
        "Sample SD": sample_sd,
        "Population CV (%)": population_cv,
        "Sample CV (%)": sample_cv,
    }


# ============================================================
# CONTINUOUS GROUPED DATA
# ============================================================

def parse_intervals(text):
    pattern = re.compile(
        r"^([+-]?(?:\d+(?:\.\d*)?|\.\d+))"
        r"-"
        r"([+-]?(?:\d+(?:\.\d*)?|\.\d+))$"
    )

    tokens = text.split()

    if not tokens:
        raise ValueError("Enter at least one class interval.")

    intervals = []

    for token in tokens:
        match = pattern.fullmatch(token)

        if not match:
            raise ValueError(
                f"Invalid interval '{token}'. Example: 0-10 or -5--1."
            )

        low, high = float(match.group(1)), float(match.group(2))

        if not math.isfinite(low) or not math.isfinite(high):
            raise ValueError("Interval boundaries must be finite.")

        if high <= low:
            raise ValueError(
                f"Upper boundary must exceed lower boundary: {token}"
            )

        intervals.append((low, high))

    for index in range(1, len(intervals)):
        if intervals[index][0] < intervals[index - 1][1]:
            raise ValueError(
                "Intervals must be ordered and must not overlap."
            )

    return intervals


def calculate_continuous_statistics(intervals, frequencies):
    if len(intervals) != len(frequencies):
        raise ValueError("Interval and frequency counts must match.")

    if not intervals or sum(frequencies) <= 0:
        raise ValueError("At least one observation is required.")

    midpoints = [(low + high) / 2 for low, high in intervals]
    widths = [high - low for low, high in intervals]
    total = sum(frequencies)

    # Mean and SD use class midpoints as approximations.
    results = calculate_statistics(midpoints, frequencies)

    target = total / 2
    cumulative = 0
    previous_cumulative = 0
    median_index = None

    for index, frequency in enumerate(frequencies):
        previous_cumulative = cumulative
        cumulative += frequency

        if cumulative >= target:
            median_index = index
            break

    if median_index is None:
        median = "Undefined"
    else:
        low, high = intervals[median_index]
        frequency = frequencies[median_index]

        if frequency == 0:
            median = "Undefined (zero-frequency median class)"
        else:
            median = low + (
                (target - previous_cumulative) / frequency
            ) * (high - low)

    # Use frequency density to identify the modal class,
    # especially when class widths differ.
    densities = [
        frequency / width
        for frequency, width in zip(frequencies, widths)
    ]

    maximum_density = max(densities)

    tied_indices = [
        index for index, density in enumerate(densities)
        if math.isclose(density, maximum_density)
    ]

    if maximum_density == 0:
        grouped_mode = "Undefined (all class frequencies are zero)"
    elif len(tied_indices) > 1:
        grouped_mode = [
            f"{intervals[index][0]:g}-{intervals[index][1]:g}"
            for index in tied_indices
        ]
    else:
        index = tied_indices[0]
        low, high = intervals[index]

        previous_density = densities[index - 1] if index > 0 else 0
        next_density = (
            densities[index + 1] if index + 1 < len(densities) else 0
        )

        denominator = (
            2 * maximum_density - previous_density - next_density
        )

        if math.isclose(denominator, 0.0):
            grouped_mode = (low + high) / 2
        else:
            grouped_mode = low + (
                (maximum_density - previous_density) / denominator
            ) * (high - low)

    results["Median"] = median
    results["Mode"] = grouped_mode
    results["Data Type"] = "Continuous grouped (estimated)"

    return results


# ============================================================
# STREAMLIT USER INTERFACE
# ============================================================

def render():
    st.title("Standard Deviation & Coefficient of Variation")
    st.write(
        "Calculate population and sample standard deviation, variance, "
        "coefficient of variation (CV), mean, median, and mode."
    )

    data_type = st.radio(
        "Choose your data type",
        [
            "Raw data",
            "Discrete data with frequencies",
            "Continuous grouped data",
        ],
    )

    values_text = ""
    frequencies_text = ""
    intervals_text = ""

    if data_type == "Raw data":
        values_text = st.text_area(
            "Enter observations",
            placeholder="Example: 10 12 15 18 20",
            height=100,
        )

    elif data_type == "Discrete data with frequencies":
        values_text = st.text_area(
            "Enter distinct values",
            placeholder="Example: 10 20 30 40",
            height=80,
        )

        frequencies_text = st.text_area(
            "Enter corresponding frequencies",
            placeholder="Example: 2 5 3 4",
            height=80,
        )

    else:
        intervals_text = st.text_area(
            "Enter class intervals",
            placeholder="Example: 0-10 10-20 20-30 30-40",
            height=80,
        )

        frequencies_text = st.text_area(
            "Enter corresponding frequencies",
            placeholder="Example: 3 7 12 8",
            height=80,
        )

        st.caption(
            "Grouped-data results are estimates based on class midpoints. "
            "Enter intervals in ascending order without overlap."
        )

    statistic_options = [
        "All statistics",
        "Mean",
        "Median",
        "Mode",
        "Population Variance",
        "Population SD",
        "Sample Variance",
        "Sample SD",
        "Population CV (%)",
        "Sample CV (%)",
    ]

    selected_statistic = st.selectbox(
        "Choose the result to focus on",
        statistic_options,
    )

    with st.expander("How are these statistics calculated?"):
        st.markdown(
            """
            - **Population SD:** divides the squared deviations by \(N\).
            - **Sample SD:** divides the squared deviations by \(N-1\).
            - **Population CV:** population SD / absolute mean × 100%.
            - **Sample CV:** sample SD / absolute mean × 100%.
            - **Grouped data:** uses class midpoints to estimate the mean
              and standard deviations.
            """
        )

    if st.button(
        "Calculate Statistics",
        type="primary",
        use_container_width=True,
    ):
        try:
            if data_type == "Raw data":
                values = parse_number_list(values_text)
                results = calculate_statistics(values)

            elif data_type == "Discrete data with frequencies":
                values = parse_number_list(values_text)
                frequencies = parse_frequencies(frequencies_text)

                if len(values) != len(frequencies):
                    raise ValueError(
                        "Enter the same number of values and frequencies."
                    )

                results = calculate_statistics(values, frequencies)

            else:
                intervals = parse_intervals(intervals_text)
                frequencies = parse_frequencies(frequencies_text)

                results = calculate_continuous_statistics(
                    intervals,
                    frequencies,
                )

            st.session_state["sd_cv_results"] = results
            st.session_state["sd_cv_data_type"] = data_type
            st.session_state["sd_cv_selected"] = selected_statistic

        except (ValueError, OverflowError, ZeroDivisionError) as exc:
            st.error(f"Could not calculate statistics: {exc}")
            return

    if "sd_cv_results" not in st.session_state:
        return

    results = st.session_state["sd_cv_results"]
    selected = st.session_state["sd_cv_selected"]
    saved_data_type = st.session_state["sd_cv_data_type"]

    st.subheader("Results")

    # Key summary metrics.
    col1, col2, col3 = st.columns(3)

    col1.metric("Number of observations (N)", results["Count"])
    col2.metric("Mean", format_number(results["Mean"]))
    col3.metric("Population SD", format_number(results["Population SD"]))

    if selected == "All statistics":
        displayed_results = results
    else:
        displayed_results = {
            selected: results[selected]
        }

    rows = []

    for name, value in displayed_results.items():
        if isinstance(value, list):
            formatted = ", ".join(str(item) for item in value)
        else:
            formatted = format_number(value) if isinstance(
                value, (int, float, complex)
            ) else str(value)

        rows.append({
            "Statistic": name,
            "Result": formatted,
        })

    st.dataframe(
        pd.DataFrame(rows),
        use_container_width=True,
        hide_index=True,
    )

    csv_data = pd.DataFrame(
        [
            {"Statistic": name, "Result": format_number(value)
             if isinstance(value, (int, float, complex)) else str(value)}
            for name, value in results.items()
        ]
    ).to_csv(index=False).encode("utf-8")

    st.download_button(
        "Download results as CSV",
        data=csv_data,
        file_name="standard_deviation_cv_results.csv",
        mime="text/csv",
        use_container_width=True,
    )

    st.info(
        "Sample SD and sample CV are undefined when N is less than 2. "
        "CV is undefined when the mean is zero. Complex-valued observations "
        "have no generally defined numerical median."
    )

    if saved_data_type == "Continuous grouped data":
        st.caption(
            "Grouped median uses the grouped median formula. "
            "Grouped mode is estimated using frequency density."
        )


if __name__ == "__main__":
    render()
