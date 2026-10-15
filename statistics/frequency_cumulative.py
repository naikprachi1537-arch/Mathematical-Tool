
import math
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


INTERVAL_PATTERN = re.compile(
    r"^\s*"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
    r"\s*-\s*"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
    r"\s*$"
)


def parse_numbers(text):
    """Parse a non-empty list of finite real numbers."""
    if not text.strip():
        raise ValueError("Please enter at least one observation.")

    try:
        values = np.array(
            [float(item) for item in text.replace(",", " ").split()],
            dtype=float,
        )
    except ValueError as exc:
        raise ValueError(
            "Enter valid numbers separated by spaces or commas."
        ) from exc

    if not np.all(np.isfinite(values)):
        raise ValueError("NaN and infinity are not allowed.")

    return values


def parse_intervals(text):
    """Parse ordered, non-overlapping class intervals."""
    intervals = []

    for item in text.split():
        match = INTERVAL_PATTERN.fullmatch(item)

        if not match:
            raise ValueError(
                f"Invalid interval '{item}'. Examples: "
                "0-5, 5-10, 10-15, or -10--5."
            )

        low = float(match.group(1))
        high = float(match.group(2))

        if not math.isfinite(low) or not math.isfinite(high):
            raise ValueError("Interval boundaries must be finite.")

        if low >= high:
            raise ValueError(
                f"The lower boundary must be below the upper boundary: {item}"
            )

        intervals.append((low, high))

    if not intervals:
        raise ValueError("Please enter at least one class interval.")

    for i in range(1, len(intervals)):
        if intervals[i][0] < intervals[i - 1][1]:
            raise ValueError(
                "Intervals must be ordered and must not overlap."
            )

    return intervals


def get_data(data_type, continuous_method, raw_text, interval_text, freq_text):
    """Create the distribution data from the selected input method."""

    if data_type in ("Raw data", "Discrete data"):
        values = parse_numbers(raw_text)

        if data_type == "Discrete data":
            if not np.all(values == np.floor(values)):
                st.info(
                    "Discrete data usually represent countable values. "
                    "Your numeric values will still be counted."
                )

        unique_values, frequencies = np.unique(
            values, return_counts=True
        )

        return {
            "data_type": data_type,
            "values": unique_values,
            "frequencies": frequencies.astype(np.int64),
            "intervals": None,
            "sample_size": len(values),
        }

    if continuous_method == "Automatic class intervals":
        values = parse_numbers(raw_text)

        if len(values) < 1:
            raise ValueError("Please enter observations.")

        # Sturges' rule for selecting the number of classes.
        bins = max(1, math.ceil(1 + 3.322 * math.log10(len(values))))
        frequencies, edges = np.histogram(values, bins=bins)

        intervals = [
            (float(edges[i]), float(edges[i + 1]))
            for i in range(len(edges) - 1)
        ]

        return {
            "data_type": data_type,
            "values": None,
            "frequencies": frequencies.astype(np.int64),
            "intervals": intervals,
            "sample_size": len(values),
        }

    intervals = parse_intervals(interval_text)
    tokens = freq_text.replace(",", " ").split()

    if not tokens:
        raise ValueError("Please enter a frequency for each interval.")

    if not all(re.fullmatch(r"\+?\d+", item) for item in tokens):
        raise ValueError("Frequencies must be non-negative integers.")

    frequencies = [int(item) for item in tokens]

    if len(intervals) != len(frequencies):
        raise ValueError(
            "The number of intervals must match the number of frequencies."
        )

    if sum(frequencies) <= 0:
        raise ValueError("The total frequency must be greater than zero.")

    return {
        "data_type": data_type,
        "values": None,
        "frequencies": np.asarray(frequencies, dtype=np.int64),
        "intervals": intervals,
        "sample_size": sum(frequencies),
    }


def build_frequency_table(data):
    """Build frequency, relative frequency, percentage and CF columns."""
    frequencies = np.asarray(data["frequencies"], dtype=np.int64)
    total = int(frequencies.sum())

    if total <= 0:
        raise ValueError("Total frequency must be positive.")

    less_cf = np.cumsum(frequencies)
    more_cf = np.cumsum(frequencies[::-1])[::-1]
    relative = frequencies / total

    table = pd.DataFrame({
        "Frequency": frequencies,
        "Relative Frequency": relative,
        "Percentage Frequency (%)": relative * 100,
        "CF (Less Than)": less_cf,
        "CF (More Than)": more_cf,
        "Relative CF (Less Than)": less_cf / total,
        "Relative CF (More Than)": more_cf / total,
        "Percentage CF (Less Than)": less_cf / total * 100,
        "Percentage CF (More Than)": more_cf / total * 100,
    })

    if data["intervals"] is None:
        table.insert(0, "Value", data["values"])
    else:
        lower = np.array([item[0] for item in data["intervals"]])
        upper = np.array([item[1] for item in data["intervals"]])
        widths = upper - lower
        midpoints = (lower + upper) / 2

        table.insert(
            0,
            "Class Interval",
            [f"{low:g} - {high:g}" for low, high in data["intervals"]],
        )
        table.insert(1, "Lower Boundary", lower)
        table.insert(2, "Upper Boundary", upper)
        table.insert(3, "Class Width", widths)
        table.insert(4, "Midpoint", midpoints)
        table["Frequency Density"] = frequencies / widths

    return table


def select_table_columns(table, table_type, is_grouped):
    """Select columns for the requested table."""
    if is_grouped:
        base = [
            "Class Interval",
            "Lower Boundary",
            "Upper Boundary",
            "Class Width",
            "Midpoint",
            "Frequency",
            "Frequency Density",
        ]
    else:
        base = ["Value", "Frequency"]

    if table_type == "Frequency distribution":
        return [column for column in base if column in table.columns]

    if table_type == "Less-than cumulative frequency":
        extra = [
            "CF (Less Than)",
            "Relative CF (Less Than)",
            "Percentage CF (Less Than)",
        ]
        return [column for column in base + extra if column in table.columns]

    if table_type == "More-than cumulative frequency":
        extra = [
            "CF (More Than)",
            "Relative CF (More Than)",
            "Percentage CF (More Than)",
        ]
        return [column for column in base + extra if column in table.columns]

    return list(table.columns)


def plot_graph(data, graph_choice):
    """Create the selected graph or all applicable graphs."""
    frequencies = np.asarray(data["frequencies"], dtype=float)
    total = frequencies.sum()

    if total <= 0:
        st.warning("Cannot plot a distribution with zero total frequency.")
        return

    is_grouped = data["intervals"] is not None

    if is_grouped:
        lower = np.array([item[0] for item in data["intervals"]])
        upper = np.array([item[1] for item in data["intervals"]])
        x = (lower + upper) / 2
    else:
        x = np.asarray(data["values"], dtype=float)
        lower = upper = None

    if graph_choice == "Recommended graph":
        graph_types = ["Histogram" if is_grouped else "Frequency bar chart"]
    elif graph_choice == "All applicable graphs":
        graph_types = [
            "Frequency bar chart",
            "Histogram",
            "Frequency polygon",
            "Less-than ogive",
            "More-than ogive",
            "Relative-frequency graph",
            "Percentage-frequency graph",
            "Relative cumulative-frequency graph",
            "Percentage cumulative-frequency graph",
        ]
    else:
        graph_types = [graph_choice]

    for graph_type in graph_types:
        if graph_type in ("Frequency polygon", "Less-than ogive", "More-than ogive"):
            if not is_grouped:
                st.info(
                    f"{graph_type} requires class intervals. "
                    "This graph was skipped for raw/discrete data."
                )
                continue

        fig, ax = plt.subplots(figsize=(9, 5))

        if graph_type == "Frequency bar chart":
            if is_grouped:
                ax.bar(
                    x,
                    frequencies,
                    width=upper - lower,
                    edgecolor="black",
                    align="center",
                )
                ax.set_xlabel("Class Midpoint")
            else:
                ax.bar(x, frequencies, edgecolor="black")
                ax.set_xticks(x)
                ax.set_xlabel("Value")

            ax.set_ylabel("Frequency")
            ax.set_title("Frequency Bar Chart")

        elif graph_type == "Histogram":
            if is_grouped:
                ax.bar(
                    lower,
                    frequencies / (upper - lower),
                    width=upper - lower,
                    align="edge",
                    edgecolor="black",
                )
                ax.set_xlabel("Class Boundaries")
                ax.set_ylabel("Frequency Density")
                ax.set_title("Frequency-Density Histogram")
            else:
                markerline, stemlines, baseline = ax.stem(
                    x, frequencies, basefmt=" "
                )
                plt.setp(markerline, markersize=6)
                ax.set_xlabel("Value")
                ax.set_ylabel("Frequency")
                ax.set_xticks(x)
                ax.set_title("Frequency Stem Plot")

        elif graph_type == "Frequency polygon":
            ax.plot(x, frequencies, marker="o", linewidth=2)
            ax.set_xlabel("Class Midpoint")
            ax.set_ylabel("Frequency")
            ax.set_title("Frequency Polygon")

        elif graph_type in ("Less-than ogive", "More-than ogive"):
            if graph_type == "Less-than ogive":
                cumulative = np.cumsum(frequencies)
                x_points = np.concatenate(([lower[0]], upper))
                y_points = np.concatenate(([0], cumulative))
            else:
                cumulative = np.cumsum(frequencies[::-1])[::-1]
                x_points = np.concatenate((lower, [upper[-1]]))
                y_points = np.concatenate((cumulative, [0]))

            ax.plot(x_points, y_points, marker="o", linewidth=2)
            ax.set_xlabel("Class Boundary")
            ax.set_ylabel("Cumulative Frequency")
            ax.set_title(graph_type)

        elif graph_type in (
            "Relative-frequency graph",
            "Percentage-frequency graph",
        ):
            if graph_type == "Relative-frequency graph":
                y = frequencies / total
                ax.set_ylabel("Relative Frequency")
            else:
                y = frequencies / total * 100
                ax.set_ylabel("Percentage Frequency (%)")

            if is_grouped:
                ax.bar(
                    x, y, width=upper - lower,
                    edgecolor="black", align="center"
                )
                ax.set_xlabel("Class Midpoint")
            else:
                ax.bar(x, y, edgecolor="black")
                ax.set_xticks(x)
                ax.set_xlabel("Value")

            ax.set_title(graph_type)

        elif graph_type in (
            "Relative cumulative-frequency graph",
            "Percentage cumulative-frequency graph",
        ):
            multiplier = (
                100 if graph_type == "Percentage cumulative-frequency graph"
                else 1
            )

            less = np.cumsum(frequencies) / total * multiplier
            more = np.cumsum(frequencies[::-1])[::-1] / total * multiplier

            if is_grouped:
                less_x = np.concatenate(([lower[0]], upper))
                less_y = np.concatenate(([0], less))
                more_x = np.concatenate((lower, [upper[-1]]))
                more_y = np.concatenate((more, [0]))
                ax.set_xlabel("Class Boundary")
            else:
                less_x, less_y = x, less
                more_x, more_y = x, more
                ax.set_xlabel("Value")

            ax.plot(less_x, less_y, marker="o", label="Less Than")
            ax.plot(more_x, more_y, marker="o", label="More Than")
            ax.set_ylabel(
                "Cumulative Percentage (%)"
                if multiplier == 100
                else "Relative Cumulative Frequency"
            )
            ax.set_title(graph_type)
            ax.legend()

        ax.grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)


def render():
    """Render the calculator inside the main Streamlit application."""
    st.title("Frequency & Cumulative Frequency Distribution")
    st.write(
        "Create frequency tables, calculate relative and cumulative "
        "frequencies, and generate statistical graphs."
    )

    data_type = st.selectbox(
        "Choose data type",
        ["Raw data", "Discrete data", "Continuous / grouped data"],
    )

    raw_text = ""
    interval_text = ""
    freq_text = ""
    continuous_method = "Automatic class intervals"

    if data_type in ("Raw data", "Discrete data"):
        raw_text = st.text_area(
            "Enter observations",
            placeholder="Example: 12 15 15 18 20 20 20 25",
            height=100,
        )
    else:
        continuous_method = st.radio(
            "How would you like to enter the grouped data?",
            ["Automatic class intervals", "Manual class intervals and frequencies"],
        )

        if continuous_method == "Automatic class intervals":
            raw_text = st.text_area(
                "Enter observations",
                placeholder="Example: 12 15 15 18 20 20 25 28 30",
                height=100,
            )
        else:
            interval_text = st.text_area(
                "Enter class intervals separated by spaces",
                placeholder="Example: 0-5 5-10 10-15 15-20",
                height=80,
            )
            freq_text = st.text_input(
                "Enter corresponding frequencies",
                placeholder="Example: 2 5 7 4",
            )

    st.subheader("Output options")

    table_type = st.selectbox(
        "Choose table type",
        [
            "Frequency distribution",
            "Less-than cumulative frequency",
            "More-than cumulative frequency",
            "Complete frequency table",
        ],
    )

    graph_choice = st.selectbox(
        "Choose graph",
        [
            "Recommended graph",
            "Frequency bar chart",
            "Histogram",
            "Frequency polygon",
            "Less-than ogive",
            "More-than ogive",
            "Relative-frequency graph",
            "Percentage-frequency graph",
            "Relative cumulative-frequency graph",
            "Percentage cumulative-frequency graph",
            "All applicable graphs",
        ],
    )

    if st.button("Calculate Distribution", type="primary", use_container_width=True):
        try:
            data = get_data(
                data_type,
                continuous_method,
                raw_text,
                interval_text,
                freq_text,
            )

            table = build_frequency_table(data)
            is_grouped = data["intervals"] is not None

            st.session_state["frequency_cumulative_data"] = data
            st.session_state["frequency_cumulative_table"] = table
            st.session_state["frequency_cumulative_table_type"] = table_type
            st.session_state["frequency_cumulative_graph_choice"] = graph_choice

        except (ValueError, OverflowError) as exc:
            st.error(str(exc))
            return

    if "frequency_cumulative_table" in st.session_state:
        data = st.session_state["frequency_cumulative_data"]
        table = st.session_state["frequency_cumulative_table"]
        saved_table_type = st.session_state["frequency_cumulative_table_type"]
        saved_graph_choice = st.session_state["frequency_cumulative_graph_choice"]

        total = int(np.asarray(data["frequencies"]).sum())

        st.subheader("Summary")
        col1, col2, col3 = st.columns(3)
        col1.metric("Total frequency (N)", f"{total}")
        col2.metric("Number of classes/values", len(data["frequencies"]))
        col3.metric(
            "Data type",
            "Grouped" if data["intervals"] is not None else data["data_type"],
        )

        st.subheader("Frequency Table")

        columns = select_table_columns(
            table,
            saved_table_type,
            data["intervals"] is not None,
        )

        st.dataframe(
            table[columns].round(4),
            use_container_width=True,
            hide_index=True,
        )

        csv_data = table.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download complete table as CSV",
            data=csv_data,
            file_name="frequency_distribution.csv",
            mime="text/csv",
            use_container_width=True,
        )

        st.subheader("Distribution Graph")
        plot_graph(data, saved_graph_choice)


if __name__ == "__main__":
    render()
