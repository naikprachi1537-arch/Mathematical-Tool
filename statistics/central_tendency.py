
import streamlit as st
from decimal import Decimal, InvalidOperation, localcontext
import math


ZERO = Decimal("0")
ONE = Decimal("1")


def parse_decimal(token):
    """Parse a finite real number."""
    try:
        value = Decimal(str(token).strip())
    except (InvalidOperation, ValueError):
        raise ValueError(f"{token!r} is not a valid real number.")

    if not value.is_finite():
        raise ValueError(f"{token!r} must be finite.")

    return value


def arithmetic_mean(data):
    with localcontext() as ctx:
        ctx.prec = 50
        return sum(data, ZERO) / Decimal(len(data))


def median(data):
    ordered = sorted(data)
    n = len(ordered)
    middle = n // 2

    if n % 2:
        return ordered[middle]

    return (ordered[middle - 1] + ordered[middle]) / Decimal(2)


def modes(data):
    counts = {}

    for value in data:
        counts[value] = counts.get(value, 0) + 1

    highest = max(counts.values())

    if highest == 1:
        return None

    return sorted(
        value for value, count in counts.items()
        if count == highest
    )


def geometric_mean(data):
    if any(value < 0 for value in data):
        raise ValueError("Geometric mean requires nonnegative values.")

    if any(value == 0 for value in data):
        return ZERO

    result = math.exp(
        sum(math.log(float(value)) for value in data) / len(data)
    )

    if not math.isfinite(result):
        raise ValueError("Result exceeds the supported numeric range.")

    return Decimal(str(result))


def harmonic_mean(data):
    if any(value == 0 for value in data):
        raise ValueError("Harmonic mean is undefined when a value is zero.")

    with localcontext() as ctx:
        ctx.prec = 50
        denominator = sum((ONE / value for value in data), ZERO)

        if denominator == 0:
            raise ValueError("The sum of reciprocals is zero.")

        return Decimal(len(data)) / denominator


def quadratic_mean(data):
    with localcontext() as ctx:
        ctx.prec = 50
        mean_square = sum(
            (value * value for value in data), ZERO
        ) / Decimal(len(data))

    result = math.sqrt(float(mean_square))

    if not math.isfinite(result):
        raise ValueError("Result exceeds the supported numeric range.")

    return Decimal(str(result))


def weighted_mean(values, weights):
    total_weight = sum(weights)

    if total_weight <= 0:
        raise ValueError("Total weight must be greater than zero.")

    return sum(
        (value * weight for value, weight in zip(values, weights)),
        ZERO,
    ) / total_weight


def frequency_mean(pairs):
    total = sum(frequency for _, frequency in pairs)

    if total <= 0:
        raise ValueError("Total frequency must be greater than zero.")

    return sum(
        (value * Decimal(frequency) for value, frequency in pairs),
        ZERO,
    ) / Decimal(total)


def frequency_geometric_mean(pairs):
    data = [
        value
        for value, frequency in pairs
        for _ in range(frequency)
    ]
    return geometric_mean(data)


def frequency_harmonic_mean(pairs):
    data = [
        value
        for value, frequency in pairs
        for _ in range(frequency)
    ]
    return harmonic_mean(data)


def frequency_quadratic_mean(pairs):
    data = [
        value
        for value, frequency in pairs
        for _ in range(frequency)
    ]
    return quadratic_mean(data)


def grouped_mean(classes):
    total = sum(frequency for _, _, frequency in classes)

    if total <= 0:
        raise ValueError("Total frequency must be greater than zero.")

    weighted_total = sum(
        (
            ((lower + upper) / Decimal(2)) * Decimal(frequency)
            for lower, upper, frequency in classes
        ),
        ZERO,
    )

    return weighted_total / Decimal(total)


def grouped_median(classes):
    total = sum(frequency for _, _, frequency in classes)

    if total <= 0:
        raise ValueError("Total frequency must be greater than zero.")

    target = Decimal(total) / Decimal(2)
    cumulative = 0

    for lower, upper, frequency in classes:
        previous = cumulative
        cumulative += frequency

        if frequency > 0 and Decimal(cumulative) >= target:
            return lower + (
                (target - Decimal(previous)) / Decimal(frequency)
            ) * (upper - lower)

    raise ValueError("Could not identify the median class.")


def grouped_mode(classes):
    if not classes:
        raise ValueError("No classes supplied.")

    max_frequency = max(frequency for _, _, frequency in classes)

    if max_frequency == 0:
        raise ValueError("Total frequency must be greater than zero.")

    modal_indices = [
        i for i, (_, _, frequency) in enumerate(classes)
        if frequency == max_frequency
    ]

    if len(modal_indices) != 1:
        raise ValueError("There are tied modal classes.")

    i = modal_indices[0]
    lower, upper, f1 = classes[i]
    f0 = classes[i - 1][2] if i > 0 else 0
    f2 = classes[i + 1][2] if i < len(classes) - 1 else 0

    denominator = Decimal(2 * f1 - f0 - f2)

    if denominator == 0:
        raise ValueError("Grouped mode formula has a zero denominator.")

    return lower + (
        Decimal(f1 - f0) / denominator
    ) * (upper - lower)


def show_result(label, calculation):
    try:
        value = calculation()

        if value is None:
            st.metric(label, "No mode")
            st.caption("Every observed value occurs only once.")
        else:
            st.metric(label, str(value))

    except (ValueError, ArithmeticError, OverflowError) as exc:
        st.warning(f"{label}: Undefined — {exc}")


def show_mean_results(data):
    st.subheader("Statistical Results")

    col1, col2, col3 = st.columns(3)

    with col1:
        show_result("Arithmetic Mean", lambda: arithmetic_mean(data))
    with col2:
        show_result("Median", lambda: median(data))
    with col3:
        show_result("Mode", lambda: modes(data))

    col4, col5, col6 = st.columns(3)

    with col4:
        show_result("Geometric Mean", lambda: geometric_mean(data))
    with col5:
        show_result("Harmonic Mean", lambda: harmonic_mean(data))
    with col6:
        show_result("Quadratic Mean (RMS)", lambda: quadratic_mean(data))


def render():
    st.title("Central Tendency Calculator")

    st.write(
        "Calculate measures of central tendency for raw observations, "
        "discrete frequency distributions, and grouped frequency data."
    )

    data_type = st.radio(
        "Choose data format",
        [
            "Raw Data",
            "Discrete Frequency Distribution",
            "Grouped Frequency Distribution",
        ],
        key="central_tendency_type",
    )

    if data_type == "Raw Data":
        st.markdown("### Enter observations")

        raw_data = st.text_area(
            "Numbers separated by spaces, commas, or new lines",
            placeholder="Example: 10, 20, 20, 30, 40",
            key="central_tendency_raw",
        )

        if st.button("Calculate Statistics", type="primary"):
            try:
                tokens = raw_data.replace(",", " ").split()

                if not tokens:
                    raise ValueError("Please enter at least one observation.")

                data = [parse_decimal(token) for token in tokens]

                st.write(f"**Number of observations:** {len(data)}")
                show_mean_results(data)

                with st.expander("View sorted data"):
                    st.write([str(value) for value in sorted(data)])

                with st.expander("Calculate weighted mean"):
                    st.caption(
                        "Enter one nonnegative weight for each observation, "
                        "in the same order as the original data."
                    )

                    weights_input = st.text_input(
                        "Weights",
                        placeholder="Example: 1, 2, 1, 3, 1",
                        key="central_tendency_weights",
                    )

                    if st.button(
                        "Calculate Weighted Mean",
                        key="central_tendency_weighted_button",
                    ):
                        try:
                            weight_tokens = (
                                weights_input.replace(",", " ").split()
                            )

                            if len(weight_tokens) != len(data):
                                raise ValueError(
                                    f"Enter exactly {len(data)} weights."
                                )

                            weights = [
                                parse_decimal(token)
                                for token in weight_tokens
                            ]

                            if any(weight < 0 for weight in weights):
                                raise ValueError(
                                    "Weights cannot be negative."
                                )

                            show_result(
                                "Weighted Mean",
                                lambda: weighted_mean(data, weights),
                            )

                        except ValueError as exc:
                            st.error(str(exc))

            except ValueError as exc:
                st.error(str(exc))

    elif data_type == "Discrete Frequency Distribution":
        st.markdown("### Enter values and frequencies")

        count = st.number_input(
            "Number of distinct values",
            min_value=1,
            max_value=30,
            value=3,
            step=1,
            key="central_tendency_discrete_count",
        )

        pairs = []

        for i in range(int(count)):
            col1, col2 = st.columns(2)

            with col1:
                value = st.text_input(
                    f"Value {i + 1}",
                    value=str(i + 1),
                    key=f"ct_discrete_value_{i}",
                )

            with col2:
                frequency = st.number_input(
                    f"Frequency {i + 1}",
                    min_value=0,
                    max_value=1000000,
                    value=1,
                    step=1,
                    key=f"ct_discrete_frequency_{i}",
                )

            pairs.append((value, int(frequency)))

        if st.button("Calculate Frequency Statistics", type="primary"):
            try:
                combined = {}

                for value_text, frequency in pairs:
                    value = parse_decimal(value_text)
                    combined[value] = combined.get(value, 0) + frequency

                pairs_parsed = sorted(combined.items())
                total = sum(frequency for _, frequency in pairs_parsed)

                if total == 0:
                    raise ValueError("Total frequency must be greater than zero.")

                expanded = [
                    value
                    for value, frequency in pairs_parsed
                    for _ in range(frequency)
                ]

                st.metric("Total Frequency", total)

                col1, col2, col3 = st.columns(3)

                with col1:
                    show_result(
                        "Arithmetic Mean",
                        lambda: frequency_mean(pairs_parsed),
                    )
                with col2:
                    show_result("Median", lambda: median(expanded))
                with col3:
                    show_result("Mode", lambda: modes(expanded))

                col4, col5, col6 = st.columns(3)

                with col4:
                    show_result(
                        "Geometric Mean",
                        lambda: frequency_geometric_mean(pairs_parsed),
                    )
                with col5:
                    show_result(
                        "Harmonic Mean",
                        lambda: frequency_harmonic_mean(pairs_parsed),
                    )
                with col6:
                    show_result(
                        "Quadratic Mean (RMS)",
                        lambda: frequency_quadratic_mean(pairs_parsed),
                    )

                with st.expander("View frequency table"):
                    st.dataframe(
                        [
                            {"Value": str(value), "Frequency": frequency}
                            for value, frequency in pairs_parsed
                        ],
                        use_container_width=True,
                        hide_index=True,
                    )

            except ValueError as exc:
                st.error(str(exc))

    else:
        st.markdown("### Enter class intervals")

        count = st.number_input(
            "Number of classes",
            min_value=1,
            max_value=30,
            value=3,
            step=1,
            key="central_tendency_grouped_count",
        )

        classes = []

        for i in range(int(count)):
            st.markdown(f"**Class {i + 1}**")
            col1, col2, col3 = st.columns(3)

            with col1:
                lower = st.text_input(
                    "Lower boundary",
                    value=str(i * 10),
                    key=f"ct_group_lower_{i}",
                )

            with col2:
                upper = st.text_input(
                    "Upper boundary",
                    value=str((i + 1) * 10),
                    key=f"ct_group_upper_{i}",
                )

            with col3:
                frequency = st.number_input(
                    "Frequency",
                    min_value=0,
                    max_value=1000000,
                    value=1,
                    step=1,
                    key=f"ct_group_frequency_{i}",
                )

            classes.append((lower, upper, int(frequency)))

        if st.button("Calculate Grouped Statistics", type="primary"):
            try:
                parsed_classes = []

                for lower_text, upper_text, frequency in classes:
                    lower = parse_decimal(lower_text)
                    upper = parse_decimal(upper_text)

                    if upper <= lower:
                        raise ValueError(
                            "Each upper boundary must exceed its lower boundary."
                        )

                    parsed_classes.append((lower, upper, frequency))

                parsed_classes.sort(key=lambda item: item[0])

                for i in range(1, len(parsed_classes)):
                    if parsed_classes[i][0] < parsed_classes[i - 1][1]:
                        raise ValueError("Class intervals must not overlap.")

                total = sum(f for _, _, f in parsed_classes)

                if total == 0:
                    raise ValueError("Total frequency must be greater than zero.")

                st.metric("Total Frequency", total)
                st.caption(
                    "Grouped median and mode are estimates. "
                    "Grouped geometric, harmonic, and quadratic means use "
                    "class midpoints."
                )

                col1, col2, col3 = st.columns(3)

                with col1:
                    show_result(
                        "Estimated Mean",
                        lambda: grouped_mean(parsed_classes),
                    )

                with col2:
                    show_result(
                        "Estimated Median",
                        lambda: grouped_median(parsed_classes),
                    )

                with col3:
                    show_result(
                        "Estimated Mode",
                        lambda: grouped_mode(parsed_classes),
                    )

                midpoint_pairs = [
                    ((lower + upper) / Decimal(2), frequency)
                    for lower, upper, frequency in parsed_classes
                ]

                col4, col5, col6 = st.columns(3)

                with col4:
                    show_result(
                        "Estimated Geometric Mean",
                        lambda: frequency_geometric_mean(midpoint_pairs),
                    )

                with col5:
                    show_result(
                        "Estimated Harmonic Mean",
                        lambda: frequency_harmonic_mean(midpoint_pairs),
                    )

                with col6:
                    show_result(
                        "Estimated Quadratic Mean",
                        lambda: frequency_quadratic_mean(midpoint_pairs),
                    )

                with st.expander("View grouped frequency table"):
                    st.dataframe(
                        [
                            {
                                "Lower Boundary": str(lower),
                                "Upper Boundary": str(upper),
                                "Midpoint": str((lower + upper) / Decimal(2)),
                                "Frequency": frequency,
                            }
                            for lower, upper, frequency in parsed_classes
                        ],
                        use_container_width=True,
                        hide_index=True,
                    )

            except ValueError as exc:
                st.error(str(exc))


if __name__ == "__main__":
    render()
