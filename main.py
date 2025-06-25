import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from pathlib import Path


def load_data():
    # Define data path using pathlib
    data_path = Path("data") / "vasculitis.data.csv"

    # Read the CSV file with semicolon as separator
    df = pd.read_csv(data_path, sep=";", comment="/")
    # Remove empty columns (those with no column name)
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    # Display information about the dataframe
    print(f"Loaded data with {df.shape[0]} rows and {df.shape[1]} columns")
    print("\nColumn names:")
    for col in df.columns:
        print(f"- {col}")

    print("\nFirst 5 rows:")
    print(df.head())

    # Display group counts
    print("\nGroup counts:")
    print(df["group"].value_counts())

    return df


def calculate_side_means(df):
    # Group by id, group, and roi, then calculate mean of parameters for each side
    # This will create a DataFrame with the mean values of both sides for each parameter
    side_means = (
        df.groupby(["id", "group", "roi"])[["adc", "fa", "asl", "t2star"]]
        .mean()
        .reset_index()
    )

    print("\nMean values of left and right sides:")
    print(side_means.head(10))

    return side_means


def create_scatter_plot(
    ax, data, x_position, color, label, jitter_range=0.05, x_offset=-0.25
):
    """Create a scatter plot with jitter at the specified position."""
    if len(data) > 0:
        jitter = np.random.uniform(-jitter_range, jitter_range, size=len(data))
        ax.scatter(
            [x_position + x_offset + j for j in jitter],
            data,
            label=label,
            color=color,
            alpha=0.7,
            s=100,
            marker="o",
        )


def create_violin_plot(
    ax, data, x_position, color, alpha=0.3, scaling=0.3, x_offset=0.05
):
    """Create a half violin plot at the specified position with extremely smooth ends."""
    if len(data) > 0:
        # Calculate data range with padding for smoother ends
        data_range = max(data) - min(data)
        min_val = min(data) - data_range * 0.1  # 10% padding
        max_val = max(data) + data_range * 0.1  # 10% padding

        # Create high-resolution data range
        x_points = np.linspace(min_val, max_val, 200)

        # Generate smooth KDE
        kde = stats.gaussian_kde(
            data, bw_method="scott"
        )  # Scott's rule often works well
        density = kde(x_points)

        # Create smooth tapering with cosine function for perfect continuity
        # This ensures no abrupt transitions at junction points

        # Calculate taper regions (20% at top and bottom)
        bottom_points = int(len(x_points) * 0.2)
        top_points = int(len(x_points) * 0.2)

        # Create smooth tapering functions using cosine (perfect for smooth transitions)
        # Cosine goes from -1 to 1, so we transform to 0 to 1
        bottom_taper = (1 - np.cos(np.linspace(0, np.pi, bottom_points))) / 2
        top_taper = (1 + np.cos(np.linspace(0, np.pi, top_points))) / 2

        # Apply the tapering
        density[:bottom_points] *= bottom_taper
        density[-top_points:] *= top_taper

        # Scale the density
        density = density / density.max() * scaling

        # Plot the half violin with interpolation for smooth rendering
        ax.fill_betweenx(
            x_points,
            x_position + x_offset,
            x_position + x_offset + density,
            color=color,
            alpha=alpha,
            interpolate=True,  # Enable interpolation for smoother curves
        )


def perform_significance_test(healthy_data, disease_data):
    """Perform statistical test to compare healthy vs disease groups.

    Returns:
        tuple: (p_value, significance_level) where significance_level is:
               '***' for p < 0.001
               '**' for p < 0.01
               '*' for p < 0.05
               'ns' for not significant
    """
    # Use Mann-Whitney U test (non-parametric alternative to t-test)
    # This is appropriate for comparing distributions that may not be normal
    stat, p_value = stats.mannwhitneyu(healthy_data, disease_data)

    # Determine significance level
    if p_value < 0.001:
        return p_value, "***"
    elif p_value < 0.01:
        return p_value, "**"
    elif p_value < 0.05:
        return p_value, "*"
    else:
        return p_value, "ns"


def add_significance_indicator(
    ax, x1, x2, y, significance, text_y_offset=0.02, line_height=0.01
):
    """Add a significance indicator (line and star) between two groups.

    Args:
        ax: The matplotlib axis to add the indicator to
        x1: x-coordinate of the first group
        x2: x-coordinate of the second group
        y: y-coordinate for the top of the line
        significance: The significance marker ('*', '**', '***', or 'ns')
        text_y_offset: Vertical offset for the significance text
        line_height: Height of the vertical portions of the bracket
    """
    # Don't add indicator for non-significant results
    if significance == "ns":
        return

    # Draw the horizontal line
    ax.plot([x1, x2], [y, y], color="black", linewidth=1)

    # Draw the vertical end caps
    ax.plot([x1, x1], [y - line_height, y], color="black", linewidth=1)
    ax.plot([x2, x2], [y - line_height, y], color="black", linewidth=1)

    # Add the significance text
    text_x = (x1 + x2) / 2
    ax.text(
        text_x,
        y + text_y_offset,
        significance,
        horizontalalignment="center",
        verticalalignment="bottom",
        fontsize=12,
    )


def create_adc_scatter_plot(mean_df):
    # Keep original groups but rename 'control' to 'healthy' for clarity
    mean_df["display_group"] = mean_df["group"].apply(
        lambda x: "healthy" if x == "control" else x
    )

    # Create a combined disease category while preserving original groups for coloring
    mean_df["position_group"] = mean_df["group"].apply(
        lambda x: "healthy" if x == "control" else "disease"
    )

    # Filter to include only the groups we want
    plot_df = mean_df[mean_df["display_group"].isin(["healthy", "vasc", "rpgn"])]

    # Set up the figure
    fig, ax = plt.subplots(figsize=(8, 6))  # Reduced width from 12 to 8

    # Define positions for the groups on x-axis (healthy and disease only)
    positions = {
        "cortex": {"healthy": 1, "disease": 1.5},  # Reduced from 2 to 1.5
        "medulla": {"healthy": 2.5, "disease": 3},  # Reduced from 3.5,4.5 to 2.5,3
    }

    # Define colors for each original group
    colors = {
        "healthy": "green",
        "vasc": "#FF7F50",  # Coral/light red-orange
        "rpgn": "#007fbf",  # Sea green/petrol blue
    }

    # Store data for significance testing
    significance_data = {}

    # Plot each group with horizontal jitter
    for roi in ["cortex", "medulla"]:
        # Plot the healthy group - scatter on left, violin on right
        healthy_data = plot_df[
            (plot_df["roi"] == roi) & (plot_df["display_group"] == "healthy")
        ]
        if len(healthy_data) > 0:
            # Create scatter plot for healthy group
            create_scatter_plot(
                ax=ax,
                data=healthy_data["adc"],
                x_position=positions[roi]["healthy"],
                color=colors["healthy"],
                label="healthy" if roi == "cortex" else None,
                jitter_range=0.05,
                x_offset=-0.05,
            )

            # Create violin plot for healthy group
            create_violin_plot(
                ax=ax,
                data=healthy_data["adc"],
                x_position=positions[roi]["healthy"],
                color=colors["healthy"],
                alpha=0.3,
                scaling=0.3,
                x_offset=0.05,
            )

            # Store healthy data for significance testing
            significance_data[roi] = {"healthy": healthy_data["adc"].values}

        # Plot disease groups - scatter on left, violin on right
        combined_disease_data = plot_df[
            (plot_df["roi"] == roi) & (plot_df["position_group"] == "disease")
        ]

        # Store combined disease data for significance testing
        if len(combined_disease_data) > 0:
            significance_data[roi]["disease"] = combined_disease_data["adc"].values

        # Create scatter plots for individual disease groups
        for disease in ["vasc", "rpgn"]:
            disease_data = plot_df[
                (plot_df["roi"] == roi) & (plot_df["display_group"] == disease)
            ]
            if len(disease_data) > 0:
                create_scatter_plot(
                    ax=ax,
                    data=disease_data["adc"],
                    x_position=positions[roi]["disease"],
                    color=colors[disease],
                    label=disease if roi == "cortex" else None,
                    jitter_range=0.05,
                    x_offset=-0.05,
                )

        # Create violin plot for combined disease groups
        if len(combined_disease_data) > 0:
            create_violin_plot(
                ax=ax,
                data=combined_disease_data["adc"],
                x_position=positions[roi]["disease"],
                color="#6a5acd",  # Light violet/thistle color
                alpha=0.3,
                scaling=0.3,
                x_offset=0.05,
            )

    # Add significance indicators for each region
    y_max = plot_df["adc"].max()
    y_range = plot_df["adc"].max() - plot_df["adc"].min()

    # Calculate positions for significance indicators
    for roi in ["cortex", "medulla"]:
        if (
            roi in significance_data
            and "healthy" in significance_data[roi]
            and "disease" in significance_data[roi]
        ):
            # Perform statistical test
            p_value, sig_marker = perform_significance_test(
                significance_data[roi]["healthy"], significance_data[roi]["disease"]
            )

            # Only add indicator if there is significance
            if sig_marker != "ns":
                # Calculate y position for the indicator (above the highest point)
                y_pos = y_max + 0.05 * y_range

                # Print significance test results
                print(
                    f"{roi.capitalize()} - Healthy vs. Disease: p={p_value:.4f} {sig_marker}"
                )

                # Add the significance indicator
                add_significance_indicator(
                    ax=ax,
                    x1=positions[roi]["healthy"],
                    x2=positions[roi]["disease"],
                    y=y_pos,
                    significance=sig_marker,
                    text_y_offset=0.02 * y_range,
                    line_height=0.01 * y_range,
                )

    # Customize the plot
    ax.set_ylabel("ADC Values (x10^-6 mm²/s)")
    ax.set_title("ADC Values by Region and Group")

    # Set x-ticks and labels
    ax.set_xticks([1.25, 2.75])  # Centered between the groups
    ax.set_xticklabels(["Cortex", "Medulla"])

    # Add a legend with custom ordering
    ax.legend(title="Groups")

    # Remove the top and right spines to create a cleaner look
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Optional: Make the bottom and left spines a bit thinner
    ax.spines["bottom"].set_linewidth(0.5)
    ax.spines["left"].set_linewidth(0.5)

    # Ensure y-axis extends high enough to show significance indicators
    if "cortex" in significance_data or "medulla" in significance_data:
        ax.set_ylim(top=y_max + 0.15 * y_range)

    # Show grid
    # ax.grid(True, linestyle="--", alpha=0.7)

    # Save and optionally show the plot
    plt.tight_layout()

    # Create the output directory if it doesn't exist using pathlib
    output_dir = Path("out")
    output_dir.mkdir(exist_ok=True)

    # Define output file paths
    png_path = output_dir / "adc_scatter_plot.png"
    svg_path = output_dir / "adc_scatter_plot.svg"

    # Save in both PNG and SVG formats
    plt.savefig(png_path, dpi=300)
    plt.savefig(svg_path, format="svg")

    # Try to show the plot interactively
    try:
        plt.show()
    except Exception as e:
        print(f"Could not display plot interactively: {e}")
        print(f"Plot saved as '{png_path}' and '{svg_path}'")

    plt.close()  # Close the figure to free memory

    print(f"Scatter plot created and saved in the '{output_dir}' directory as:")
    print(f"- {png_path.name} (raster format)")
    print(f"- {svg_path.name} (vector format)")


def main():
    # Load the data
    df = load_data()

    # Calculate means of left and right sides
    mean_df = calculate_side_means(df)

    # Create ADC scatter plot
    matplotlib.use("TkAgg")  # or 'Qt5Agg' if you have PyQt5 installed
    create_adc_scatter_plot(mean_df)

    # Optional: Save the results to a new CSV file
    # output_path = Path("data") / "side_means.csv"
    # mean_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
