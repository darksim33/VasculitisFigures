import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def load_data():
    # Read the CSV file with semicolon as separator
    df = pd.read_csv("data/vasculitis.data.csv", sep=";", comment="/")
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
    """Create a half violin plot at the specified position."""
    if len(data) > 0:
        kde = stats.gaussian_kde(data)
        data_range = np.linspace(data.min(), data.max(), 100)
        density = kde(data_range)

        # Scale the density
        density = density / density.max() * scaling

        # Plot the half violin
        ax.fill_betweenx(
            data_range,
            x_position + x_offset,  # Start from center-right
            x_position + x_offset + density,  # Extend right
            color=color,
            alpha=alpha,
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
    fig, ax = plt.subplots(figsize=(12, 6))

    # Define positions for the groups on x-axis (healthy and disease only)
    positions = {
        "cortex": {"healthy": 1, "disease": 2},
        "medulla": {"healthy": 4, "disease": 5},
    }

    # Define colors for each original group
    colors = {"healthy": "green", "vasc": "orange", "rpgn": "blue"}

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

        # Plot disease groups - scatter on left, violin on right
        combined_disease_data = plot_df[
            (plot_df["roi"] == roi) & (plot_df["position_group"] == "disease")
        ]

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
                color="grey",
                alpha=0.3,
                scaling=0.3,
                x_offset=0.05,
            )

    # Customize the plot
    ax.set_ylabel("ADC Values (x10^-6 mm²/s)")
    ax.set_title("ADC Values by Region and Group")

    # Set x-ticks and labels
    ax.set_xticks([1.5, 4.5])
    ax.set_xticklabels(["Cortex", "Medulla"])

    # Add a legend with custom ordering
    ax.legend(title="Groups")

    # Show grid
    ax.grid(True, linestyle="--", alpha=0.7)

    # Save and optionally show the plot
    plt.tight_layout()
    plt.savefig("adc_scatter_plot.png", dpi=300)

    # Try to show the plot interactively
    try:
        plt.show()
    except Exception as e:
        print(f"Could not display plot interactively: {e}")
        print(
            "Plot saved as 'adc_scatter_plot.png' - you can open it with an image viewer"
        )

    plt.close()  # Close the figure to free memory

    print("Scatter plot created and saved as 'adc_scatter_plot.png'")


def main():
    # Load the data
    df = load_data()

    # Calculate means of left and right sides
    mean_df = calculate_side_means(df)

    # Create ADC scatter plot
    matplotlib.use("TkAgg")  # or 'Qt5Agg' if you have PyQt5 installed
    create_adc_scatter_plot(mean_df)

    # Optional: Save the results to a new CSV file
    # mean_df.to_csv('data/side_means.csv', index=False)


if __name__ == "__main__":
    main()
