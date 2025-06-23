import pandas as pd
import matplotlib.pyplot as plt


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

    # Import numpy for random jitter
    import numpy as np
    
    # Plot each group with horizontal jitter
    for roi in ["cortex", "medulla"]:
        # First, plot the healthy group
        healthy_data = plot_df[(plot_df["roi"] == roi) & (plot_df["display_group"] == "healthy")]
        if len(healthy_data) > 0:
            # Add random jitter to x-positions (± 0.15)
            jitter = np.random.uniform(-0.15, 0.15, size=len(healthy_data))
            ax.scatter(
                [positions[roi]["healthy"] + j for j in jitter],  # x-positions with jitter
                healthy_data["adc"],  # y-values (adc)
                label="healthy" if roi == "cortex" else None,  # Only add to legend once
                color=colors["healthy"],  # color based on group
                alpha=0.7,  # transparency
                s=100,  # marker size
                marker="o"  # use circle marker for all points
            )
            
            # Add half violin plot for healthy
            from scipy import stats
            
            # Calculate kernel density
            kde = stats.gaussian_kde(healthy_data["adc"])
            adc_range = np.linspace(healthy_data["adc"].min(), healthy_data["adc"].max(), 100)
            density = kde(adc_range)
            
            # Scale the density (adjust the scaling factor as needed)
            scaling = 0.3
            density = density / density.max() * scaling
            
            # Plot the half violin on the left side
            ax.fill_betweenx(
                adc_range,
                positions[roi]["healthy"] - density,
                positions[roi]["healthy"],
                color=colors["healthy"],
                alpha=0.3
            )
        
        # Then plot disease groups (vasc and rpgn) at the same position but with different colors
        combined_disease_data = plot_df[(plot_df["roi"] == roi) & (plot_df["position_group"] == "disease")]
        individual_disease_points = []
        
        for disease in ["vasc", "rpgn"]:
            disease_data = plot_df[(plot_df["roi"] == roi) & (plot_df["display_group"] == disease)]
            if len(disease_data) > 0:
                # Add random jitter to x-positions (± 0.15)
                jitter = np.random.uniform(-0.15, 0.15, size=len(disease_data))
                ax.scatter(
                    [positions[roi]["disease"] + j for j in jitter],  # x-positions with jitter
                    disease_data["adc"],  # y-values (adc)
                    label=disease if roi == "cortex" else None,  # Only add to legend once
                    color=colors[disease],  # color based on original group
                    alpha=0.7,  # transparency
                    s=100,  # marker size
                    marker="o"  # use circle marker for all points
                )
                individual_disease_points.extend(disease_data["adc"])
        
        # Add half violin plot for combined disease
        if len(combined_disease_data) > 0:
            # Calculate kernel density
            kde = stats.gaussian_kde(combined_disease_data["adc"])
            adc_range = np.linspace(combined_disease_data["adc"].min(), combined_disease_data["adc"].max(), 100)
            density = kde(adc_range)
            
            # Scale the density (adjust the scaling factor as needed)
            scaling = 0.3
            density = density / density.max() * scaling
            
            # Plot the half violin on the right side
            ax.fill_betweenx(
                adc_range,
                positions[roi]["disease"],
                positions[roi]["disease"] + density,
                color="grey",  # Using grey for combined disease
                alpha=0.3
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

    # Save and show the plot
    plt.tight_layout()
    plt.savefig("adc_scatter_plot.png", dpi=300)
    plt.show()

    print("Scatter plot created and saved as 'adc_scatter_plot.png'")


def main():
    # Load the data
    df = load_data()

    # Calculate means of left and right sides
    mean_df = calculate_side_means(df)

    # Create ADC scatter plot
    create_adc_scatter_plot(mean_df)

    # Optional: Save the results to a new CSV file
    # mean_df.to_csv('data/side_means.csv', index=False)


if __name__ == "__main__":
    main()
