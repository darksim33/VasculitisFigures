import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from pathlib import Path


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


def perform_significance_test(group1_data, group2_data):
    """Perform statistical test to compare two groups.

    Returns:
        tuple: (p_value, significance_level) where significance_level is:
               '***' for p < 0.001
               '**' for p < 0.01
               '*' for p < 0.05
               'ns' for not significant
    """
    # Use Mann-Whitney U test (non-parametric alternative to t-test)
    # This is appropriate for comparing distributions that may not be normal
    stat, p_value = stats.mannwhitneyu(group1_data, group2_data)

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


def get_parameter_info():
    """
    Get display information for each parameter

    Returns:
        dict: Information for each parameter
    """
    return {
        "adc": {
            "title": "ADC Values",
            "ylabel": r"ADC [$\times 10^{-6}$ mm²/s]",
        },
        "fa": {"title": "FA Values", "ylabel": "FA [a.u.]"},
        "asl": {"title": "ASL Values", "ylabel": "ASL [ml/100g/min]"},
        "t2star": {"title": "T2* Values", "ylabel": "T2* [ms]"},
    }


def get_positions(include_control=True):
    """
    Define positions for the groups on x-axis

    Args:
        include_control: Whether control/healthy group is included

    Returns:
        dict: Positions for each group and region
    """
    if include_control:
        return {
            "cortex": {"healthy": 1, "disease": 1.5},
            "medulla": {"healthy": 2.5, "disease": 3},
        }
    else:
        return {
            "cortex": {"disease": 1},
            "medulla": {"disease": 2.5},
        }


def get_colors(include_control=True):
    """
    Define colors for each group

    Args:
        include_control: Whether control/healthy group is included

    Returns:
        dict: Colors for each group
    """
    colors = {
        "vasc": "#FF7F50",  # Coral/light red-orange
        "rpgn": "#007fbf",  # Sea green/petrol blue
    }

    if include_control:
        colors["healthy"] = "green"

    return colors


def plot_groups_with_significance(
    ax, plot_df, parameter, positions, colors, include_control=True
):
    """
    Plot scatter and violin plots for all groups and add significance indicators.

    Args:
        ax: The matplotlib axis to plot on
        plot_df: DataFrame with the data prepared for plotting
        parameter: Parameter to plot ("adc", "fa", "asl", or "t2star")
        positions: Dictionary with position information for each group
        colors: Dictionary with color information for each group
        include_control: Whether to include the control group

    Returns:
        A dictionary containing data used for significance testing
    """
    # Store data for significance testing
    significance_data = {}

    # Plot each group with horizontal jitter
    for roi in ["cortex", "medulla"]:
        if include_control:
            # Plot the healthy group - scatter on left, violin on right
            healthy_data = plot_df[
                (plot_df["roi"] == roi) & (plot_df["display_group"] == "healthy")
            ]
            if len(healthy_data) > 0:
                # Create scatter plot for healthy group
                create_scatter_plot(
                    ax=ax,
                    data=healthy_data[parameter],
                    x_position=positions[roi]["healthy"],
                    color=colors["healthy"],
                    label="healthy" if roi == "cortex" else None,
                    jitter_range=0.05,
                    x_offset=-0.05,
                )

                # Create violin plot for healthy group
                create_violin_plot(
                    ax=ax,
                    data=healthy_data[parameter],
                    x_position=positions[roi]["healthy"],
                    color=colors["healthy"],
                    alpha=0.3,
                    scaling=0.3,
                    x_offset=0.05,
                )

                # Store healthy data for significance testing
                significance_data[roi] = {"healthy": healthy_data[parameter].values}

        # Get disease data for this region
        if include_control:
            # Plot disease groups - scatter on left, violin on right
            combined_disease_data = plot_df[
                (plot_df["roi"] == roi) & (plot_df["position_group"] == "disease")
            ]

            # Store combined disease data for significance testing
            if len(combined_disease_data) > 0:
                significance_data[roi]["disease"] = combined_disease_data[
                    parameter
                ].values
        else:
            # Get data for all disease groups in this region
            combined_disease_data = plot_df[(plot_df["roi"] == roi)]

        # Create scatter plots for individual disease groups
        for disease in ["vasc", "rpgn"]:
            disease_data = plot_df[
                (plot_df["roi"] == roi) & (plot_df["display_group"] == disease)
            ]
            if len(disease_data) > 0:
                create_scatter_plot(
                    ax=ax,
                    data=disease_data[parameter],
                    x_position=positions[roi]["disease"],
                    color=colors[disease],
                    label=disease if roi == "cortex" else None,
                    jitter_range=0.05,
                    x_offset=-0.05,
                )

                # Store disease data for possible significance testing between disease groups
                if not include_control:
                    if roi not in significance_data:
                        significance_data[roi] = {}
                    significance_data[roi][disease] = disease_data[parameter].values

        # Create violin plot for combined disease groups
        if len(combined_disease_data) > 0:
            create_violin_plot(
                ax=ax,
                data=combined_disease_data[parameter],
                x_position=positions[roi]["disease"],
                color="#6a5acd",  # Light violet/thistle color
                alpha=0.3,
                scaling=0.3,
                x_offset=0.05,
            )

    return significance_data


def add_significance_indicators_to_plot(
    ax, plot_df, parameter, positions, significance_data, include_control=True
):
    """
    Add significance indicators to the plot based on statistical tests.

    Args:
        ax: The matplotlib axis to add indicators to
        plot_df: DataFrame with the data
        parameter: Parameter being plotted
        positions: Dictionary with position information for each group
        significance_data: Dictionary with data used for significance testing
        include_control: Whether control group is included
    """
    # Add significance indicators for each region
    y_max = plot_df[parameter].max()
    y_range = plot_df[parameter].max() - plot_df[parameter].min()

    # Calculate positions for significance indicators
    for roi in ["cortex", "medulla"]:
        if include_control:
            # Test between healthy and disease
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
                        f"{roi.capitalize()} - Healthy vs. Disease ({parameter.upper()}): p={p_value:.4f} {sig_marker}"
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
        else:
            # Test between disease groups
            if (
                roi in significance_data
                and "vasc" in significance_data[roi]
                and "rpgn" in significance_data[roi]
                and len(significance_data[roi]["vasc"]) > 0
                and len(significance_data[roi]["rpgn"]) > 0
            ):
                # Perform statistical test between disease groups
                p_value, sig_marker = perform_significance_test(
                    significance_data[roi]["vasc"], significance_data[roi]["rpgn"]
                )

                # Only add indicator if there is significance
                if sig_marker != "ns":
                    # Calculate y position for the indicator (above the highest point)
                    y_pos = y_max + 0.05 * y_range

                    # Print significance test results
                    print(
                        f"{roi.capitalize()} - VASC vs. RPGN ({parameter.upper()}): p={p_value:.4f} {sig_marker}"
                    )

                    # Since we only have one x-position per region now, we'll add a small offset for the significance bar
                    x_offset = 0.1

                    # Add the significance indicator
                    add_significance_indicator(
                        ax=ax,
                        x1=positions[roi]["disease"] - x_offset,
                        x2=positions[roi]["disease"] + x_offset,
                        y=y_pos,
                        significance=sig_marker,
                        text_y_offset=0.02 * y_range,
                        line_height=0.01 * y_range,
                    )

    # Ensure y-axis extends high enough to show significance indicators
    if any(roi in significance_data for roi in ["cortex", "medulla"]):
        ax.set_ylim(top=y_max + 0.15 * y_range)


def customize_parameter_plot(
    ax, parameter_info, parameter, include_control=True, plot_title=True
):
    """
    Customize the appearance of the parameter plot.

    Args:
        ax: The matplotlib axis to customize
        parameter_info: Dictionary with parameter display information
        parameter: Parameter being plotted
        include_control: Whether control group is included
        plot_title: Whether to display the plot title
    """
    # Customize the plot
    ax.set_ylabel(parameter_info[parameter]["ylabel"])
    title_suffix = " by Region and Group"
    if plot_title:
        ax.set_title(f"{parameter_info[parameter]['title']}{title_suffix}")

    # Set x-ticks and labels
    if include_control:
        ax.set_xticks([1.25, 2.75])  # Centered between the groups
    else:
        ax.set_xticks([1, 2.5])  # Centered at the positions
    ax.set_xticklabels(["Cortex", "Medulla"])

    # Add a legend with custom ordering
    ax.legend(title="Groups")

    # Remove the top and right spines to create a cleaner look
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Optional: Make the bottom and left spines a bit thinner
    ax.spines["bottom"].set_linewidth(0.5)
    ax.spines["left"].set_linewidth(0.5)


def save_parameter_plot(fig, parameter, include_control=True):
    """
    Save the parameter plot to files.

    Args:
        fig: The matplotlib figure to save
        parameter: Parameter that was plotted
        include_control: Whether control group was included
    """
    # Create the output directory if it doesn't exist using pathlib
    output_dir = Path("out")
    output_dir.mkdir(exist_ok=True)

    # Define output file paths
    control_suffix = "_with_control" if include_control else "_no_control"
    png_path = output_dir / f"{parameter}_scatter_plot{control_suffix}.png"
    svg_path = output_dir / f"{parameter}_scatter_plot{control_suffix}.svg"

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

    print(
        f"{parameter.upper()} plot created and saved in the '{output_dir}' directory as:"
    )
    print(f"- {png_path.name} (raster format)")
    print(f"- {svg_path.name} (vector format)")


def create_parameter_scatter_plot(plot_df, parameter="adc", include_control=True):
    """
    Create scatter and violin plots for the specified parameter.

    Args:
        plot_df: DataFrame with the data prepared for plotting
        parameter: Parameter to plot ("adc", "fa", "asl", or "t2star")
        include_control: Whether to include the control group
    """
    # Parameter display information
    parameter_info = get_parameter_info()

    # Set up the figure
    fig_width = 8 if include_control else 7
    fig, ax = plt.subplots(figsize=(fig_width, 6))

    # Define positions and colors
    positions = get_positions(include_control)
    colors = get_colors(include_control)

    # Plot groups and get significance data
    significance_data = plot_groups_with_significance(
        ax, plot_df, parameter, positions, colors, include_control
    )

    # Add significance indicators to the plot
    add_significance_indicators_to_plot(
        ax, plot_df, parameter, positions, significance_data, include_control
    )

    # Customize the plot appearance
    customize_parameter_plot(ax, parameter_info, parameter, include_control)

    # Finalize layout and save the plot
    plt.tight_layout()
    save_parameter_plot(fig, parameter, include_control)


def create_all_parameters_figure(plot_df, include_control=True):
    """
    Create a figure with subplots for all parameters (ADC, FA, ASL, T2star).

    Args:
        plot_df: DataFrame with the data prepared for plotting
        include_control: Whether to include the control group
    """
    # Parameter display information
    parameter_info = get_parameter_info()
    parameters = ["adc", "fa", "asl", "t2star"]

    # Define positions and colors
    positions = get_positions(include_control)
    colors = get_colors(include_control)

    # Set up the figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()  # Flatten to easily iterate

    # Plot each parameter in its own subplot
    for idx, parameter in enumerate(parameters):
        ax = axes[idx]

        # Plot groups and get significance data
        significance_data = plot_groups_with_significance(
            ax, plot_df, parameter, positions, colors, include_control
        )

        # Add significance indicators to the plot
        add_significance_indicators_to_plot(
            ax, plot_df, parameter, positions, significance_data, include_control
        )

        # Customize the plot appearance (with smaller title and no legend for non-first subplots)
        customize_subplot_appearance(
            ax,
            parameter_info,
            parameter,
            include_control,
            show_legend=(idx == 0),
            show_title=False,
        )

    # Add an overall title
    # fig.suptitle(
    #     "Comparison of All Parameters by Region and Group", fontsize=16, y=0.98
    # )

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)  # Make room for the suptitle

    # Save the combined figure
    save_combined_figure(fig, include_control)

    return fig


def customize_subplot_appearance(
    ax,
    parameter_info,
    parameter,
    include_control=True,
    show_legend=True,
    show_title=True,
):
    """
    Customize the appearance of a subplot for a specific parameter.

    Args:
        ax: The matplotlib axis to customize
        parameter_info: Dictionary with parameter display information
        parameter: Parameter being plotted
        include_control: Whether control group is included
        show_legend: Whether to show the legend on this subplot
    """
    # Customize the plot
    ax.set_ylabel(parameter_info[parameter]["ylabel"])
    if show_title:
        ax.set_title(parameter_info[parameter]["title"])

    # Set x-ticks and labels
    if include_control:
        ax.set_xticks([1.25, 2.75])  # Centered between the groups
    else:
        ax.set_xticks([1, 2.5])  # Centered at the positions
    ax.set_xticklabels(["Cortex", "Medulla"])

    # Add a legend only on the first subplot to avoid redundancy
    if show_legend:
        ax.legend(title="Groups", loc="upper right")

    # Remove the top and right spines to create a cleaner look
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Optional: Make the bottom and left spines a bit thinner
    ax.spines["bottom"].set_linewidth(0.5)
    ax.spines["left"].set_linewidth(0.5)


def save_combined_figure(fig, include_control=True):
    """
    Save the combined parameters figure to files.

    Args:
        fig: The matplotlib figure to save
        include_control: Whether control group was included
    """
    # Create the output directory if it doesn't exist using pathlib
    output_dir = Path("out")
    output_dir.mkdir(exist_ok=True)

    # Define output file paths
    control_suffix = "_with_control" if include_control else "_no_control"
    png_path = output_dir / f"all_parameters_figure{control_suffix}.png"
    svg_path = output_dir / f"all_parameters_figure{control_suffix}.svg"

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

    print(
        f"Combined figure with all parameters created and saved in the '{output_dir}' directory as:"
    )
    print(f"- {png_path.name} (raster format)")
    print(f"- {svg_path.name} (vector format)")


# Set the matplotlib backend (can be moved to main.py)
def set_matplotlib_backend():
    """Set the matplotlib backend for interactive plotting"""
    matplotlib.use("TkAgg")  # or 'Qt5Agg' if you have PyQt5 installed
