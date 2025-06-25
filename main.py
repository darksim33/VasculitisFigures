from data_handler import load_data, calculate_side_means, prepare_data_for_plotting
from plotting import (
    create_parameter_scatter_plot,
    create_all_parameters_plot,
    set_matplotlib_backend,
)


def main():
    # Load the data
    df = load_data()

    # Calculate means of left and right sides
    mean_df = calculate_side_means(df)

    # Set the matplotlib backend
    set_matplotlib_backend()

    # Available plotting options
    parameters = ["adc", "fa", "asl", "t2star"]

    print("\nPlot options:")
    print("1. Individual parameter plots")
    print("2. Combined plot with all parameters as subplots")
    print("3. Individual parameter plots (excluding control group)")
    print("4. Combined plot with all parameters as subplots (excluding control group)")

    try:
        plot_choice = int(input("\nSelect plot type (1-4): "))
    except ValueError:
        print(
            "Invalid input. Defaulting to individual parameter plots with control group."
        )
        plot_choice = 1

    include_control = plot_choice < 3  # Options 1-2 include control, 3-4 exclude it
    plot_df = prepare_data_for_plotting(mean_df, include_control=include_control)

    # Create combined plot (options 2 and 4)
    if plot_choice == 2 or plot_choice == 4:
        # Create a combined plot with all parameters
        print(
            f"\nGenerating combined plot with all parameters {'with' if include_control else 'without'} control group..."
        )
        create_all_parameters_plot(plot_df, include_control=include_control)
    else:
        # Create individual parameter plots (options 1 and 3)
        print("\nAvailable parameters to plot:")
        for i, param in enumerate(parameters, 1):
            print(f"{i}. {param.upper()}")

        choice = input(
            "\nEnter parameter numbers to plot (comma-separated, or 'all'): "
        )

        if choice.lower() == "all":
            selected_params = parameters
        else:
            try:
                # Convert input to list of indices (1-based), then get parameter names
                indices = [int(idx.strip()) - 1 for idx in choice.split(",")]
                selected_params = [
                    parameters[i] for i in indices if 0 <= i < len(parameters)
                ]
            except (ValueError, IndexError) as e:
                print(f"Invalid input: {e}. Plotting ADC by default.")
                selected_params = ["adc"]

        print(
            f"\nPlotting parameters: {', '.join(param.upper() for param in selected_params)}"
        )

        # Create plots for selected parameters
        for param in selected_params:
            print(f"\nGenerating plot for {param.upper()}...")
            create_parameter_scatter_plot(
                plot_df, parameter=param, include_control=include_control
            )


if __name__ == "__main__":
    main()
