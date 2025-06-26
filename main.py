import sys
from data_handler import load_data, calculate_side_means, prepare_data_for_plotting
from plotting import (
    create_parameter_scatter_plot,
    set_matplotlib_backend,
    create_all_parameters_figure,
)


def main(args=None):
    # Load the data
    df = load_data()

    # Calculate means of left and right sides
    mean_df = calculate_side_means(df)

    # Set the matplotlib backend
    set_matplotlib_backend()

    # Available plotting options
    parameters = ["adc", "fa", "asl", "t2star"]

    # Prepare data for plotting (always include control group)
    plot_df = prepare_data_for_plotting(mean_df)

    # Check if arguments were passed from command line
    if args:
        choice = args[0]
    else:
        # Create individual parameter plots
        print("\nAvailable parameters to plot:")
        for i, param in enumerate(parameters, 1):
            print(f"{i}. {param.upper()}")

        choice = input(
            "\nEnter parameter numbers to plot (comma-separated, or 'all', or 'combined'): "
        )

    if choice.lower() == "all":
        selected_params = parameters
    elif choice.lower() == "combined":
        create_all_parameters_figure(plot_df)
        return
    else:
        try:
            # Try to interpret as comma-separated indices
            if all(c.isdigit() or c == "," or c.isspace() for c in choice):
                indices = [int(idx.strip()) - 1 for idx in choice.split(",")]
                selected_params = [
                    parameters[i] for i in indices if 0 <= i < len(parameters)
                ]
            # Try to interpret as comma-separated parameter names
            else:
                param_list = [p.strip().lower() for p in choice.split(",")]
                selected_params = [p for p in param_list if p in parameters]
                if not selected_params:
                    print(
                        f"Invalid parameter names. Available: {', '.join(parameters)}"
                    )
                    selected_params = ["adc"]
        except (ValueError, IndexError) as e:
            print(f"Invalid input: {e}. Plotting ADC by default.")
            selected_params = ["adc"]

    print(
        f"\nPlotting parameters: {', '.join(param.upper() for param in selected_params)}"
    )

    # Create plots for selected parameters
    for param in selected_params:
        print(f"\nGenerating plot for {param.upper()}...")
        create_parameter_scatter_plot(plot_df, parameter=param)


if __name__ == "__main__":
    # If arguments are passed from command line, use them
    if len(sys.argv) > 1:
        main(sys.argv[1:])
    else:
        main()
