import pandas as pd
from pathlib import Path


def load_data():
    """
    Load and preprocess the vasculitis data from CSV.

    Returns:
        DataFrame: Preprocessed data
    """
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
    """
    Calculate mean values for both sides of each parameter.

    Args:
        df: DataFrame with the raw data

    Returns:
        DataFrame: Processed data with mean values
    """
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


def prepare_data_for_plotting(mean_df, include_control=True):
    """
    Prepare data for plotting by adding display and position groups.

    Args:
        mean_df: DataFrame with the mean values
        include_control: Whether to include the control group in the plot

    Returns:
        DataFrame: Data ready for plotting
    """
    # Create a copy of the dataframe first to avoid SettingWithCopyWarning
    plot_df = mean_df.copy()

    # Filter data if needed (removing control group)
    plot_df = plot_df[plot_df["group"] != "chronical"]

    # Keep original groups but rename 'control' to 'healthy' for clarity
    plot_df.loc[:, "display_group"] = plot_df["group"].apply(
        lambda x: "healthy" if x == "control" else x
    )

    # Create a combined disease category while preserving original groups for coloring
    plot_df.loc[:, "position_group"] = plot_df["group"].apply(
        lambda x: "healthy" if x in ("control", "healthy") else "disease"
    )

    # Filter to include only the groups we want
    plot_df = plot_df[plot_df["display_group"].isin(["healthy", "vasc", "rpgn"])]

    return plot_df
