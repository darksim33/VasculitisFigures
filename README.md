# Vasculitis Figures

A tool for visualizing MRI parameters (ADC, FA, ASL, T2star) from vasculitis patients.

## Features

- Visualize ADC, FA, ASL, and T2star parameters for different regions (cortex, medulla)
- Create individual scatter and half-violin plots for each parameter
- Create combined subplot figures with all parameters
- Compare different patient groups (vasc, rpgn) with optional control group
- Statistical significance testing with Mann-Whitney U test
- Output high-quality PNG and SVG figures

## Project Structure

- `main.py` - Entry point and CLI interface
- `data_handler.py` - Data loading and processing functions
- `plotting.py` - Visualization functions
- `data/` - Directory containing the raw data
- `out/` - Output directory for generated figures

## Usage

Run the main script:

```bash
python main.py
```

Follow the prompts to select:
1. Individual parameter plots
2. Combined plot with all parameters as subplots
3. Individual parameter plots (excluding control group)
4. Combined plot with all parameters as subplots (excluding control group)

For individual plots, you can choose specific parameters to visualize.

## Dependencies

- pandas
- matplotlib
- numpy
- scipy

## Installation

```bash
# Install dependencies using pip
pip install -r requirements.txt

# Or using Python's newer dependency management tools
pip install .
```
