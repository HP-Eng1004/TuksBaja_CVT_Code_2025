# TuksBaja_CVT_Code_2025

This repository contains the CVT (Continuously Variable Transmission) Model Code produced by JL (Hannes) Pretorius in 2025 as part of his MRN (Mini Research Project) for the TuksBaja team at the University of Pretoria. The code focuses on simulating, analyzing, and optimizing CVT performance for Baja SAE vehicles, specifically designed for the new Kohler/Rehlko engine. It builds on prior work by students like Nathan Mills (2019) and Mulambu (2021).

The main file is `CVT_Model_2025.py`, which serves as the core simulation engine tailored for the Kohler/Rehlko engine. Other files support data reading, plotting, part definitions, and specialized algorithms (e.g., for Briggs & Stratton engines). The code uses Python with libraries like NumPy, Plotly, Matplotlib, and Marimo (a reactive notebook framework), allowing interactive simulations and visualizations.

## Key Features
- Interactive sliders for CVT parameters (e.g., flyweights, springs, ramp angles).
- Simulations of CVT shifting, torque transfer, force balance, and slip risk optimized for the Kohler/Rehlko engine.
- Data analysis from DAQ (Data Acquisition) files for bench and vehicle tests.
- Exhaustive search for optimal setups based on RPM goals and tolerances.
- Plots for RPM vs. speed, radial forces, errors, and more.

## File Explanations
Here's a breakdown of each file in the repository and its purpose:

- **`CVT_Model_2025.py`**: The core CVT simulation model, written for the Kohler/Rehlko engine. It defines functions like `cvt_simulation` to compute CVT behavior (e.g., engine RPM vs. vehicle speed, forces, torque, and slip) based on parameters such as flyweights, springs, and shim. Includes interactive Marimo app for parameter tuning and plotting. This is the primary file to run for simulations.

- **`CVT_Parts_2025.py`**: Defines CVT component data (e.g., flyweights, primary/secondary springs, helices, pretensions). Includes arrays for geometric coefficients, stiffnesses, and names. Used as a database imported by other files. Also generates plots for spring stiffnesses and flyweight coefficients.

- **`CVT_Plotting_2025.py`**: Contains plotting functions (e.g., `plot_torque_transfer`, `plot_radial_force`, `plot_error`) using Plotly. Visualizes torque capacity, radial forces, errors, force balance heatmaps, slip risk scatter plots, RPM surfaces, and engagement RPM bar charts. Imported by simulation files for output.

- **`CVT_Algorithm_Final_2025.py`**: An exhaustive search algorithm to find optimal CVT setups for the Kohler/Rehlko engine. Runs simulations across all parameter combinations, filters based on RPM goals/tolerances, and generates plots (e.g., shift curves, force balance, slip risk). Builds on modified algorithms from Mills (2019) and Mulambu (2021).

- **`CVT_Briggs_Algorithm_2025.py`**: Specialized algorithm for Briggs & Stratton engines. Includes `cvt_simulation_briggs` for torque-adjusted simulations and comparisons between old/new models. Features interactive sliders and Plotly visualizations for diameter fixes and shift comparisons.

- **`DAQ_CVT_Bench_Plotting_2025.py`**: Processes and plots data from bench tests using DAQ files (.bin or .txt). Includes file pickers, channel calibration, low-pass filtering, and plots (e.g., RPM vs. speed, CVT ratio over time). Uses Marimo for interactive UI.

- **`CarX_Briggs_CVT_Plotting.py`**: Similar to `DAQ_CVT_Bench_Plotting_2025.py` but tailored for Car X tests (e.g., Briggs engine data from September 2025 onward). Handles channel info for front wheels, pressures, brakes, and generates plots like CVT ratio and speed comparisons.

- **`readV3.py`**: A utility script to read .bin DAQ files (provided by VDG). Parses binary data into structured channels (e.g., time, RPM). Used by plotting scripts like `DAQ_CVT_Bench_Plotting_2025.py`.

## Installation
1. Install Python 3.8+ (recommended via Anaconda for scientific libraries).
2. Clone the repo: `git clone https://github.com/yourusername/TuksBaja_CVT_Code_2025.git`
3. Install dependencies: `pip install numpy scipy matplotlib plotly marimo`
4. For Marimo apps: `pip install marimo` (run files with `marimo run filename.py`).

## Usage
- Run the main simulation: `marimo run CVT_Model_2025.py` (opens an interactive notebook in your browser for Kohler/Rehlko engine simulations).
- For bench data: Select a .bin/.txt file in `DAQ_CVT_Bench_Plotting_2025.py` via the Marimo UI.
- Customize parameters in sliders (e.g., flyweights, RPM goals) and view plots.
- For exhaustive searches: Run `CVT_Algorithm_Final_2025.py` to generate setup recommendations for the Kohler/Rehlko engine.

## Contributing
Contributions are welcome! Fork the repo, make changes, and submit a pull request. For issues, open a GitHub issue.

## License
MIT License. See LICENSE file for details.

## Acknowledgments
- Based on prior TuksBaja work by Nathan Mills (2019) and Mulambu (2021).
- Thanks to the TuksBaja team and supervisors.

