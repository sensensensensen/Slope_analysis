# Slope Analysis GUI
This is used to view CLOUD processed data (especially mass spectra) and perform slope analysis, based on python.

**Version:** 4.3 (Final)

A comprehensive Python-based graphical user interface (GUI) designed for loading, visualizing, and analyzing tracer data. This tool specializes in dynamic slope analysis using log-log linear regression, making it ideal for atmospheric science, chemistry, and time-series signal analysis.

Built with **PyQt5**, **pandas**, and **matplotlib**.

## 🚀 Features

### 1. Robust Data Loading
* **Multi-file Support:** Load up to 10 tracer files simultaneously (`.csv`, `.txt`, `.dat`).
* **Smart Parsing:** Automatically detects headers, delimiters, and time formats.
* **Composition Data:** Link auxiliary composition files (Mass/SumFormula) to your tracer data.

### 2. Advanced Preprocessing
* **Time Management:** Set global analysis time ranges and time resolution (resampling).
* **Background Subtraction:** Define background time windows to automatically calculate and subtract background noise.
* **Unit Conversion:** Apply unit factors (e.g., scaling by 1e5) dynamically.

### 3. Visualization Tools
* **Tracer Plots:** Visualize time series data for multiple variables across multiple files.
* **Slope Plots:** Scatter plots showing the relationship between Slope and Formulas, color-coded by R² (coefficient of determination).
* **Interactive Viewing:** Zoom, pan, and save plots using the integrated Matplotlib toolbar.

### 4. Analytical Capabilities
* **Slope Analysis:** Performs Log10-Log10 linear regression to calculate Slopes and R².
* **Flexible Correlation:** Calculate slopes against:
    * Time ($t$, $t^2$, $t^3$, $t^4$)
    * Other loaded variables (X-axis vs Y-axis correlation).
* **Global Analysis:** Compare variables across different files in a unified "Global Slope Table".

### 5. Data Export
* **Formats:** Export results and processed data to **Excel (.xlsx)**, **CSV**, or **MATLAB (.mat)** formats.
* **Data Viewer:** Inspect merged and processed data tables before exporting.

---

## 🛠️ Installation & Requirements

Ensure you have Python 3.8+ installed.

### 1. Clone the repository
```bash
git clone [https://github.com/yourusername/Slope_Analysis_GUI.git](https://github.com/yourusername/Slope_Analysis_GUI.git)
cd Slope_Analysis_GUI
