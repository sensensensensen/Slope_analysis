# -*- coding: utf-8 -*-
"""
Slope Analysis GUI
Version: 4.3 (Final, All Features Implemented)

A graphical user interface for loading, visualizing, and analyzing tracer data.
This tool allows for dynamic slope analysis based on user-defined parameters.

Required Libraries:
- PyQt5
- pandas
- numpy
- matplotlib
- scipy
- openpyxl
"""

import sys
import os
import pandas as pd
import numpy as np
from scipy import stats
from scipy.io import savemat
import matplotlib

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QFileDialog, QVBoxLayout,
    QHBoxLayout, QGridLayout, QLabel, QPushButton, QCheckBox, QTableWidget,
    QTableWidgetItem, QLineEdit, QAction, QMessageBox, QScrollArea,
    QDialog, QDialogButtonBox, QHeaderView, QSizePolicy, QFrame
)
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont, QDoubleValidator

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import matplotlib.cm as cm


# =============================================================================
#  UTILITY/HELPER FUNCTIONS
# =============================================================================

def detect_header_and_delimiter_for_tracer(file_path, lines_to_scan=50):
    """
    Robustly detects the header row index and delimiter for a tracer data file.
    """
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = [f.readline() for _ in range(lines_to_scan)]
    except Exception as e:
        print(f"Error reading file for header detection: {e}")
        return 0, '\t'

    potential_delimiters = ['\t', ',', ';']

    for i, line in enumerate(lines):
        line = line.strip()
        if not line:
            continue

        for delim in potential_delimiters:
            parts = line.split(delim)
            if len(parts) > 2:
                try:
                    pd.to_datetime(parts[0], dayfirst=False, yearfirst=False)
                    numeric_count = sum(
                        1 for part in parts[1:] if part.strip().replace('.', '', 1).replace('-', '', 1).isdigit())
                    if numeric_count >= len(parts) / 2:
                        header_row_index = max(0, i - 1)
                        for next_line in lines[i + 1:]:
                            if next_line.strip():
                                next_parts = next_line.strip().split(delim)
                                if abs(len(next_parts) - len(parts)) <= 2:
                                    return header_row_index, delim
                                break
                except (ValueError, TypeError, IndexError):
                    continue

    print("Warning: Advanced header detection failed. Falling back to default.")
    return 0, ','


def detect_composition_header(file_path):
    """Detects the header row for a composition file."""
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for i, line in enumerate(f):
                if i > 30: break
                upper_line = line.strip().upper()
                if "MASS" in upper_line and "SUMFORMULA" in upper_line:
                    return i
    except Exception as e:
        print(f"Error reading for composition header: {e}")
    return 0


def calculate_slope_r2(x_series, y_series):
    """
    Performs log10-log10 linear regression using the user-specified
    multi-step filtering and alignment logic.
    """
    df = pd.DataFrame({'x': x_series, 'y': y_series}).dropna()

    if len(df) < 2:
        return np.nan, np.nan, len(df)

    x_log = np.log10(df['x'][df['x'] > 0])
    y_log = np.log10(df['y'][df['y'] > 0])

    common_index = x_log.index.intersection(y_log.index)
    x_log = x_log.loc[common_index]
    y_log = y_log.loc[common_index]

    n_points = len(x_log)
    if n_points < 2:
        return np.nan, np.nan, n_points

    if np.var(x_log) < 1e-10:
        if np.array_equal(x_log.values, y_log.values):
            return 1.0, 1.0, n_points
        return np.nan, np.nan, n_points

    try:
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_log, y_log)
        r_squared = r_value ** 2
        return slope, r_squared, n_points
    except Exception as e:
        print(f"Linear regression failed: {e}")
        return np.nan, np.nan, n_points


# =============================================================================
#  REUSABLE DIALOG WINDOWS
# =============================================================================

class PlottingWindow(QDialog):
    """A reusable dialog window to display a Matplotlib figure with a toolbar."""

    def __init__(self, figure, title="Plot Window", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumSize(800, 600)

        self.setWindowFlags(
            Qt.Window |
            Qt.WindowMinimizeButtonHint |
            Qt.WindowMaximizeButtonHint |
            Qt.WindowCloseButtonHint
        )

        layout = QVBoxLayout(self)

        canvas = FigureCanvas(figure)
        toolbar = NavigationToolbar(canvas, self)

        layout.addWidget(toolbar)
        layout.addWidget(canvas)

        self.setLayout(layout)


class SlopeTableWindow(QDialog):
    """A reusable dialog window to display slope results in a table with a save button."""

    def __init__(self, df, title="Slope Analysis Results", parent=None):
        super().__init__(parent)
        self.df_results = df
        self.setWindowTitle(title)
        self.setMinimumSize(600, 400)

        self.setWindowFlags(
            Qt.Window |
            Qt.WindowMinimizeButtonHint |
            Qt.WindowMaximizeButtonHint |
            Qt.WindowCloseButtonHint
        )

        layout = QVBoxLayout(self)

        self.table = QTableWidget()
        self.populate_table()

        button_box = QDialogButtonBox()
        save_button = button_box.addButton("Save As...", QDialogButtonBox.ActionRole)
        close_button = button_box.addButton(QDialogButtonBox.Close)

        save_button.clicked.connect(self._on_save_data)
        close_button.clicked.connect(self.accept)

        layout.addWidget(self.table)
        layout.addWidget(button_box)
        self.setLayout(layout)

    def populate_table(self):
        df_display = self.df_results.copy()
        for col in ['Slope', 'R2']:
            if col in df_display.columns:
                df_display[col] = df_display[col].apply(
                    lambda x: f'{x:.4f}' if isinstance(x, (float, np.floating)) else x)

        df = df_display.reset_index()
        self.table.setRowCount(df.shape[0])
        self.table.setColumnCount(df.shape[1])
        self.table.setHorizontalHeaderLabels(df.columns)

        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                self.table.setItem(i, j, QTableWidgetItem(str(df.iat[i, j])))

        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.table.resizeColumnsToContents()

    def _on_save_data(self):
        filters = "Excel Workbook (*.xlsx);;CSV (Comma delimited) (*.csv);;MATLAB MAT-file (*.mat)"
        path, _ = QFileDialog.getSaveFileName(self, "Save Slope Table", "", filters)

        if not path:
            return

        try:
            if path.endswith('.xlsx'):
                self.df_results.to_excel(path, index=True)
            elif path.endswith('.csv'):
                self.df_results.to_csv(path, index=True)
            elif path.endswith('.mat'):
                mat_dict = {col.replace(' ', '_'): self.df_results[col].values for col in self.df_results.columns}
                if isinstance(self.df_results.index, pd.MultiIndex):
                    for i, level_name in enumerate(self.df_results.index.names):
                        mat_dict[level_name] = self.df_results.index.get_level_values(i)
                else:
                    mat_dict[self.df_results.index.name] = self.df_results.index.values
                savemat(path, mat_dict)
            else:
                QMessageBox.warning(self, "Unsupported Format",
                                    "File extension not recognized. Please choose a valid format.")
                return

            QMessageBox.information(self, "Success", f"Data successfully saved to:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save file:\n{str(e)}")


class DataViewerWindow(QDialog):
    """A dedicated window for viewing and saving merged data."""

    def __init__(self, df, title="Data Viewer", parent=None):
        super().__init__(parent)
        self.df_results = df
        self.setWindowTitle(title)
        self.setMinimumSize(800, 600)

        self.setWindowFlags(
            Qt.Window |
            Qt.WindowMinimizeButtonHint |
            Qt.WindowMaximizeButtonHint |
            Qt.WindowCloseButtonHint
        )

        layout = QVBoxLayout(self)

        self.table = QTableWidget()
        self.populate_table()

        button_box = QDialogButtonBox()
        save_button = button_box.addButton("Save As...", QDialogButtonBox.ActionRole)
        close_button = button_box.addButton(QDialogButtonBox.Close)

        save_button.clicked.connect(self._on_save_data)
        close_button.clicked.connect(self.accept)

        layout.addWidget(self.table)
        layout.addWidget(button_box)
        self.setLayout(layout)

    def populate_table(self):
        df_display = self.df_results.copy()

        for col in df_display.columns:
            if pd.api.types.is_numeric_dtype(df_display[col]):
                df_display[col] = df_display[col].apply(lambda x: f'{x:.3e}' if not pd.isna(x) else "")

        df = df_display
        self.table.setRowCount(df.shape[0])
        self.table.setColumnCount(df.shape[1])
        self.table.setHorizontalHeaderLabels(df.columns)

        for i in range(df.shape[0]):
            for j in range(df.shape[1]):
                self.table.setItem(i, j, QTableWidgetItem(str(df.iat[i, j])))

        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.Interactive)
        self.table.resizeColumnsToContents()

    def _on_save_data(self):
        filters = "Excel Workbook (*.xlsx);;CSV (Comma delimited) (*.csv);;MATLAB MAT-file (*.mat)"
        path, _ = QFileDialog.getSaveFileName(self, "Save Data", "", filters)

        if not path:
            return

        try:
            if path.endswith('.xlsx'):
                self.df_results.to_excel(path, index=False)
            elif path.endswith('.csv'):
                self.df_results.to_csv(path, index=False)
            elif path.endswith('.mat'):
                mat_dict = {col.replace(' ', '_').replace('-', '_'): self.df_results[col].values for col in
                            self.df_results.columns}
                savemat(path, mat_dict)
            else:
                QMessageBox.warning(self, "Unsupported Format",
                                    "File extension not recognized. Please choose a valid format.")
                return

            QMessageBox.information(self, "Success", f"Data successfully saved to:\n{path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save file:\n{str(e)}")


class TimeTransformDialog(QDialog):
    """Dialog to choose the time transformation for slope analysis."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Choose X-axis Time Transformation")
        self.selection = None

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Select the function of time (t) for the x-axis:"))

        buttons = {
            "t (Linear Time)": "t",
            "t² (Quadratic)": "t2",
            "t³ (Cubic)": "t3",
            "t⁴ (Quartic)": "t4"
        }

        for text, key in buttons.items():
            btn = QPushButton(text)
            btn.clicked.connect(lambda _, k=key: self.set_selection(k))
            layout.addWidget(btn)

        self.setLayout(layout)

    def set_selection(self, key):
        self.selection = key
        self.accept()


# =============================================================================
#  MINIMIZED WIDGET (for docking)
# =============================================================================
class MinimizedWidget(QFrame):
    """A small placeholder widget for a minimized TracerFileWidget."""
    restore_requested = pyqtSignal()

    def __init__(self, file_name, parent=None):
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        self.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.setFixedWidth(150)

        layout = QVBoxLayout(self)

        file_label = QLabel(file_name)
        file_label.setWordWrap(True)

        restore_btn = QPushButton("Restore")
        restore_btn.clicked.connect(self.restore_requested.emit)

        layout.addWidget(file_label, alignment=Qt.AlignCenter)
        layout.addStretch()
        layout.addWidget(restore_btn)
        self.setLayout(layout)


# =============================================================================
#  TRACER FILE WIDGET (The core component for each loaded file)
# =============================================================================

class TracerFileWidget(QWidget):
    """A self-contained widget for analyzing one loaded tracer file."""
    widget_deleted = pyqtSignal(int)
    minimize_requested = pyqtSignal(int)

    def __init__(self, file_path, data_dict, index, parent=None):
        super().__init__(parent)
        self.file_path = file_path
        self.file_name = os.path.basename(file_path)
        self.data_dict = data_dict
        self.index = index
        self.main_window = parent
        self.results_df_list = []
        self.is_minimized = False

        self._init_ui()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        self.setMinimumWidth(450)

        self.container_frame = QFrame()
        self.container_frame.setStyleSheet("QFrame { border: 1px solid #BBBBBB; border-radius: 5px; }")
        container_layout = QVBoxLayout(self.container_frame)

        header_frame = QWidget()
        header_layout = QVBoxLayout(header_frame)
        header_layout.setContentsMargins(5, 5, 5, 2)

        top_bar_layout = QHBoxLayout()
        title_label = QLabel(f"<b>{self.file_name}</b>")
        title_label.setToolTip(self.file_path)
        self.time_range_label = QLabel("Loading time range...")
        self.time_range_label.setStyleSheet("color: gray;")

        title_vbox = QVBoxLayout()
        title_vbox.addWidget(title_label)
        title_vbox.addWidget(self.time_range_label)

        minimize_button = QPushButton("_")
        minimize_button.setFixedSize(24, 24)
        minimize_button.setToolTip("Minimize this window")
        minimize_button.clicked.connect(self._on_minimize_widget)

        self.load_comp_btn = QPushButton("Load Comp")
        self.load_comp_btn.setToolTip("Load a composition file for this tracer")
        self.load_comp_btn.clicked.connect(self._on_load_composition)

        delete_button = QPushButton("X")
        delete_button.setFixedSize(24, 24)
        delete_button.setToolTip("Remove this file from the analysis")
        delete_button.setStyleSheet(
            "QPushButton { color: red; font-weight: bold; border: 1px solid red; border-radius: 12px; }")
        delete_button.clicked.connect(self._on_delete_widget)

        top_bar_layout.addLayout(title_vbox)
        top_bar_layout.addStretch()
        top_bar_layout.addWidget(minimize_button)
        top_bar_layout.addWidget(self.load_comp_btn)
        top_bar_layout.addWidget(delete_button)
        header_layout.addLayout(top_bar_layout)

        container_layout.addWidget(header_frame)

        factor_widget = QWidget()
        factor_layout = QHBoxLayout(factor_widget)
        factor_layout.setContentsMargins(5, 0, 5, 5)
        factor_layout.addWidget(QLabel("Unit Factor:"))
        self.factor_input = QLineEdit("1.0")
        validator = QDoubleValidator()
        validator.setNotation(QDoubleValidator.ScientificNotation)
        self.factor_input.setValidator(validator)

        apply_factor_btn = QPushButton("Apply Factor")
        apply_factor_btn.clicked.connect(self._on_apply_factor)
        factor_layout.addWidget(self.factor_input)
        factor_layout.addWidget(apply_factor_btn)

        self.table = QTableWidget()
        self.table.setColumnCount(6)
        self.table.setHorizontalHeaderLabels(["X-axis", "Selected", "Plotted", "Formulas", "Composition", "Background"])

        header = self.table.horizontalHeader()
        for i in range(6):
            header.setSectionResizeMode(i, QHeaderView.Interactive)

        self.table.setColumnWidth(0, 50)
        self.table.setColumnWidth(1, 60)
        self.table.setColumnWidth(2, 50)
        self.table.setColumnWidth(3, 150)
        self.table.setColumnWidth(4, 150)
        self.table.setColumnWidth(5, 120)
        self.populate_table()

        button_grid = QGridLayout()
        self.tracer_plot_btn = QPushButton("TracerPlot")
        self.slope_table_btn = QPushButton("SlopeTable")
        self.slope_plot_btn = QPushButton("SlopePlot")
        self.slope_plot_btn.setEnabled(False)

        button_grid.addWidget(self.tracer_plot_btn, 0, 0)
        button_grid.addWidget(self.slope_table_btn, 0, 1)
        button_grid.addWidget(self.slope_plot_btn, 0, 2)

        search_layout = QHBoxLayout()
        self.search_input = QLineEdit()
        self.search_input.setPlaceholderText("Search Formula...")
        search_btn = QPushButton("Locate")
        search_layout.addWidget(self.search_input)
        search_layout.addWidget(search_btn)

        batch_select_layout = QHBoxLayout()
        select_all_btn = QPushButton("Select All")
        deselect_all_btn = QPushButton("Deselect All")
        de_xaxis_all_btn = QPushButton("De-select X-axis All")
        de_plotted_all_btn = QPushButton("De-select Plotted All")
        batch_select_layout.addWidget(select_all_btn)
        batch_select_layout.addWidget(deselect_all_btn)
        batch_select_layout.addWidget(de_xaxis_all_btn)
        batch_select_layout.addWidget(de_plotted_all_btn)

        container_layout.addWidget(factor_widget)
        container_layout.addWidget(self.table)
        container_layout.addLayout(button_grid)
        container_layout.addLayout(search_layout)
        container_layout.addLayout(batch_select_layout)
        main_layout.addWidget(self.container_frame)

        self.tracer_plot_btn.clicked.connect(self._on_tracer_plot)
        self.slope_table_btn.clicked.connect(self._on_slope_table)
        self.slope_plot_btn.clicked.connect(self._on_slope_plot)
        search_btn.clicked.connect(self._on_search)
        self.search_input.returnPressed.connect(self._on_search)
        select_all_btn.clicked.connect(lambda: self._toggle_all_checkboxes('Selected', True))
        deselect_all_btn.clicked.connect(lambda: self._toggle_all_checkboxes('Selected', False))
        de_xaxis_all_btn.clicked.connect(lambda: self._toggle_all_checkboxes('X-axis', False))
        de_plotted_all_btn.clicked.connect(lambda: self._toggle_all_checkboxes('Plotted', False))

    def set_full_time_range_text(self, start_time, end_time):
        start_str = start_time.strftime('%Y-%m-%d %H:%M')
        end_str = end_time.strftime('%Y-%m-%d %H:%M')
        self.time_range_label.setText(f"{start_str} to {end_str}")

    def _on_minimize_widget(self):
        self.minimize_requested.emit(self.index)

    def populate_table(self):
        df = self.data_dict.get('original_df')
        if df is None: return

        formulas = df.columns
        self.table.setRowCount(len(formulas))

        for i, formula in enumerate(formulas):
            for j in range(3):
                cell_widget = QWidget()
                chk_box = QCheckBox()
                layout = QHBoxLayout(cell_widget)
                layout.addWidget(chk_box)
                layout.setAlignment(Qt.AlignCenter)
                layout.setContentsMargins(0, 0, 0, 0)
                self.table.setCellWidget(i, j, cell_widget)

            self.table.setItem(i, 3, QTableWidgetItem(str(formula)))
            self.table.setItem(i, 4, QTableWidgetItem(""))
            self.table.setItem(i, 5, QTableWidgetItem(""))

        self.update_composition_column()
        self.update_background_column()

    def update_composition_column(self):
        comp_df = self.data_dict.get('composition_df')
        if comp_df is None or comp_df.empty:
            for i in range(self.table.rowCount()):
                self.table.item(i, 4).setText("")
            return

        has_mass = 'Mass' in comp_df.columns
        has_formula = 'SumFormula' in comp_df.columns

        for i in range(self.table.rowCount()):
            if i < len(comp_df):
                parts = []
                row_data = comp_df.iloc[i]
                if has_mass: parts.append(str(row_data.get('Mass', 'N/A')))
                if has_formula: parts.append(str(row_data.get('SumFormula', 'N/A')))
                self.table.item(i, 4).setText(" | ".join(parts))

    def update_background_column(self):
        bg_values = self.data_dict.get('background_values')
        was_nan = self.data_dict.get('background_was_nan')

        formulas = [self.table.item(i, 3).text() for i in range(self.table.rowCount())]

        for i, formula in enumerate(formulas):
            if bg_values is not None and formula in bg_values.index:
                value = bg_values[formula]
                text = f"{value:.3e}"
                if was_nan is not None and formula in was_nan.index and was_nan[formula]:
                    text = "0 (Actual is NAN)"
                self.table.item(i, 5).setText(text)
            else:
                self.table.item(i, 5).setText("")

    def _get_checked_formulas(self, column_name):
        col_map = {"X-axis": 0, "Selected": 1, "Plotted": 2}
        col_idx = col_map.get(column_name)
        if col_idx is None: return []

        checked_formulas = []
        for i in range(self.table.rowCount()):
            cell_widget = self.table.cellWidget(i, col_idx)
            chk_box = cell_widget.findChild(QCheckBox)
            if chk_box and chk_box.isChecked():
                formula = self.table.item(i, 3).text()
                checked_formulas.append(formula)
        return checked_formulas

    def _on_tracer_plot(self):
        plotted_formulas = self._get_checked_formulas("Plotted")
        if not plotted_formulas:
            QMessageBox.warning(self, "Selection Error",
                                "Please select at least one variable in the 'Plotted' column to plot.")
            return

        df = self.data_dict.get('processed_df')
        if df is None or df.empty:
            QMessageBox.critical(self, "Data Error", "Processed data is not available for plotting.")
            return

        fig = Figure(figsize=(10, 7), dpi=100)
        ax = fig.add_subplot(111)

        for formula in plotted_formulas:
            if formula in df.columns:
                color = self.main_window.get_color_for_formula(formula)
                ax.plot(df.index, df[formula], label=formula, color=color)

        ax.set_yscale('log')

        axis_label_fontsize = 14
        tick_label_fontsize = 12
        legend_fontsize = 10

        ax.set_title(f"Tracer Time Series for {self.file_name}")
        ax.set_xlabel("Time", fontsize=axis_label_fontsize)
        ax.set_ylabel("Value (Factored, BG Subtracted)", fontsize=axis_label_fontsize)
        ax.tick_params(axis='both', which='major', labelsize=tick_label_fontsize)
        ax.legend(fontsize=legend_fontsize)
        ax.grid(True, which='both', linestyle='--', alpha=0.6)
        fig.tight_layout()

        plot_window = PlottingWindow(fig, title=f"Tracer Plot - {self.file_name}", parent=self)
        plot_window.exec_()

    def _on_slope_table(self):
        self.results_df_list = []

        selected_y = self._get_checked_formulas("Selected")
        if not selected_y:
            QMessageBox.warning(self, "Selection Error", "Please select variables in the 'Selected' column.")
            return

        df = self.data_dict.get('processed_df')
        if df is None or df.empty:
            QMessageBox.critical(self, "Data Error", "Processed data is not available for analysis.")
            return

        selected_x = self._get_checked_formulas("X-axis")

        if not selected_x:
            dialog = TimeTransformDialog(self)
            if dialog.exec_() == QDialog.Accepted and dialog.selection:
                time_transform = dialog.selection
                time_index_seconds = (df.index - df.index[0]).total_seconds()
                x_series_data = pd.Series(time_index_seconds, index=df.index, name=time_transform)
                if time_transform != 't':
                    power = int(time_transform[1])
                    x_series_data = x_series_data ** power

                results = {y: calculate_slope_r2(x_series_data, df[y]) for y in selected_y}
                results_df = pd.DataFrame.from_dict(results, orient='index', columns=['Slope', 'R2', 'Data Number'])
                results_df.index.name = "Formulas"

                self.results_df_list.append({'df': results_df, 'title': f"vs {time_transform}"})
                table_window = SlopeTableWindow(results_df,
                                                title=f"Slope Results (vs {time_transform}) - {self.file_name}",
                                                parent=self)
                table_window.exec_()
        else:
            for x_var in selected_x:
                results = {y: calculate_slope_r2(df[x_var], df[y]) for y in selected_y}
                results_df = pd.DataFrame.from_dict(results, orient='index', columns=['Slope', 'R2', 'Data Number'])
                results_df.index.name = "Formulas"

                self.results_df_list.append({'df': results_df, 'title': f"vs {x_var}"})
                table_window = SlopeTableWindow(results_df, title=f"Slope Results (vs {x_var}) - {self.file_name}",
                                                parent=self)
                table_window.exec_()

        if self.results_df_list:
            self.slope_plot_btn.setEnabled(True)

    def _on_slope_plot(self):
        if not hasattr(self, 'results_df_list') or not self.results_df_list:
            QMessageBox.warning(self, "No Data", "Please generate a SlopeTable first before plotting.")
            return

        for result_item in self.results_df_list:
            df = result_item['df'].dropna(subset=['Slope', 'R2'])
            title_suffix = result_item['title']

            if df.empty:
                QMessageBox.information(self, "No Data", f"No valid data points to plot for analysis '{title_suffix}'.")
                continue

            fig = Figure(figsize=(10, 7), dpi=100)
            ax = fig.add_subplot(111)

            df_low_r2 = df[df['R2'] < 0.6]
            df_high_r2 = df[df['R2'] >= 0.6]

            if not df_low_r2.empty: ax.scatter(df_low_r2.index, df_low_r2['Slope'], c='lightgray', label='R² < 0.6',
                                               s=50, zorder=2)
            if not df_high_r2.empty:
                colors = df_high_r2['R2']
                cmap = matplotlib.colormaps.get_cmap('viridis')
                sc = ax.scatter(df_high_r2.index, df_high_r2['Slope'], c=colors, cmap=cmap, vmin=0.6, vmax=1.0,
                                label='R² >= 0.6', s=50, zorder=3)
                fig.colorbar(sc, ax=ax, label='R² Value')

            for i, row in df.iterrows():
                ax.text(i, row['Slope'], f"  {int(row['Data Number'])}", fontsize=8, verticalalignment='bottom')

            axis_label_fontsize = 14
            tick_label_fontsize = 12
            legend_fontsize = 10

            plot_title = f"Slope Plot for {self.file_name} ({title_suffix})"
            ax.set_title(plot_title)
            ax.set_ylabel("Slope", fontsize=axis_label_fontsize)
            ax.set_xlabel("Formulas", fontsize=axis_label_fontsize)
            ax.tick_params(axis='y', which='major', labelsize=tick_label_fontsize)
            ax.tick_params(axis='x', rotation=90, labelsize=8)
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend(fontsize=legend_fontsize)
            fig.tight_layout()

            plot_window = PlottingWindow(fig, title=plot_title, parent=self)
            plot_window.exec_()

    def _on_search(self):
        search_text = self.search_input.text().strip().lower()
        if not search_text: return

        for i in range(self.table.rowCount()):
            formula = self.table.item(i, 3).text().lower()
            if search_text in formula:
                self.table.scrollToItem(self.table.item(i, 3), QTableWidget.PositionAtCenter)
                self.table.selectRow(i)
                chk_box = self.table.cellWidget(i, 2).findChild(QCheckBox)
                if chk_box: chk_box.setChecked(True)
                return

        QMessageBox.information(self, "Not Found", f"The formula '{self.search_input.text()}' was not found.")

    def _toggle_all_checkboxes(self, column_name, state):
        col_map = {"X-axis": 0, "Selected": 1, "Plotted": 2}
        col_idx = col_map.get(column_name)
        if col_idx is None: return

        for i in range(self.table.rowCount()):
            chk_box = self.table.cellWidget(i, col_idx).findChild(QCheckBox)
            if chk_box: chk_box.setChecked(state)

    def _on_apply_factor(self):
        try:
            factor = float(self.factor_input.text())
            if factor <= 0:
                QMessageBox.warning(self, "Input Error", "Unit Factor must be a positive number.")
                self.factor_input.setText(str(self.data_dict.get('unit_factor', 1.0)))
                return

            self.data_dict['unit_factor'] = factor
            self.main_window.update_all_files()
            QMessageBox.information(self, "Success", f"Factor {factor} applied. Data recalculated.")
        except (ValueError, TypeError):
            QMessageBox.warning(self, "Input Error", "Please enter a valid number (e.g., 100 or 1e5).")

    def _on_delete_widget(self):
        reply = QMessageBox.question(self, "Confirm Deletion",
                                     f"Are you sure you want to remove the file '{self.file_name}'?",
                                     QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.widget_deleted.emit(self.index)

    def _on_load_composition(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open Composition Data File", "", "Data Files (*.csv *.txt)")
        if not path: return

        try:
            skip = detect_composition_header(path)
            df = pd.read_csv(path, sep=r'\s+|,|\t', skiprows=skip, engine='python', skipinitialspace=True)
            df.columns = df.columns.str.strip()
            for col in df.columns:
                if col.lower() == 'mass': df.rename(columns={col: 'Mass'}, inplace=True)
                if col.lower() == 'sumformula': df.rename(columns={col: 'SumFormula'}, inplace=True)

            if 'Mass' not in df.columns or 'SumFormula' not in df.columns:
                raise ValueError("Key columns 'Mass' or 'SumFormula' not found.")

            self.data_dict['composition_df'] = df
            self.update_composition_column()
            QMessageBox.information(self, "Success", f"Composition data loaded for\n{self.file_name}")

        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load composition file:\n{e}")
            self.data_dict['composition_df'] = None
            self.update_composition_column()

    def update_index(self, new_index):
        self.index = new_index


# =============================================================================
#  MAIN APPLICATION WINDOW
# =============================================================================

class SlopeAnalysisGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Slope Analysis GUI")
        self.setGeometry(100, 100, 1200, 800)

        self.loaded_files = []
        self.tracer_widgets = []
        self.placeholder_widgets = {}
        self.global_results_df_list = []

        self.formula_color_map = {}
        self.color_pool = self._initialize_color_cycle(200)
        self.next_color_index = 0

        self._init_ui()

    def _initialize_color_cycle(self, num_colors_needed=50):
        qualitative_cmaps = ['tab20', 'Set3', 'tab20b', 'Set2', 'Paired', 'Accent']
        generated_colors = []
        seen_color_hashes = set()

        for cmap_name in qualitative_cmaps:
            try:
                cmap = matplotlib.colormaps.get_cmap(cmap_name)
                colors_from_cmap = cmap.colors
                for color in colors_from_cmap:
                    color_tuple_hash = tuple(round(c, 4) for c in color[:3])
                    if color_tuple_hash not in seen_color_hashes:
                        generated_colors.append(color)
                        seen_color_hashes.add(color_tuple_hash)
                        if len(generated_colors) >= num_colors_needed:
                            return generated_colors[:num_colors_needed]
            except Exception as e:
                print(f"Warning: Could not load colormap '{cmap_name}': {e}")

        if len(generated_colors) < num_colors_needed:
            cmap_cont = matplotlib.colormaps.get_cmap('hsv')
            needed_more = num_colors_needed - len(generated_colors)
            for i in np.linspace(0, 1, needed_more + 10):
                color = cmap_cont(i)
                color_tuple_hash = tuple(round(c, 4) for c in color[:3])
                if color_tuple_hash not in seen_color_hashes:
                    generated_colors.append(color)
                    seen_color_hashes.add(color_tuple_hash)
                    if len(generated_colors) >= num_colors_needed:
                        break

        return generated_colors[:num_colors_needed]

    def get_color_for_formula(self, formula_name):
        if formula_name not in self.formula_color_map:
            color = self.color_pool[self.next_color_index % len(self.color_pool)]
            self.formula_color_map[formula_name] = color
            self.next_color_index += 1
        return self.formula_color_map[formula_name]

    def _init_ui(self):
        self._create_menus()

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QVBoxLayout(main_widget)

        self._create_global_controls()
        main_layout.addWidget(self.global_controls_widget)

        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self.widgets_container = QWidget()
        self.widgets_layout = QHBoxLayout(self.widgets_container)
        self.widgets_layout.setAlignment(Qt.AlignLeft)

        self.scroll_area.setWidget(self.widgets_container)
        main_layout.addWidget(self.scroll_area)

    def _create_menus(self):
        menubar = self.menuBar()
        file_menu = menubar.addMenu("&File")

        load_tracer_action = QAction("Load Tracer Data", self)
        load_tracer_action.triggered.connect(self.load_tracer_data)
        file_menu.addAction(load_tracer_action)
        file_menu.addSeparator()

        exit_action = QAction("Exit", self)
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        data_viewer_menu = menubar.addMenu("Data Viewer")
        view_selected_action = QAction("Selected Data", self)
        view_selected_action.triggered.connect(lambda: self._show_data_viewer("Selected"))
        view_plotted_action = QAction("Plotted Data", self)
        view_plotted_action.triggered.connect(lambda: self._show_data_viewer("Plotted"))
        data_viewer_menu.addAction(view_selected_action)
        data_viewer_menu.addAction(view_plotted_action)

        self.slope_table_menu = menubar.addMenu("SlopeTable")
        global_slope_table_action = QAction("Generate Global SlopeTable", self)
        global_slope_table_action.triggered.connect(self._on_global_slope_table)
        self.slope_table_menu.addAction(global_slope_table_action)

        self.slope_plot_menu = menubar.addMenu("SlopePlot")
        global_tracer_plot_action = QAction("Global TracerPlot", self)
        global_tracer_plot_action.triggered.connect(self._on_global_tracer_plot)
        self.global_slope_plot_action = QAction("Global SlopePlot", self)
        self.global_slope_plot_action.triggered.connect(self._on_global_slope_plot)
        self.global_slope_plot_action.setEnabled(False)
        self.slope_plot_menu.addAction(global_tracer_plot_action)
        self.slope_plot_menu.addAction(self.global_slope_plot_action)

    def _create_global_controls(self):
        self.global_controls_widget = QWidget()
        layout = QGridLayout(self.global_controls_widget)

        self.restore_all_btn = QPushButton("Restore All Windows")
        self.restore_all_btn.clicked.connect(self.restore_all_windows)
        layout.addWidget(self.restore_all_btn, 0, 5)

        layout.addWidget(QLabel("<b>Time Range:</b>"), 0, 0)
        self.start_time_input = QLineEdit()
        self.start_time_input.setPlaceholderText("yyyy-mm-dd hh:mm")
        layout.addWidget(self.start_time_input, 0, 1)
        layout.addWidget(QLabel("to"), 0, 2)
        self.end_time_input = QLineEdit()
        self.end_time_input.setPlaceholderText("yyyy-mm-dd hh:mm")
        layout.addWidget(self.end_time_input, 0, 3)
        self.set_time_btn = QPushButton("Set Time Period")
        layout.addWidget(self.set_time_btn, 0, 4)

        layout.addWidget(QLabel("<b>Background Range:</b>"), 1, 0)
        self.bg_start_time_input = QLineEdit()
        self.bg_start_time_input.setPlaceholderText("yyyy-mm-dd hh:mm")
        layout.addWidget(self.bg_start_time_input, 1, 1)
        layout.addWidget(QLabel("to"), 1, 2)
        self.bg_end_time_input = QLineEdit()
        self.bg_end_time_input.setPlaceholderText("yyyy-mm-dd hh:mm")
        layout.addWidget(self.bg_end_time_input, 1, 3)
        self.subtract_bg_btn = QPushButton("Subtract BG")
        layout.addWidget(self.subtract_bg_btn, 1, 4)

        layout.addWidget(QLabel("<b>Time Resolution:</b>"), 2, 0)
        self.reso_input = QLineEdit("30s")
        self.reso_input.setPlaceholderText("e.g., 60S, 5min, 1H")
        layout.addWidget(self.reso_input, 2, 1, 1, 3)
        self.set_reso_btn = QPushButton("Set Time Resolution")
        layout.addWidget(self.set_reso_btn, 2, 4)

        self.set_time_btn.clicked.connect(self._on_set_time_period)
        self.set_reso_btn.clicked.connect(self.update_all_files)
        self.subtract_bg_btn.clicked.connect(self._on_subtract_background)

        self.global_controls_widget.setEnabled(False)

    def _on_set_time_period(self):
        start_str = self.start_time_input.text().strip()
        end_str = self.end_time_input.text().strip()

        if not start_str or not end_str:
            QMessageBox.warning(self, "Input Error", "Please enter both a start and end time.")
            return

        try:
            start_time = pd.to_datetime(start_str, format='%Y-%m-%d %H:%M')
            end_time = pd.to_datetime(end_str, format='%Y-%m-%d %H:%M')

            if start_time >= end_time:
                QMessageBox.warning(self, "Input Error", "Start time must be before end time.")
                return
        except ValueError:
            QMessageBox.warning(self, "Input Error", "Invalid date format. Please use 'yyyy-mm-dd hh:mm'.")
            return

        self.update_all_files()
        QMessageBox.information(self, "Success", "Time period has been successfully updated.")

    def load_tracer_data(self):
        if len(self.loaded_files) >= 10:
            QMessageBox.warning(self, "Limit Reached", "You can load a maximum of 10 tracer files.")
            return

        path, _ = QFileDialog.getOpenFileName(self, "Open Tracer Data File", "", "Data Files (*.csv *.txt *.dat)")
        if not path: return

        if any(f['path'] == path for f in self.loaded_files):
            QMessageBox.warning(self, "Duplicate File", "This file has already been loaded.")
            return

        try:
            skip, sep = detect_header_and_delimiter_for_tracer(path)
            df_raw = pd.read_csv(path, sep=sep, skiprows=skip, header=0, skipinitialspace=True, engine='python')

            unixtime_cols = [col for col in df_raw.columns if 'unixtime' in col.lower()]
            if unixtime_cols:
                df_raw.drop(columns=unixtime_cols, inplace=True)

            df_raw.columns = df_raw.columns.str.strip()
            time_col_name = df_raw.columns[0]

            df_raw[time_col_name] = pd.to_datetime(df_raw[time_col_name], errors='coerce', dayfirst=False,
                                                   yearfirst=False)

            df_raw.dropna(subset=[time_col_name], inplace=True)
            if df_raw.empty:
                raise ValueError("DataFrame is empty after parsing dates and cleaning index.")

            df_raw.set_index(time_col_name, inplace=True)

            for col in df_raw.columns:
                df_raw[col] = pd.to_numeric(df_raw[col], errors='coerce')

            df_raw.dropna(axis=1, how='all', inplace=True)
            df = df_raw

        except Exception as e:
            QMessageBox.critical(self, "Load Error", f"Failed to load or parse tracer file:\n{e}")
            return

        if self.loaded_files:
            base_df = self.loaded_files[0]['original_df']
            base_start, base_end = base_df.index.min(), base_df.index.max()
            new_start, new_end = df.index.min(), df.index.max()

            if not (new_start < base_end and new_end > base_start):
                msg = (f"The time range of '{os.path.basename(path)}' does not overlap "
                       f"with the base file '{os.path.basename(self.loaded_files[0]['path'])}'.\n\n"
                       f"Base Range: {base_start} to {base_end}\n"
                       f"New File Range: {new_start} to {new_end}")
                QMessageBox.warning(self, "Time Range Error", msg)
                return

        index = len(self.loaded_files)
        data_dict = {
            'path': path, 'original_df': df, 'unit_factor': 1.0,
            'processed_df': None, 'composition_df': None,
            'background_values': None, 'background_was_nan': None
        }
        self.loaded_files.append(data_dict)

        widget = TracerFileWidget(path, data_dict, index, self)
        widget.set_full_time_range_text(df.index.min(), df.index.max())
        widget.minimize_requested.connect(self.toggle_minimize_widget)
        widget.widget_deleted.connect(self.delete_tracer_widget)

        self.tracer_widgets.append(widget)
        self.widgets_layout.addWidget(widget)

        if len(self.loaded_files) == 1:
            self.global_controls_widget.setEnabled(True)
            self.start_time_input.setText(df.index.min().strftime('%Y-%m-%d %H:%M'))
            self.end_time_input.setText(df.index.max().strftime('%Y-%m-%d %H:%M'))

        self.update_all_files()

    def toggle_minimize_widget(self, index):
        widget = self.tracer_widgets[index]

        if widget.is_minimized:
            placeholder = self.placeholder_widgets.pop(index)
            self.widgets_layout.replaceWidget(placeholder, widget)
            placeholder.deleteLater()
            widget.show()
            widget.is_minimized = False
        else:
            placeholder = MinimizedWidget(widget.file_name)
            placeholder.restore_requested.connect(lambda: self.toggle_minimize_widget(index))
            self.widgets_layout.replaceWidget(widget, placeholder)
            widget.hide()
            self.placeholder_widgets[index] = placeholder
            widget.is_minimized = True

    def restore_all_windows(self):
        for index in list(self.placeholder_widgets.keys()):
            self.toggle_minimize_widget(index)

    def recalculate_processed_data(self, index):
        if not (0 <= index < len(self.loaded_files)): return

        data_dict = self.loaded_files[index]
        df = data_dict['original_df'].copy()

        factor = data_dict.get('unit_factor', 1.0)
        numeric_cols = df.select_dtypes(include=np.number).columns
        df[numeric_cols] = df[numeric_cols] * factor

        try:
            reso = self.reso_input.text().strip()
            if reso:
                df = df.resample(reso).mean()
        except Exception as e:
            print(f"Resampling error: {e}")

        try:
            start_time = pd.to_datetime(self.start_time_input.text(), format='%Y-%m-%d %H:%M')
            end_time = pd.to_datetime(self.end_time_input.text(), format='%Y-%m-%d %H:%M')
            if start_time < end_time:
                df = df.loc[start_time:end_time]
        except Exception:
            pass

        if data_dict.get('background_values') is not None:
            bg_values = data_dict['background_values']
            df = df.subtract(bg_values, axis='columns')

        df.dropna(how='all', inplace=True)
        data_dict['processed_df'] = df

    def update_all_files(self):
        for i in range(len(self.loaded_files)):
            self.recalculate_processed_data(i)
        QApplication.processEvents()
        print("All files updated.")

    def delete_tracer_widget(self, index_to_delete):
        if not (0 <= index_to_delete < len(self.tracer_widgets)): return

        for child in self.findChildren(QDialog):
            if isinstance(child, (PlottingWindow, DataViewerWindow)):
                child.close()
        self.global_slope_plot_action.setEnabled(False)
        self.global_results_df_list = []

        # Determine which widget is at the index and remove it
        widget_to_remove = self.widgets_layout.itemAt(index_to_delete).widget()
        self.widgets_layout.removeWidget(widget_to_remove)
        widget_to_remove.deleteLater()

        # Remove from internal lists
        self.tracer_widgets.pop(index_to_delete)
        self.loaded_files.pop(index_to_delete)
        self.placeholder_widgets.pop(index_to_delete, None)

        # Re-index all subsequent widgets and placeholders
        for i in range(index_to_delete, len(self.tracer_widgets)):
            self.tracer_widgets[i].update_index(i)

        new_placeholders = {}
        for old_idx, placeholder in self.placeholder_widgets.items():
            if old_idx > index_to_delete:
                new_placeholders[old_idx - 1] = placeholder
            else:
                new_placeholders[old_idx] = placeholder
        self.placeholder_widgets = new_placeholders

        if not self.tracer_widgets:
            self.global_controls_widget.setEnabled(False)
            self.start_time_input.clear()
            self.end_time_input.clear()
            self.bg_start_time_input.clear()
            self.bg_end_time_input.clear()
        elif index_to_delete == 0 and self.tracer_widgets:
            new_base_df = self.loaded_files[0]['original_df']
            self.start_time_input.setText(new_base_df.index.min().strftime('%Y-%m-%d %H:%M'))
            self.end_time_input.setText(new_base_df.index.max().strftime('%Y-%m-%d %H:%M'))
            self.update_all_files()

    def _on_subtract_background(self):
        try:
            bg_start = pd.to_datetime(self.bg_start_time_input.text(), format='%Y-%m-%d %H:%M')
            bg_end = pd.to_datetime(self.bg_end_time_input.text(), format='%Y-%m-%d %H:%M')
            if bg_start >= bg_end:
                QMessageBox.warning(self, "Input Error", "Background start time must be before end time.")
                return
        except ValueError:
            QMessageBox.warning(self, "Input Error",
                                "Please enter a valid background time range in 'yyyy-mm-dd hh:mm' format.")
            return

        for i, data_dict in enumerate(self.loaded_files):
            df_orig = data_dict['original_df']
            factor = data_dict.get('unit_factor', 1.0)
            file_start, file_end = df_orig.index.min(), df_orig.index.max()

            if not (bg_start < file_end and bg_end > file_start):
                QMessageBox.warning(self, "Range Warning",
                                    f"The background range is outside the total time range for file:\n{os.path.basename(data_dict['path'])}\n\nBackground for this file will be set to 0.")
                bg_series_final = pd.Series(0, index=df_orig.columns)
                was_nan = pd.Series(True, index=df_orig.columns)
            else:
                factored_df = df_orig.copy()
                numeric_cols = factored_df.select_dtypes(include=np.number).columns
                factored_df[numeric_cols] *= factor

                resampled_df = factored_df
                try:
                    reso = self.reso_input.text().strip()
                    if reso:
                        resampled_df = factored_df.resample(reso).mean()
                except Exception as e:
                    print(f"BG Resampling error for {data_dict['path']}: {e}")

                bg_slice = resampled_df.loc[bg_start:bg_end]

                bg_series_raw = bg_slice.mean()
                was_nan = bg_series_raw.isna()
                bg_series_final = bg_series_raw.fillna(0)

            data_dict['background_values'] = bg_series_final
            data_dict['background_was_nan'] = was_nan

            self.tracer_widgets[i].update_background_column()

        self.update_all_files()
        QMessageBox.information(self, "Success", "Background values calculated and subtracted. Data has been updated.")

    def _on_global_tracer_plot(self):
        if not self.loaded_files: return
        fig = Figure(figsize=(10, 7), dpi=100)
        ax = fig.add_subplot(111)

        plotted_anything = False
        for widget in self.tracer_widgets:
            plotted_formulas = widget._get_checked_formulas("Plotted")
            df = widget.data_dict.get('processed_df')
            if df is None: continue
            for formula in plotted_formulas:
                color = self.get_color_for_formula(formula)
                ax.plot(df.index, df[formula], label=f"{widget.file_name[:15]}... - {formula}", color=color)
                plotted_anything = True
        title = "Global Tracer Plot"

        if not plotted_anything:
            QMessageBox.warning(self, "Selection Error", "No variables selected for plotting in any file.")
            return

        ax.set_yscale('log')

        axis_label_fontsize = 14
        tick_label_fontsize = 12
        legend_fontsize = 10

        ax.set_title(title)
        ax.set_xlabel("Time", fontsize=axis_label_fontsize)
        ax.set_ylabel("Value (Factored, BG Subtracted)", fontsize=axis_label_fontsize)
        ax.tick_params(axis='both', which='major', labelsize=tick_label_fontsize)
        ax.legend(fontsize=legend_fontsize)
        ax.grid(True, which='both', linestyle='--', alpha=0.6)
        fig.tight_layout()

        plot_window = PlottingWindow(fig, title=title, parent=self)
        plot_window.exec_()

    def _on_global_slope_table(self):
        if not self.loaded_files: return
        self.global_results_df_list = []

        x_instructions = [{'widget_index': i, 'x_name': x_name} for i, w in enumerate(self.tracer_widgets) for x_name in
                          w._get_checked_formulas("X-axis")]
        y_instructions = [{'widget_index': i, 'y_name': y_name} for i, w in enumerate(self.tracer_widgets) for y_name in
                          w._get_checked_formulas("Selected")]

        if not y_instructions:
            QMessageBox.warning(self, "Selection Error", "Please select variables in the 'Selected' column.")
            return

        if not x_instructions:
            dialog = TimeTransformDialog(self)
            if dialog.exec_() == QDialog.Accepted and dialog.selection:
                time_transform = dialog.selection
                all_results = {}
                for y_instr in y_instructions:
                    y_widget = self.tracer_widgets[y_instr['widget_index']]
                    y_df = y_widget.data_dict['processed_df']
                    if y_df is None or y_df.empty: continue
                    y_name = y_instr['y_name']

                    time_index_seconds = (y_df.index - y_df.index[0]).total_seconds()
                    x_series_data = pd.Series(time_index_seconds, index=y_df.index)
                    if time_transform != 't':
                        power = int(time_transform[1])
                        x_series_data = x_series_data ** power

                    slope, r2, n = calculate_slope_r2(x_series_data, y_df[y_name])
                    result_key = f"{y_widget.file_name} - {y_name}"
                    all_results[result_key] = {'Slope': slope, 'R2': r2, 'Data Number': n}

                results_df = pd.DataFrame.from_dict(all_results, orient='index')
                results_df.index.name = "Y-Variable (Source - Formula)"
                self.global_results_df_list.append({'df': results_df, 'title': f"Global Results vs {time_transform}"})
                table_window = SlopeTableWindow(results_df, title=f"Global Slope Results (vs {time_transform})",
                                                parent=self)
                table_window.exec_()
        else:
            for x_instr in x_instructions:
                master_widget_idx, master_x_name = x_instr['widget_index'], x_instr['x_name']
                master_df = self.loaded_files[master_widget_idx]['processed_df']
                if master_df is None or master_x_name not in master_df: continue
                master_x_series = master_df[master_x_name]

                current_table_results = {}
                for y_instr in y_instructions:
                    y_widget_idx, y_name = y_instr['widget_index'], y_instr['y_name']
                    y_df = self.loaded_files[y_widget_idx]['processed_df']
                    if y_df is None or y_name not in y_df: continue
                    y_series = y_df[y_name]

                    combined = pd.concat([master_x_series.rename('master_x'), y_series.rename('current_y')], axis=1)
                    aligned = combined.dropna()

                    slope, r2, n = calculate_slope_r2(aligned['master_x'], aligned['current_y'])

                    y_source_file = os.path.basename(self.loaded_files[y_widget_idx]['path'])
                    result_key = f"{y_source_file} - {y_name}"
                    current_table_results[result_key] = {'Slope': slope, 'R2': r2, 'Data Number': n}

                results_df = pd.DataFrame.from_dict(current_table_results, orient='index')
                results_df.index.name = 'Y-Variable (Source - Formula)'
                title_suffix = f"vs {master_x_name} from {os.path.basename(self.loaded_files[master_widget_idx]['path'])}"
                self.global_results_df_list.append({'df': results_df, 'title': title_suffix})

                table_window = SlopeTableWindow(results_df, title=f"Global Slope Results ({title_suffix})", parent=self)
                table_window.exec_()

        if self.global_results_df_list:
            self.global_slope_plot_action.setEnabled(True)

    def _on_global_slope_plot(self):
        if not hasattr(self, 'global_results_df_list') or not self.global_results_df_list:
            QMessageBox.warning(self, "No Data", "Please generate the Global SlopeTable first.")
            return

        for result_item in self.global_results_df_list:
            df_full = result_item['df']
            title_suffix = result_item['title']
            df = df_full.dropna(subset=['Slope', 'R2']).reset_index()
            if df.empty:
                continue

            fig = Figure(figsize=(12, 8), dpi=100)
            ax = fig.add_subplot(111)

            df['FormulaLabel'] = df['Y-Variable (Source - Formula)'].apply(lambda x: x.split(' - ')[-1])

            df_low_r2 = df[df['R2'] < 0.6]
            df_high_r2 = df[df['R2'] >= 0.6]

            if not df_low_r2.empty: ax.scatter(df_low_r2['FormulaLabel'], df_low_r2['Slope'], c='lightgray',
                                               label='R² < 0.6', s=50, zorder=2)
            if not df_high_r2.empty:
                colors = df_high_r2['R2']
                cmap = matplotlib.colormaps.get_cmap('viridis')
                sc = ax.scatter(df_high_r2['FormulaLabel'], df_high_r2['Slope'], c=colors, cmap=cmap, vmin=0.6,
                                vmax=1.0, label='R² >= 0.6', s=50, zorder=3)
                fig.colorbar(sc, ax=ax, label='R² Value')

            for _, row in df.iterrows():
                ax.text(row['FormulaLabel'], row['Slope'], f"  {int(row['Data Number'])}", fontsize=8)

            axis_label_fontsize = 14
            tick_label_fontsize = 12
            legend_fontsize = 10

            plot_title = f"Global Slope Plot ({title_suffix})"
            ax.set_title(plot_title)
            ax.set_ylabel("Slope", fontsize=axis_label_fontsize)
            ax.set_xlabel("", fontsize=axis_label_fontsize)
            ax.tick_params(axis='y', which='major', labelsize=tick_label_fontsize)
            ax.tick_params(axis='x', rotation=90, labelsize=10)
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend(fontsize=legend_fontsize)
            fig.tight_layout()

            plot_window = PlottingWindow(fig, title=plot_title, parent=self)
            plot_window.exec_()

    def _show_data_viewer(self, mode):
        if not self.loaded_files:
            QMessageBox.information(self, "No Data", "Please load a tracer file first.")
            return

        series_to_display = []
        for widget in self.tracer_widgets:
            formulas = widget._get_checked_formulas(mode)
            df = widget.data_dict.get('processed_df')
            if df is None: continue

            for formula in formulas:
                if formula in df.columns:
                    series = df[formula].copy()

                    short_filename = (widget.file_name[:20] + '...') if len(widget.file_name) > 20 else widget.file_name
                    series.name = f"{short_filename} - {formula}"
                    series_to_display.append(series)

        if not series_to_display:
            QMessageBox.information(self, "No Selection", f"No variables selected in the '{mode}' column.")
            return

        try:
            combined_df = pd.concat(series_to_display, axis=1)
            combined_df.sort_index(inplace=True)
            display_df = combined_df.reset_index().rename(columns={'index': 'Time'})

            viewer_window = DataViewerWindow(display_df, title=f"Data Viewer - {mode} Data", parent=self)
            viewer_window.exec_()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Could not create data view:\n{e}")

    def closeEvent(self, event):
        """Ensure a graceful exit, especially in IDEs like Spyder."""
        for child in self.findChildren(QDialog):
            child.close()
        QApplication.instance().quit()
        event.accept()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = SlopeAnalysisGUI()
    win.show()
    sys.exit(app.exec_())