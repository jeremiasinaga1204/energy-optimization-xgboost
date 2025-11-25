"""
Configuration file for Energy Consumption Analysis and Optimization System
BTI Campus Building - Universitas Pertahanan RI

This module contains all configuration parameters used throughout the project.
"""

import os
from pathlib import Path

# =============================================================================
# BASE PATHS
# =============================================================================
# Get the project root directory
PROJECT_ROOT = Path(__file__).parent.absolute()

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = DATA_DIR / "results"

# Output directories
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
TABLES_DIR = OUTPUTS_DIR / "tables"

# =============================================================================
# ELECTRICITY TARIFF CONFIGURATION
# =============================================================================
# Standard PLN (Perusahaan Listrik Negara) tariff rate in Indonesia
# Based on commercial building rates (tariff B2/B3)
ELECTRICITY_TARIFF = 1467  # Rupiah per kWh

# =============================================================================
# TIME CONFIGURATION
# =============================================================================
# Working days per month (excluding weekends)
WORKING_DAYS_PER_MONTH = 22

# Number of days for simulation
SIMULATION_DAYS = 90

# Months per year
MONTHS_PER_YEAR = 12

# =============================================================================
# ENVIRONMENTAL FACTORS
# =============================================================================
# CO2 emission factor (kg CO2 per kWh) - Indonesia grid average
CO2_EMISSION_FACTOR = 0.85

# =============================================================================
# MODEL CONFIGURATION
# =============================================================================
# Test set size ratio for train-test split
MODEL_TEST_SIZE = 0.2

# Random state for reproducibility
RANDOM_STATE = 42

# =============================================================================
# XGBOOST HYPERPARAMETERS
# =============================================================================
XGBOOST_PARAMS = {
    'learning_rate': 0.1,
    'max_depth': 5,
    'n_estimators': 100,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'random_state': RANDOM_STATE,
    'objective': 'reg:squarederror',
    'n_jobs': -1
}

# =============================================================================
# SIMULATION PARAMETERS
# =============================================================================
# Temperature range for simulation (Celsius)
MIN_TEMPERATURE = 24
MAX_TEMPERATURE = 30

# Occupancy rate range (percentage)
MIN_OCCUPANCY = 0
MAX_OCCUPANCY = 100

# Working hours (8 AM to 5 PM)
WORKING_HOUR_START = 8
WORKING_HOUR_END = 17

# =============================================================================
# VISUALIZATION SETTINGS
# =============================================================================
# Figure DPI for high resolution output
FIGURE_DPI = 300

# Figure sizes (width, height) in inches
FIGURE_SIZE_LARGE = (12, 8)
FIGURE_SIZE_MEDIUM = (10, 6)
FIGURE_SIZE_SMALL = (8, 5)

# Font sizes
FONT_SIZE_TITLE = 14
FONT_SIZE_LABEL = 12
FONT_SIZE_TICK = 10

# Color palettes
COLOR_PALETTE = 'viridis'
CATEGORY_PALETTE = 'Set2'
BAR_PALETTE = 'tab10'

# =============================================================================
# OPTIMIZATION SCENARIOS
# =============================================================================
OPTIMIZATION_SCENARIOS = {
    'inverter_ac': {
        'name': 'Replace with Inverter AC',
        'savings_percentage': 0.30,  # 30% savings
        'applicable_categories': ['Cooling'],
        'applicable_devices': ['AC Split']
    },
    'led_replacement': {
        'name': 'LED Replacement',
        'savings_percentage': 0.60,  # 60% savings
        'applicable_categories': ['Lighting'],
        'applicable_devices': []
    },
    'optimize_hours': {
        'name': 'Optimize Operating Hours',
        'savings_percentage': 0.15,  # 15% savings
        'applicable_categories': ['all'],
        'applicable_devices': []
    },
    'power_management': {
        'name': 'Power Management for Computers',
        'savings_percentage': 0.20,  # 20% savings
        'applicable_categories': ['Computing'],
        'applicable_devices': []
    },
    'peak_shifting': {
        'name': 'Peak Load Shifting',
        'savings_percentage': 0.10,  # 10% cost savings through off-peak usage
        'applicable_categories': ['all'],
        'applicable_devices': []
    }
}

# =============================================================================
# FILE NAMES
# =============================================================================
# Input files
DEVICE_INVENTORY_FILE = "device_inventory_template.xlsx"

# Output files
PROCESSED_DATA_FILE = "processed_energy_data.xlsx"
CONSUMPTION_SUMMARY_FILE = "consumption_summary.xlsx"
OPTIMIZATION_RECOMMENDATIONS_FILE = "optimization_recommendations.xlsx"
MODEL_FILE = "xgboost_energy_model.pkl"

# Figure files
CONSUMPTION_BY_DEVICE_FIGURE = "consumption_by_device.png"
CONSUMPTION_PIE_CHART_FIGURE = "consumption_pie_chart.png"
PREDICTION_VS_ACTUAL_FIGURE = "prediction_vs_actual.png"
FEATURE_IMPORTANCE_FIGURE = "feature_importance.png"
DAILY_PATTERN_FIGURE = "daily_pattern.png"
CONSUMPTION_DISTRIBUTION_FIGURE = "consumption_distribution.png"
MONTHLY_CONSUMPTION_FIGURE = "monthly_consumption.png"
HEATMAP_SCHEDULE_FIGURE = "heatmap_schedule.png"

# =============================================================================
# DEVICE CATEGORIES
# =============================================================================
DEVICE_CATEGORIES = [
    'Cooling',
    'Lighting',
    'Computing',
    'Presentation',
    'Office',
    'Appliance',
    'IT_Infrastructure',
    'Security'
]


def get_full_path(directory: Path, filename: str) -> Path:
    """
    Get the full path for a file in a specified directory.
    
    Args:
        directory: Directory path (Path object)
        filename: Name of the file
        
    Returns:
        Full path as a Path object
    """
    return directory / filename


def create_directories() -> None:
    """
    Create all necessary directories if they don't exist.
    """
    directories = [
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        RESULTS_DIR,
        FIGURES_DIR,
        TABLES_DIR
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    # Print configuration summary
    print("=" * 60)
    print("Energy Optimization Project Configuration")
    print("=" * 60)
    print(f"Project Root: {PROJECT_ROOT}")
    print(f"Electricity Tariff: Rp {ELECTRICITY_TARIFF:,}/kWh")
    print(f"Working Days/Month: {WORKING_DAYS_PER_MONTH}")
    print(f"Simulation Days: {SIMULATION_DAYS}")
    print(f"Model Test Size: {MODEL_TEST_SIZE * 100}%")
    print("=" * 60)
