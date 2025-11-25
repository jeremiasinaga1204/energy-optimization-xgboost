"""
Energy Consumption Analysis and Optimization Package

This package provides tools for analyzing energy consumption patterns
and optimizing energy usage for the BTI Campus Building using XGBoost.

Modules:
    - data_input: Functions for loading and validating device inventory data
    - energy_calculator: Energy consumption and cost calculations
    - xgboost_model: XGBoost model implementation for energy prediction
    - visualization: Plotting and visualization functions
    - optimization: Energy optimization recommendations
"""

__version__ = '1.0.0'
__author__ = 'Energy Optimization Team'

from . import data_input
from . import energy_calculator
from . import xgboost_model
from . import visualization
from . import optimization

__all__ = [
    'data_input',
    'energy_calculator',
    'xgboost_model',
    'visualization',
    'optimization'
]
