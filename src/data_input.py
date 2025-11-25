"""
Data Input Module for Energy Consumption Analysis

This module handles loading, validation, and preprocessing of device inventory
data from Excel files.

Functions:
    - load_device_inventory: Load device data from Excel template
    - validate_data: Validate and clean input data
    - calculate_power_metrics: Calculate power consumption metrics
    - export_processed_data: Export processed data to Excel
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Tuple

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_device_inventory(filepath: Optional[Path] = None) -> pd.DataFrame:
    """
    Load device inventory data from Excel file.
    
    This function reads the device inventory template and returns a cleaned
    DataFrame with all device information.
    
    Args:
        filepath: Path to the Excel file. If None, uses default template path.
        
    Returns:
        DataFrame containing device inventory data with columns:
        - No: Device number
        - Device_Name: Name of the device
        - Quantity: Number of devices
        - Power_Watt: Power consumption per device in Watts
        - Operating_Hours_Per_Day: Daily operating hours
        - Device_Category: Category classification
        
    Raises:
        FileNotFoundError: If the Excel file doesn't exist
        ValueError: If required columns are missing
        
    Example:
        >>> df = load_device_inventory()
        >>> print(df.head())
    """
    if filepath is None:
        filepath = config.RAW_DATA_DIR / config.DEVICE_INVENTORY_FILE
    
    logger.info(f"Loading device inventory from: {filepath}")
    
    try:
        # Load Excel file
        df = pd.read_excel(filepath, sheet_name='Device_Inventory', engine='openpyxl')
        logger.info(f"Successfully loaded {len(df)} devices from inventory")
        
        # Validate required columns
        required_columns = ['No', 'Device_Name', 'Quantity', 'Power_Watt', 
                          'Operating_Hours_Per_Day', 'Device_Category']
        
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        return df
        
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        raise
    except Exception as e:
        logger.error(f"Error loading device inventory: {e}")
        raise


def validate_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Validate and clean the device inventory data.
    
    This function performs data validation, handles missing values,
    and ensures data types are correct.
    
    Args:
        df: Raw device inventory DataFrame
        
    Returns:
        Tuple containing:
        - Cleaned DataFrame
        - Dictionary with validation report
        
    Example:
        >>> df_raw = load_device_inventory()
        >>> df_clean, report = validate_data(df_raw)
        >>> print(report['total_devices'])
    """
    logger.info("Validating device inventory data...")
    
    validation_report = {
        'total_devices': len(df),
        'missing_values': {},
        'invalid_values': {},
        'cleaned_records': 0,
        'warnings': []
    }
    
    # Create a copy to avoid modifying original
    df_clean = df.copy()
    
    # Check for missing values
    for col in df_clean.columns:
        missing_count = df_clean[col].isna().sum()
        if missing_count > 0:
            validation_report['missing_values'][col] = missing_count
            logger.warning(f"Column '{col}' has {missing_count} missing values")
    
    # Fill missing values
    # For numeric columns, use 0
    numeric_cols = ['Quantity', 'Power_Watt', 'Operating_Hours_Per_Day']
    for col in numeric_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0)
    
    # For Device_Name, use 'Unknown Device'
    if 'Device_Name' in df_clean.columns:
        df_clean['Device_Name'] = df_clean['Device_Name'].fillna('Unknown Device')
    
    # For Device_Category, use 'Other'
    if 'Device_Category' in df_clean.columns:
        df_clean['Device_Category'] = df_clean['Device_Category'].fillna('Other')
    
    # Validate numeric ranges
    # Quantity must be positive
    invalid_qty = (df_clean['Quantity'] <= 0).sum()
    if invalid_qty > 0:
        validation_report['invalid_values']['Quantity'] = invalid_qty
        validation_report['warnings'].append(f"{invalid_qty} devices have invalid quantity (<=0)")
        df_clean.loc[df_clean['Quantity'] <= 0, 'Quantity'] = 1
    
    # Power must be positive
    invalid_power = (df_clean['Power_Watt'] <= 0).sum()
    if invalid_power > 0:
        validation_report['invalid_values']['Power_Watt'] = invalid_power
        validation_report['warnings'].append(f"{invalid_power} devices have invalid power (<=0)")
        df_clean.loc[df_clean['Power_Watt'] <= 0, 'Power_Watt'] = 10  # Default to 10W
    
    # Operating hours must be between 0 and 24
    invalid_hours = ((df_clean['Operating_Hours_Per_Day'] < 0) | 
                     (df_clean['Operating_Hours_Per_Day'] > 24)).sum()
    if invalid_hours > 0:
        validation_report['invalid_values']['Operating_Hours_Per_Day'] = invalid_hours
        validation_report['warnings'].append(f"{invalid_hours} devices have invalid operating hours")
        df_clean.loc[df_clean['Operating_Hours_Per_Day'] < 0, 'Operating_Hours_Per_Day'] = 0
        df_clean.loc[df_clean['Operating_Hours_Per_Day'] > 24, 'Operating_Hours_Per_Day'] = 24
    
    # Reset index
    df_clean = df_clean.reset_index(drop=True)
    df_clean['No'] = range(1, len(df_clean) + 1)
    
    validation_report['cleaned_records'] = len(df_clean)
    
    logger.info(f"Validation complete. {len(df_clean)} records processed.")
    
    return df_clean, validation_report


def calculate_power_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate power consumption metrics for each device.
    
    This function adds calculated columns for total power and daily energy
    consumption based on device quantity and operating hours.
    
    Args:
        df: Device inventory DataFrame with validated data
        
    Returns:
        DataFrame with additional columns:
        - Total_Power_Watt: Total power consumption (Quantity × Power_Watt)
        - Daily_Energy_kWh: Daily energy consumption (Total_Power × Hours / 1000)
        - Monthly_Energy_kWh: Monthly energy consumption (Daily × Working_Days)
        - Monthly_Cost_Rp: Monthly cost (Monthly_Energy × Tariff)
        
    Example:
        >>> df_validated = validate_data(df_raw)[0]
        >>> df_metrics = calculate_power_metrics(df_validated)
        >>> print(df_metrics['Daily_Energy_kWh'].sum())
    """
    logger.info("Calculating power consumption metrics...")
    
    df_metrics = df.copy()
    
    # Calculate Total Power (Watts)
    # Total_Power = Quantity × Power per device
    df_metrics['Total_Power_Watt'] = df_metrics['Quantity'] * df_metrics['Power_Watt']
    
    # Calculate Daily Energy Consumption (kWh)
    # Energy (kWh) = Power (W) × Time (hours) / 1000
    df_metrics['Daily_Energy_kWh'] = (
        df_metrics['Total_Power_Watt'] * df_metrics['Operating_Hours_Per_Day'] / 1000
    )
    
    # Calculate Monthly Energy Consumption (kWh)
    # Monthly = Daily × Working Days per Month
    df_metrics['Monthly_Energy_kWh'] = (
        df_metrics['Daily_Energy_kWh'] * config.WORKING_DAYS_PER_MONTH
    )
    
    # Calculate Monthly Cost (Rupiah)
    # Cost = Energy × Tariff
    df_metrics['Monthly_Cost_Rp'] = (
        df_metrics['Monthly_Energy_kWh'] * config.ELECTRICITY_TARIFF
    )
    
    # Calculate Annual Energy and Cost
    df_metrics['Annual_Energy_kWh'] = df_metrics['Monthly_Energy_kWh'] * 12
    df_metrics['Annual_Cost_Rp'] = df_metrics['Monthly_Cost_Rp'] * 12
    
    # Round values for readability
    numeric_columns = ['Total_Power_Watt', 'Daily_Energy_kWh', 'Monthly_Energy_kWh',
                      'Monthly_Cost_Rp', 'Annual_Energy_kWh', 'Annual_Cost_Rp']
    for col in numeric_columns:
        df_metrics[col] = df_metrics[col].round(2)
    
    logger.info("Power metrics calculation complete")
    
    return df_metrics


def export_processed_data(df: pd.DataFrame, 
                         filepath: Optional[Path] = None,
                         include_summary: bool = True) -> Path:
    """
    Export processed device data to Excel file.
    
    This function saves the processed DataFrame with all calculated metrics
    to an Excel file in the processed data directory.
    
    Args:
        df: Processed DataFrame with power metrics
        filepath: Output file path. If None, uses default path.
        include_summary: Whether to include a summary sheet
        
    Returns:
        Path to the exported file
        
    Example:
        >>> df_processed = calculate_power_metrics(df_validated)
        >>> output_path = export_processed_data(df_processed)
        >>> print(f"Data exported to: {output_path}")
    """
    if filepath is None:
        # Ensure directory exists
        config.PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
        filepath = config.PROCESSED_DATA_DIR / config.PROCESSED_DATA_FILE
    
    logger.info(f"Exporting processed data to: {filepath}")
    
    try:
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Write main data
            df.to_excel(writer, sheet_name='Device_Data', index=False)
            
            if include_summary:
                # Create summary by category
                summary = df.groupby('Device_Category').agg({
                    'Device_Name': 'count',
                    'Quantity': 'sum',
                    'Total_Power_Watt': 'sum',
                    'Daily_Energy_kWh': 'sum',
                    'Monthly_Energy_kWh': 'sum',
                    'Monthly_Cost_Rp': 'sum'
                }).rename(columns={'Device_Name': 'Device_Types'})
                
                summary = summary.round(2)
                summary.to_excel(writer, sheet_name='Summary_By_Category')
                
                # Create overall summary
                overall = pd.DataFrame({
                    'Metric': [
                        'Total Device Types',
                        'Total Device Count',
                        'Total Power (kW)',
                        'Daily Energy (kWh)',
                        'Monthly Energy (kWh)',
                        'Monthly Cost (Rp)',
                        'Annual Energy (kWh)',
                        'Annual Cost (Rp)'
                    ],
                    'Value': [
                        len(df),
                        df['Quantity'].sum(),
                        df['Total_Power_Watt'].sum() / 1000,
                        df['Daily_Energy_kWh'].sum(),
                        df['Monthly_Energy_kWh'].sum(),
                        df['Monthly_Cost_Rp'].sum(),
                        df['Annual_Energy_kWh'].sum(),
                        df['Annual_Cost_Rp'].sum()
                    ]
                })
                overall['Value'] = overall['Value'].round(2)
                overall.to_excel(writer, sheet_name='Overall_Summary', index=False)
        
        logger.info(f"Data successfully exported to {filepath}")
        return filepath
        
    except Exception as e:
        logger.error(f"Error exporting data: {e}")
        raise


def get_device_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get a summary of the device inventory.
    
    Args:
        df: Device inventory DataFrame with metrics
        
    Returns:
        Dictionary containing summary statistics
        
    Example:
        >>> summary = get_device_summary(df_processed)
        >>> print(f"Total daily consumption: {summary['total_daily_kwh']} kWh")
    """
    summary = {
        'total_device_types': len(df),
        'total_devices': int(df['Quantity'].sum()),
        'categories': df['Device_Category'].unique().tolist(),
        'total_power_kw': round(df['Total_Power_Watt'].sum() / 1000, 2),
        'total_daily_kwh': round(df['Daily_Energy_kWh'].sum(), 2),
        'total_monthly_kwh': round(df['Monthly_Energy_kWh'].sum(), 2),
        'total_monthly_cost_rp': round(df['Monthly_Cost_Rp'].sum(), 2),
        'total_annual_kwh': round(df['Annual_Energy_kWh'].sum(), 2),
        'total_annual_cost_rp': round(df['Annual_Cost_Rp'].sum(), 2),
        'top_consumer': df.loc[df['Daily_Energy_kWh'].idxmax(), 'Device_Name'] if len(df) > 0 else None
    }
    
    return summary


def load_and_process_data(filepath: Optional[Path] = None) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Convenience function to load, validate, and process data in one call.
    
    This is the main entry point for data loading in the application.
    
    Args:
        filepath: Path to the Excel file. If None, uses default template.
        
    Returns:
        Tuple containing:
        - Processed DataFrame with all metrics
        - Dictionary with processing summary
        
    Example:
        >>> df, summary = load_and_process_data()
        >>> print(f"Loaded {summary['total_devices']} devices")
    """
    # Load data
    df_raw = load_device_inventory(filepath)
    
    # Validate data
    df_validated, validation_report = validate_data(df_raw)
    
    # Calculate metrics
    df_processed = calculate_power_metrics(df_validated)
    
    # Get summary
    summary = get_device_summary(df_processed)
    summary['validation_report'] = validation_report
    
    return df_processed, summary


if __name__ == "__main__":
    # Test the module
    print("Testing data_input module...")
    print("-" * 50)
    
    try:
        # Load and process data
        df, summary = load_and_process_data()
        
        print(f"Total device types: {summary['total_device_types']}")
        print(f"Total devices: {summary['total_devices']}")
        print(f"Total daily consumption: {summary['total_daily_kwh']} kWh")
        print(f"Total monthly cost: Rp {summary['total_monthly_cost_rp']:,.0f}")
        
        # Export processed data
        output_path = export_processed_data(df)
        print(f"\nData exported to: {output_path}")
        
    except FileNotFoundError:
        print("Note: Device inventory template not found.")
        print("Please create the template file first.")
