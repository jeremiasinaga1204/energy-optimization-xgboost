"""
Energy Calculator Module for Energy Consumption Analysis

This module provides functions for calculating energy consumption,
costs, and generating summary statistics.

Functions:
    - calculate_daily_consumption: Calculate daily energy consumption
    - calculate_monthly_consumption: Calculate monthly energy consumption
    - calculate_energy_costs: Calculate energy costs based on tariff
    - get_consumption_statistics: Generate summary statistics
    - get_consumption_by_category: Breakdown consumption by category
    - export_summary_tables: Export summary tables to Excel
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def calculate_daily_consumption(df: pd.DataFrame, 
                                device_col: str = 'Device_Name',
                                power_col: str = 'Total_Power_Watt',
                                hours_col: str = 'Operating_Hours_Per_Day') -> pd.DataFrame:
    """
    Calculate daily energy consumption for each device.
    
    Energy formula: E(kWh) = P(W) × t(h) / 1000
    
    Args:
        df: Device inventory DataFrame
        device_col: Column name for device names
        power_col: Column name for total power
        hours_col: Column name for operating hours
        
    Returns:
        DataFrame with daily consumption calculations:
        - Device_Name
        - Daily_kWh: Daily energy consumption in kWh
        - Percentage: Percentage of total consumption
        
    Example:
        >>> daily = calculate_daily_consumption(df_processed)
        >>> print(f"Total daily: {daily['Daily_kWh'].sum()} kWh")
    """
    logger.info("Calculating daily energy consumption...")
    
    result = pd.DataFrame()
    result['Device_Name'] = df[device_col]
    
    # Calculate energy consumption: Power (W) × Hours / 1000 = kWh
    result['Daily_kWh'] = (df[power_col] * df[hours_col] / 1000).round(2)
    
    # Calculate percentage of total
    total_daily = result['Daily_kWh'].sum()
    result['Percentage'] = ((result['Daily_kWh'] / total_daily) * 100).round(2)
    
    # Sort by consumption (descending)
    result = result.sort_values('Daily_kWh', ascending=False).reset_index(drop=True)
    
    logger.info(f"Total daily consumption: {total_daily:.2f} kWh")
    
    return result


def calculate_monthly_consumption(df: pd.DataFrame,
                                  daily_col: str = 'Daily_Energy_kWh',
                                  working_days: Optional[int] = None) -> pd.DataFrame:
    """
    Calculate monthly energy consumption for each device.
    
    Args:
        df: Device inventory DataFrame with daily consumption
        daily_col: Column name for daily energy consumption
        working_days: Number of working days per month. Uses config if None.
        
    Returns:
        DataFrame with monthly consumption calculations
        
    Example:
        >>> monthly = calculate_monthly_consumption(df_processed)
        >>> print(f"Total monthly: {monthly['Monthly_kWh'].sum()} kWh")
    """
    if working_days is None:
        working_days = config.WORKING_DAYS_PER_MONTH
    
    logger.info(f"Calculating monthly consumption ({working_days} working days)...")
    
    result = df[['Device_Name', 'Device_Category', daily_col]].copy()
    result['Monthly_kWh'] = (result[daily_col] * working_days).round(2)
    
    # Calculate percentage of total
    total_monthly = result['Monthly_kWh'].sum()
    result['Percentage'] = ((result['Monthly_kWh'] / total_monthly) * 100).round(2)
    
    # Sort by consumption (descending)
    result = result.sort_values('Monthly_kWh', ascending=False).reset_index(drop=True)
    
    logger.info(f"Total monthly consumption: {total_monthly:.2f} kWh")
    
    return result


def calculate_energy_costs(df: pd.DataFrame,
                          energy_col: str = 'Monthly_Energy_kWh',
                          tariff: Optional[float] = None) -> pd.DataFrame:
    """
    Calculate energy costs based on electricity tariff.
    
    Cost formula: Cost = Energy (kWh) × Tariff (Rp/kWh)
    
    Args:
        df: DataFrame with energy consumption data
        energy_col: Column name for energy consumption in kWh
        tariff: Electricity tariff in Rp/kWh. Uses config if None.
        
    Returns:
        DataFrame with cost calculations:
        - Device_Name
        - Energy_kWh: Energy consumption
        - Cost_Rp: Cost in Rupiah
        - Cost_Percentage: Percentage of total cost
        
    Example:
        >>> costs = calculate_energy_costs(df_processed)
        >>> print(f"Total monthly cost: Rp {costs['Cost_Rp'].sum():,.0f}")
    """
    if tariff is None:
        tariff = config.ELECTRICITY_TARIFF
    
    logger.info(f"Calculating energy costs (Tariff: Rp {tariff:,}/kWh)...")
    
    result = df[['Device_Name', 'Device_Category']].copy()
    result['Energy_kWh'] = df[energy_col].round(2)
    result['Cost_Rp'] = (result['Energy_kWh'] * tariff).round(0)
    
    # Calculate percentage
    total_cost = result['Cost_Rp'].sum()
    result['Cost_Percentage'] = ((result['Cost_Rp'] / total_cost) * 100).round(2)
    
    # Sort by cost (descending)
    result = result.sort_values('Cost_Rp', ascending=False).reset_index(drop=True)
    
    logger.info(f"Total monthly cost: Rp {total_cost:,.0f}")
    
    return result


def get_consumption_statistics(df: pd.DataFrame,
                               value_col: str = 'Daily_Energy_kWh') -> Dict[str, float]:
    """
    Generate summary statistics for energy consumption.
    
    Args:
        df: DataFrame with energy consumption data
        value_col: Column to calculate statistics for
        
    Returns:
        Dictionary with statistics:
        - count: Number of devices
        - mean: Average consumption
        - median: Median consumption
        - std: Standard deviation
        - min: Minimum consumption
        - max: Maximum consumption
        - total: Total consumption
        - range: Max - Min
        
    Example:
        >>> stats = get_consumption_statistics(df_processed)
        >>> print(f"Average daily consumption: {stats['mean']:.2f} kWh")
    """
    logger.info(f"Calculating consumption statistics for '{value_col}'...")
    
    values = df[value_col]
    
    stats = {
        'count': int(len(values)),
        'mean': round(values.mean(), 2),
        'median': round(values.median(), 2),
        'std': round(values.std(), 2),
        'min': round(values.min(), 2),
        'max': round(values.max(), 2),
        'total': round(values.sum(), 2),
        'range': round(values.max() - values.min(), 2),
        'q1': round(values.quantile(0.25), 2),
        'q3': round(values.quantile(0.75), 2),
        'iqr': round(values.quantile(0.75) - values.quantile(0.25), 2)
    }
    
    return stats


def get_consumption_by_category(df: pd.DataFrame,
                                category_col: str = 'Device_Category',
                                energy_col: str = 'Daily_Energy_kWh') -> pd.DataFrame:
    """
    Create consumption breakdown by device category.
    
    Args:
        df: Device inventory DataFrame with energy consumption
        category_col: Column name for categories
        energy_col: Column name for energy values
        
    Returns:
        DataFrame with category breakdown:
        - Category: Device category
        - Device_Count: Number of devices in category
        - Total_Daily_kWh: Total daily consumption
        - Total_Monthly_kWh: Total monthly consumption
        - Total_Monthly_Cost: Monthly cost in Rupiah
        - Percentage: Percentage of total consumption
        
    Example:
        >>> breakdown = get_consumption_by_category(df_processed)
        >>> print(breakdown)
    """
    logger.info("Creating consumption breakdown by category...")
    
    # Group by category
    category_stats = df.groupby(category_col).agg({
        'Device_Name': 'count',
        'Quantity': 'sum',
        energy_col: 'sum',
        'Monthly_Energy_kWh': 'sum',
        'Monthly_Cost_Rp': 'sum'
    }).reset_index()
    
    # Rename columns
    category_stats.columns = [
        'Category', 'Device_Types', 'Total_Devices', 
        'Daily_kWh', 'Monthly_kWh', 'Monthly_Cost_Rp'
    ]
    
    # Calculate percentages
    total_daily = category_stats['Daily_kWh'].sum()
    category_stats['Percentage'] = (
        (category_stats['Daily_kWh'] / total_daily) * 100
    ).round(2)
    
    # Round numerical columns
    category_stats['Daily_kWh'] = category_stats['Daily_kWh'].round(2)
    category_stats['Monthly_kWh'] = category_stats['Monthly_kWh'].round(2)
    category_stats['Monthly_Cost_Rp'] = category_stats['Monthly_Cost_Rp'].round(0)
    
    # Sort by consumption (descending)
    category_stats = category_stats.sort_values('Daily_kWh', ascending=False)
    category_stats = category_stats.reset_index(drop=True)
    
    logger.info(f"Found {len(category_stats)} categories")
    
    return category_stats


def get_top_consumers(df: pd.DataFrame,
                      n: int = 10,
                      energy_col: str = 'Daily_Energy_kWh') -> pd.DataFrame:
    """
    Get the top N energy consuming devices.
    
    Args:
        df: Device inventory DataFrame
        n: Number of top consumers to return
        energy_col: Column name for energy values
        
    Returns:
        DataFrame with top N consumers
        
    Example:
        >>> top10 = get_top_consumers(df_processed, n=10)
        >>> print(top10)
    """
    logger.info(f"Getting top {n} energy consumers...")
    
    # Select relevant columns
    cols = ['Device_Name', 'Device_Category', 'Quantity', 'Power_Watt',
            'Operating_Hours_Per_Day', 'Total_Power_Watt', energy_col,
            'Monthly_Energy_kWh', 'Monthly_Cost_Rp']
    
    available_cols = [c for c in cols if c in df.columns]
    result = df[available_cols].copy()
    
    # Sort and get top N
    result = result.sort_values(energy_col, ascending=False).head(n)
    result = result.reset_index(drop=True)
    result.index = result.index + 1  # Start index from 1
    
    return result


def calculate_annual_projection(df: pd.DataFrame,
                               monthly_col: str = 'Monthly_Energy_kWh') -> Dict[str, Any]:
    """
    Calculate annual energy consumption projection.
    
    Args:
        df: Device inventory DataFrame with monthly consumption
        monthly_col: Column name for monthly energy consumption
        
    Returns:
        Dictionary with annual projections:
        - annual_kwh: Total annual consumption in kWh
        - annual_mwh: Total annual consumption in MWh
        - annual_cost_rp: Total annual cost in Rupiah
        - annual_cost_million: Total annual cost in millions of Rupiah
        - avg_monthly_kwh: Average monthly consumption
        - avg_daily_kwh: Average daily consumption
        
    Example:
        >>> projection = calculate_annual_projection(df_processed)
        >>> print(f"Annual: {projection['annual_mwh']:.2f} MWh")
    """
    logger.info("Calculating annual projection...")
    
    total_monthly = df[monthly_col].sum()
    total_annual_kwh = total_monthly * config.MONTHS_PER_YEAR
    annual_cost = total_annual_kwh * config.ELECTRICITY_TARIFF
    
    projection = {
        'annual_kwh': round(total_annual_kwh, 2),
        'annual_mwh': round(total_annual_kwh / 1000, 2),
        'annual_cost_rp': round(annual_cost, 0),
        'annual_cost_million': round(annual_cost / 1_000_000, 2),
        'avg_monthly_kwh': round(total_monthly, 2),
        'avg_daily_kwh': round(total_monthly / config.WORKING_DAYS_PER_MONTH, 2),
        'avg_hourly_kw': round((total_monthly / config.WORKING_DAYS_PER_MONTH) / 8, 2)  # Assume 8 hour day
    }
    
    logger.info(f"Annual projection: {projection['annual_mwh']} MWh")
    
    return projection


def export_summary_tables(df: pd.DataFrame,
                         output_dir: Optional[Path] = None,
                         filename: Optional[str] = None) -> Path:
    """
    Export all summary tables to Excel file.
    
    This function creates a comprehensive Excel file with multiple sheets
    containing various summary tables.
    
    Args:
        df: Device inventory DataFrame with all calculated metrics
        output_dir: Output directory path. Uses config if None.
        filename: Output filename. Uses config if None.
        
    Returns:
        Path to the exported file
        
    Example:
        >>> output = export_summary_tables(df_processed)
        >>> print(f"Tables exported to: {output}")
    """
    if output_dir is None:
        output_dir = config.TABLES_DIR
    if filename is None:
        filename = config.CONSUMPTION_SUMMARY_FILE
    
    # Ensure directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / filename
    
    logger.info(f"Exporting summary tables to: {filepath}")
    
    try:
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Sheet 1: Complete device data
            df.to_excel(writer, sheet_name='Device_Data', index=False)
            
            # Sheet 2: Category breakdown
            category_breakdown = get_consumption_by_category(df)
            category_breakdown.to_excel(writer, sheet_name='Category_Breakdown', index=False)
            
            # Sheet 3: Top consumers
            top_consumers = get_top_consumers(df, n=10)
            top_consumers.to_excel(writer, sheet_name='Top_10_Consumers', index=True)
            
            # Sheet 4: Statistics
            stats = get_consumption_statistics(df)
            stats_df = pd.DataFrame({
                'Statistic': list(stats.keys()),
                'Value': list(stats.values())
            })
            stats_df.to_excel(writer, sheet_name='Statistics', index=False)
            
            # Sheet 5: Annual projection
            projection = calculate_annual_projection(df)
            proj_df = pd.DataFrame({
                'Metric': [
                    'Annual Energy (kWh)',
                    'Annual Energy (MWh)',
                    'Annual Cost (Rp)',
                    'Annual Cost (Million Rp)',
                    'Average Monthly (kWh)',
                    'Average Daily (kWh)',
                    'Average Hourly Power (kW)'
                ],
                'Value': [
                    projection['annual_kwh'],
                    projection['annual_mwh'],
                    projection['annual_cost_rp'],
                    projection['annual_cost_million'],
                    projection['avg_monthly_kwh'],
                    projection['avg_daily_kwh'],
                    projection['avg_hourly_kw']
                ]
            })
            proj_df.to_excel(writer, sheet_name='Annual_Projection', index=False)
            
            # Sheet 6: Cost analysis
            cost_df = calculate_energy_costs(df)
            cost_df.to_excel(writer, sheet_name='Cost_Analysis', index=False)
        
        logger.info(f"Summary tables exported successfully")
        return filepath
        
    except Exception as e:
        logger.error(f"Error exporting tables: {e}")
        raise


def generate_consumption_report(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate a comprehensive consumption report.
    
    Args:
        df: Device inventory DataFrame with all metrics
        
    Returns:
        Dictionary containing comprehensive report data
        
    Example:
        >>> report = generate_consumption_report(df_processed)
        >>> print(report['summary']['total_monthly_cost'])
    """
    logger.info("Generating comprehensive consumption report...")
    
    # Get all statistics and breakdowns
    stats = get_consumption_statistics(df)
    category_breakdown = get_consumption_by_category(df)
    top_consumers = get_top_consumers(df, n=10)
    projection = calculate_annual_projection(df)
    
    report = {
        'summary': {
            'total_devices': int(df['Quantity'].sum()),
            'total_device_types': len(df),
            'total_categories': len(category_breakdown),
            'total_daily_kwh': round(df['Daily_Energy_kWh'].sum(), 2),
            'total_monthly_kwh': round(df['Monthly_Energy_kWh'].sum(), 2),
            'total_monthly_cost': round(df['Monthly_Cost_Rp'].sum(), 0),
            'tariff_per_kwh': config.ELECTRICITY_TARIFF
        },
        'statistics': stats,
        'category_breakdown': category_breakdown.to_dict('records'),
        'top_10_consumers': top_consumers.to_dict('records'),
        'annual_projection': projection,
        'cost_breakdown': {
            'by_category': category_breakdown[['Category', 'Monthly_Cost_Rp', 'Percentage']].to_dict('records')
        }
    }
    
    logger.info("Consumption report generated successfully")
    
    return report


if __name__ == "__main__":
    # Test the module
    print("Testing energy_calculator module...")
    print("-" * 50)
    
    # Create sample data for testing
    sample_data = pd.DataFrame({
        'Device_Name': ['AC Split 1 PK', 'LED Light 18W', 'Desktop PC'],
        'Device_Category': ['Cooling', 'Lighting', 'Computing'],
        'Quantity': [20, 150, 50],
        'Power_Watt': [900, 18, 300],
        'Operating_Hours_Per_Day': [8, 10, 8],
        'Total_Power_Watt': [18000, 2700, 15000],
        'Daily_Energy_kWh': [144.0, 27.0, 120.0],
        'Monthly_Energy_kWh': [3168.0, 594.0, 2640.0],
        'Monthly_Cost_Rp': [4647456.0, 871398.0, 3872880.0]
    })
    
    # Test functions
    stats = get_consumption_statistics(sample_data)
    print(f"Statistics: {stats}")
    
    category = get_consumption_by_category(sample_data)
    print(f"\nCategory breakdown:\n{category}")
    
    projection = calculate_annual_projection(sample_data)
    print(f"\nAnnual projection: {projection}")
