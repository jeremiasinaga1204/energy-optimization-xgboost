"""
Optimization Module for Energy Consumption Analysis

This module provides functions for identifying energy optimization opportunities
and generating recommendations with ROI calculations.

Functions:
    - identify_top_consumers: Get top energy consuming devices
    - calculate_savings_scenarios: Calculate potential savings for various scenarios
    - generate_recommendations: Generate optimization recommendations
    - calculate_roi: Calculate return on investment
    - export_recommendations: Export recommendations to Excel
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Estimated costs for various optimization measures (in Rupiah)
ESTIMATED_COSTS = {
    'inverter_ac_unit': 8_000_000,      # Cost per unit to replace with inverter AC
    'led_bulb': 50_000,                 # Cost per LED bulb replacement
    'power_strip': 150_000,             # Smart power strip per workstation
    'occupancy_sensor': 500_000,        # Occupancy sensor per room
    'timer_switch': 200_000,            # Timer switch per circuit
    'voltage_optimizer': 5_000_000,     # Voltage optimizer per building
    'power_management_software': 100_000  # Per computer license
}


def identify_top_consumers(df: pd.DataFrame,
                          n: int = 10,
                          energy_col: str = 'Monthly_Energy_kWh') -> pd.DataFrame:
    """
    Identify the top N energy consuming devices.
    
    Args:
        df: Device inventory DataFrame with energy metrics
        n: Number of top consumers to identify
        energy_col: Column name for energy values
        
    Returns:
        DataFrame with top N consumers and their metrics
        
    Example:
        >>> top_devices = identify_top_consumers(df_processed, n=10)
    """
    logger.info(f"Identifying top {n} energy consumers...")
    
    # Select relevant columns
    columns = ['Device_Name', 'Device_Category', 'Quantity', 'Power_Watt',
               'Operating_Hours_Per_Day', 'Daily_Energy_kWh', 
               'Monthly_Energy_kWh', 'Monthly_Cost_Rp']
    
    available_cols = [c for c in columns if c in df.columns]
    result = df[available_cols].copy()
    
    # Sort by energy consumption and get top N
    result = result.sort_values(energy_col, ascending=False).head(n)
    result = result.reset_index(drop=True)
    result.index = result.index + 1  # Start index from 1
    
    # Calculate cumulative percentage
    total_energy = df[energy_col].sum()
    result['Percentage_of_Total'] = (result[energy_col] / total_energy * 100).round(2)
    result['Cumulative_Percentage'] = result['Percentage_of_Total'].cumsum().round(2)
    
    logger.info(f"Top {n} consumers account for {result['Cumulative_Percentage'].iloc[-1]:.1f}% of total consumption")
    
    return result


def calculate_scenario_savings(df: pd.DataFrame,
                               scenario_name: str,
                               scenario_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Calculate potential savings for a specific optimization scenario.
    
    Args:
        df: Device inventory DataFrame
        scenario_name: Name of the scenario
        scenario_config: Scenario configuration from config
        
    Returns:
        Dictionary with savings calculations
        
    Example:
        >>> savings = calculate_scenario_savings(df, 'inverter_ac', config)
    """
    logger.info(f"Calculating savings for scenario: {scenario_name}")
    
    savings_pct = scenario_config['savings_percentage']
    applicable_categories = scenario_config['applicable_categories']
    applicable_devices = scenario_config.get('applicable_devices', [])
    
    # Filter applicable devices
    if 'all' in applicable_categories:
        applicable_df = df.copy()
    else:
        applicable_df = df[df['Device_Category'].isin(applicable_categories)].copy()
    
    # Further filter by device name if specified
    if applicable_devices:
        device_filter = applicable_df['Device_Name'].str.contains('|'.join(applicable_devices), case=False, na=False)
        applicable_df = applicable_df[device_filter]
    
    if len(applicable_df) == 0:
        return {
            'scenario_name': scenario_name,
            'applicable_devices': 0,
            'current_monthly_kwh': 0,
            'potential_savings_kwh': 0,
            'potential_savings_rp': 0,
            'savings_percentage': savings_pct * 100
        }
    
    current_monthly = applicable_df['Monthly_Energy_kWh'].sum()
    potential_savings_kwh = current_monthly * savings_pct
    potential_savings_rp = potential_savings_kwh * config.ELECTRICITY_TARIFF
    
    return {
        'scenario_name': scenario_name,
        'scenario_description': scenario_config['name'],
        'applicable_devices': len(applicable_df),
        'total_quantity': int(applicable_df['Quantity'].sum()),
        'current_monthly_kwh': round(current_monthly, 2),
        'potential_savings_kwh': round(potential_savings_kwh, 2),
        'potential_savings_rp': round(potential_savings_rp, 0),
        'savings_percentage': savings_pct * 100,
        'annual_savings_kwh': round(potential_savings_kwh * 12, 2),
        'annual_savings_rp': round(potential_savings_rp * 12, 0)
    }


def calculate_all_scenarios(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate savings for all optimization scenarios.
    
    Args:
        df: Device inventory DataFrame
        
    Returns:
        DataFrame with all scenario calculations
        
    Example:
        >>> scenarios_df = calculate_all_scenarios(df_processed)
    """
    logger.info("Calculating all optimization scenarios...")
    
    results = []
    
    for scenario_name, scenario_config in config.OPTIMIZATION_SCENARIOS.items():
        savings = calculate_scenario_savings(df, scenario_name, scenario_config)
        results.append(savings)
    
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('annual_savings_rp', ascending=False)
    results_df = results_df.reset_index(drop=True)
    
    # Calculate total potential savings
    total_monthly_savings = results_df['potential_savings_kwh'].sum()
    total_annual_savings = results_df['annual_savings_rp'].sum()
    
    logger.info(f"Total potential annual savings: Rp {total_annual_savings:,.0f}")
    
    return results_df


def generate_recommendations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate detailed optimization recommendations for each device/category.
    
    Args:
        df: Device inventory DataFrame
        
    Returns:
        DataFrame with recommendations including:
        - Device/Category
        - Current consumption
        - Recommendation
        - Potential savings
        - Estimated cost
        - ROI months
        
    Example:
        >>> recommendations = generate_recommendations(df_processed)
    """
    logger.info("Generating optimization recommendations...")
    
    recommendations = []
    
    # Sort by energy consumption
    df_sorted = df.sort_values('Monthly_Energy_kWh', ascending=False)
    
    for _, device in df_sorted.iterrows():
        device_name = device['Device_Name']
        category = device['Device_Category']
        quantity = device['Quantity']
        current_kwh = device['Monthly_Energy_kWh']
        current_cost = device['Monthly_Cost_Rp']
        
        rec = {
            'Device': device_name,
            'Category': category,
            'Quantity': quantity,
            'Current_Monthly_kWh': round(current_kwh, 2),
            'Current_Monthly_Cost_Rp': round(current_cost, 0)
        }
        
        # Generate recommendation based on category
        if 'AC' in device_name.upper() and category == 'Cooling':
            # AC optimization - inverter replacement
            savings_pct = 0.30
            savings_kwh = current_kwh * savings_pct
            savings_rp = savings_kwh * config.ELECTRICITY_TARIFF
            estimated_cost = quantity * ESTIMATED_COSTS['inverter_ac_unit']
            
            rec.update({
                'Recommendation': 'Replace with inverter AC units for 30% efficiency gain',
                'Potential_Savings_kWh': round(savings_kwh, 2),
                'Savings_Percentage': 30,
                'Monthly_Savings_Rp': round(savings_rp, 0),
                'Estimated_Cost_Rp': estimated_cost,
                'ROI_Months': round(estimated_cost / savings_rp, 1) if savings_rp > 0 else 0
            })
            
        elif category == 'Lighting':
            # Lighting optimization
            savings_pct = 0.60 if 'LED' not in device_name.upper() else 0.15
            savings_kwh = current_kwh * savings_pct
            savings_rp = savings_kwh * config.ELECTRICITY_TARIFF
            estimated_cost = quantity * ESTIMATED_COSTS['led_bulb'] + ESTIMATED_COSTS['occupancy_sensor'] * 5
            
            rec.update({
                'Recommendation': 'Install occupancy sensors and use T5/T8 LED tubes' if 'LED' in device_name.upper() else 'Replace with LED and add occupancy sensors',
                'Potential_Savings_kWh': round(savings_kwh, 2),
                'Savings_Percentage': savings_pct * 100,
                'Monthly_Savings_Rp': round(savings_rp, 0),
                'Estimated_Cost_Rp': estimated_cost,
                'ROI_Months': round(estimated_cost / savings_rp, 1) if savings_rp > 0 else 0
            })
            
        elif category == 'Computing':
            # Computing optimization - power management
            savings_pct = 0.20
            savings_kwh = current_kwh * savings_pct
            savings_rp = savings_kwh * config.ELECTRICITY_TARIFF
            estimated_cost = quantity * ESTIMATED_COSTS['power_management_software']
            
            rec.update({
                'Recommendation': 'Enable power management and scheduled shutdown',
                'Potential_Savings_kWh': round(savings_kwh, 2),
                'Savings_Percentage': 20,
                'Monthly_Savings_Rp': round(savings_rp, 0),
                'Estimated_Cost_Rp': estimated_cost,
                'ROI_Months': round(estimated_cost / savings_rp, 1) if savings_rp > 0 else 0
            })
            
        elif category == 'IT_Infrastructure':
            # IT infrastructure - virtualization and scheduling
            savings_pct = 0.15
            savings_kwh = current_kwh * savings_pct
            savings_rp = savings_kwh * config.ELECTRICITY_TARIFF
            estimated_cost = ESTIMATED_COSTS['timer_switch'] * 2
            
            rec.update({
                'Recommendation': 'Optimize cooling, use timer for non-critical systems',
                'Potential_Savings_kWh': round(savings_kwh, 2),
                'Savings_Percentage': 15,
                'Monthly_Savings_Rp': round(savings_rp, 0),
                'Estimated_Cost_Rp': estimated_cost,
                'ROI_Months': round(estimated_cost / savings_rp, 1) if savings_rp > 0 else 0
            })
            
        elif 'Fan' in device_name:
            # Fan optimization
            savings_pct = 0.25
            savings_kwh = current_kwh * savings_pct
            savings_rp = savings_kwh * config.ELECTRICITY_TARIFF
            estimated_cost = quantity * ESTIMATED_COSTS['timer_switch'] // 4
            
            rec.update({
                'Recommendation': 'Use timer controls and reduce usage with improved ventilation',
                'Potential_Savings_kWh': round(savings_kwh, 2),
                'Savings_Percentage': 25,
                'Monthly_Savings_Rp': round(savings_rp, 0),
                'Estimated_Cost_Rp': estimated_cost,
                'ROI_Months': round(estimated_cost / savings_rp, 1) if savings_rp > 0 else 0
            })
            
        else:
            # General optimization - reduce operating hours
            savings_pct = 0.15
            savings_kwh = current_kwh * savings_pct
            savings_rp = savings_kwh * config.ELECTRICITY_TARIFF
            estimated_cost = ESTIMATED_COSTS['timer_switch']
            
            rec.update({
                'Recommendation': 'Reduce operating hours by 15% through scheduling',
                'Potential_Savings_kWh': round(savings_kwh, 2),
                'Savings_Percentage': 15,
                'Monthly_Savings_Rp': round(savings_rp, 0),
                'Estimated_Cost_Rp': estimated_cost,
                'ROI_Months': round(estimated_cost / savings_rp, 1) if savings_rp > 0 else 0
            })
        
        recommendations.append(rec)
    
    recommendations_df = pd.DataFrame(recommendations)
    
    # Sort by potential savings
    recommendations_df = recommendations_df.sort_values('Potential_Savings_kWh', ascending=False)
    recommendations_df = recommendations_df.reset_index(drop=True)
    recommendations_df.index = recommendations_df.index + 1
    
    logger.info(f"Generated {len(recommendations_df)} recommendations")
    
    return recommendations_df


def calculate_total_savings(recommendations_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate total potential savings from all recommendations.
    
    Args:
        recommendations_df: DataFrame with recommendations
        
    Returns:
        Dictionary with total savings calculations
        
    Example:
        >>> totals = calculate_total_savings(recommendations_df)
    """
    logger.info("Calculating total potential savings...")
    
    total_current_kwh = recommendations_df['Current_Monthly_kWh'].sum()
    total_savings_kwh = recommendations_df['Potential_Savings_kWh'].sum()
    total_savings_rp = recommendations_df['Monthly_Savings_Rp'].sum()
    total_cost = recommendations_df['Estimated_Cost_Rp'].sum()
    
    totals = {
        'current_monthly_consumption_kwh': round(total_current_kwh, 2),
        'current_monthly_cost_rp': round(total_current_kwh * config.ELECTRICITY_TARIFF, 0),
        'potential_monthly_savings_kwh': round(total_savings_kwh, 2),
        'potential_monthly_savings_rp': round(total_savings_rp, 0),
        'potential_annual_savings_kwh': round(total_savings_kwh * 12, 2),
        'potential_annual_savings_rp': round(total_savings_rp * 12, 0),
        'total_implementation_cost_rp': round(total_cost, 0),
        'overall_savings_percentage': round(total_savings_kwh / total_current_kwh * 100, 2) if total_current_kwh > 0 else 0,
        'average_roi_months': round(total_cost / total_savings_rp, 1) if total_savings_rp > 0 else 0,
        'co2_reduction_kg_annual': round(total_savings_kwh * 12 * 0.85, 2)  # ~0.85 kg CO2 per kWh in Indonesia
    }
    
    logger.info(f"Total potential annual savings: Rp {totals['potential_annual_savings_rp']:,.0f}")
    logger.info(f"Average ROI: {totals['average_roi_months']:.1f} months")
    
    return totals


def get_priority_recommendations(recommendations_df: pd.DataFrame,
                                 n: int = 5,
                                 max_roi_months: float = 24) -> pd.DataFrame:
    """
    Get priority recommendations based on ROI and savings potential.
    
    Args:
        recommendations_df: DataFrame with recommendations
        n: Number of recommendations to return
        max_roi_months: Maximum ROI period to consider (months)
        
    Returns:
        DataFrame with priority recommendations
        
    Example:
        >>> priority = get_priority_recommendations(recommendations_df, n=5)
    """
    logger.info(f"Getting top {n} priority recommendations (ROI <= {max_roi_months} months)...")
    
    # Filter by ROI
    filtered = recommendations_df[recommendations_df['ROI_Months'] <= max_roi_months].copy()
    
    # Sort by savings potential
    filtered = filtered.sort_values('Monthly_Savings_Rp', ascending=False)
    
    # Get top N
    result = filtered.head(n)
    result = result.reset_index(drop=True)
    result.index = result.index + 1
    
    return result


def export_recommendations(recommendations_df: pd.DataFrame,
                          total_savings: Dict[str, Any],
                          scenario_df: Optional[pd.DataFrame] = None,
                          output_dir: Optional[Path] = None,
                          filename: Optional[str] = None) -> Path:
    """
    Export optimization recommendations to Excel file.
    
    Args:
        recommendations_df: DataFrame with recommendations
        total_savings: Dictionary with total savings calculations
        scenario_df: Optional DataFrame with scenario analysis
        output_dir: Output directory. Uses config if None.
        filename: Output filename. Uses config if None.
        
    Returns:
        Path to exported file
        
    Example:
        >>> path = export_recommendations(recommendations, totals)
    """
    if output_dir is None:
        output_dir = config.TABLES_DIR
    if filename is None:
        filename = config.OPTIMIZATION_RECOMMENDATIONS_FILE
    
    # Ensure directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / filename
    
    logger.info(f"Exporting recommendations to: {filepath}")
    
    try:
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Sheet 1: Detailed recommendations
            recommendations_df.to_excel(writer, sheet_name='Recommendations', index=True)
            
            # Sheet 2: Priority recommendations
            priority = get_priority_recommendations(recommendations_df)
            priority.to_excel(writer, sheet_name='Priority_Actions', index=True)
            
            # Sheet 3: Savings summary
            summary_df = pd.DataFrame({
                'Metric': [
                    'Current Monthly Consumption (kWh)',
                    'Current Monthly Cost (Rp)',
                    'Potential Monthly Savings (kWh)',
                    'Potential Monthly Savings (Rp)',
                    'Potential Annual Savings (kWh)',
                    'Potential Annual Savings (Rp)',
                    'Total Implementation Cost (Rp)',
                    'Overall Savings Percentage (%)',
                    'Average ROI Period (Months)',
                    'CO2 Reduction Annual (kg)'
                ],
                'Value': [
                    total_savings['current_monthly_consumption_kwh'],
                    f"Rp {total_savings['current_monthly_cost_rp']:,.0f}",
                    total_savings['potential_monthly_savings_kwh'],
                    f"Rp {total_savings['potential_monthly_savings_rp']:,.0f}",
                    total_savings['potential_annual_savings_kwh'],
                    f"Rp {total_savings['potential_annual_savings_rp']:,.0f}",
                    f"Rp {total_savings['total_implementation_cost_rp']:,.0f}",
                    f"{total_savings['overall_savings_percentage']:.2f}%",
                    f"{total_savings['average_roi_months']:.1f}",
                    f"{total_savings['co2_reduction_kg_annual']:,.2f}"
                ]
            })
            summary_df.to_excel(writer, sheet_name='Savings_Summary', index=False)
            
            # Sheet 4: Scenario analysis (if provided)
            if scenario_df is not None:
                scenario_df.to_excel(writer, sheet_name='Scenario_Analysis', index=False)
            
            # Sheet 5: By category summary
            category_summary = recommendations_df.groupby('Category').agg({
                'Current_Monthly_kWh': 'sum',
                'Potential_Savings_kWh': 'sum',
                'Monthly_Savings_Rp': 'sum',
                'Estimated_Cost_Rp': 'sum'
            }).round(2)
            category_summary['ROI_Months'] = (
                category_summary['Estimated_Cost_Rp'] / category_summary['Monthly_Savings_Rp']
            ).round(1)
            category_summary = category_summary.sort_values('Monthly_Savings_Rp', ascending=False)
            category_summary.to_excel(writer, sheet_name='By_Category')
        
        logger.info("Recommendations exported successfully")
        return filepath
        
    except Exception as e:
        logger.error(f"Error exporting recommendations: {e}")
        raise


def generate_optimization_report(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate a comprehensive optimization report.
    
    This is the main entry point for optimization analysis.
    
    Args:
        df: Device inventory DataFrame with metrics
        
    Returns:
        Dictionary containing all optimization analysis results
        
    Example:
        >>> report = generate_optimization_report(df_processed)
    """
    logger.info("Generating comprehensive optimization report...")
    
    # Generate recommendations
    recommendations = generate_recommendations(df)
    
    # Calculate totals
    totals = calculate_total_savings(recommendations)
    
    # Calculate scenarios
    scenarios = calculate_all_scenarios(df)
    
    # Get priority actions
    priority = get_priority_recommendations(recommendations)
    
    # Identify top consumers
    top_consumers = identify_top_consumers(df)
    
    # Export to Excel
    export_path = export_recommendations(recommendations, totals, scenarios)
    
    report = {
        'recommendations': recommendations,
        'total_savings': totals,
        'scenarios': scenarios,
        'priority_actions': priority,
        'top_consumers': top_consumers,
        'export_path': export_path
    }
    
    logger.info("Optimization report generated successfully")
    
    return report


if __name__ == "__main__":
    # Test the module
    print("Testing optimization module...")
    print("-" * 50)
    
    # Create sample data
    sample_devices = pd.DataFrame({
        'Device_Name': ['AC Split 1 PK', 'AC Split 1.5 PK', 'LED Light 18W', 'Desktop PC', 'Server Rack'],
        'Device_Category': ['Cooling', 'Cooling', 'Lighting', 'Computing', 'IT_Infrastructure'],
        'Quantity': [20, 15, 150, 50, 3],
        'Power_Watt': [900, 1200, 18, 300, 1500],
        'Operating_Hours_Per_Day': [8, 8, 10, 8, 24],
        'Total_Power_Watt': [18000, 18000, 2700, 15000, 4500],
        'Daily_Energy_kWh': [144.0, 144.0, 27.0, 120.0, 108.0],
        'Monthly_Energy_kWh': [3168.0, 3168.0, 594.0, 2640.0, 2376.0],
        'Monthly_Cost_Rp': [4647456.0, 4647456.0, 871398.0, 3872880.0, 3485592.0]
    })
    
    # Generate report
    report = generate_optimization_report(sample_devices)
    
    print("\nTotal Savings Summary:")
    for key, value in report['total_savings'].items():
        print(f"  {key}: {value}")
    
    print(f"\nRecommendations exported to: {report['export_path']}")
