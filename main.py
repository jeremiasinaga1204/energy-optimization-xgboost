#!/usr/bin/env python3
"""
Main Pipeline Script for Energy Consumption Analysis and Optimization

This script runs the complete analysis pipeline including:
1. Data loading and preprocessing
2. Energy consumption calculations
3. XGBoost model training and evaluation
4. Visualization generation
5. Optimization recommendations
6. Results export

Usage:
    python main.py

Author: Energy Optimization Team
"""

import sys
import logging
from pathlib import Path
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)


def main():
    """
    Run the complete energy analysis pipeline.
    """
    print("\n" + "=" * 70)
    print("ENERGY CONSUMPTION ANALYSIS AND OPTIMIZATION SYSTEM")
    print("BTI Campus Building - Universitas Pertahanan RI")
    print("Using XGBoost Algorithm")
    print("=" * 70)
    print(f"\nAnalysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Import modules
    try:
        import config
        from src import data_input, energy_calculator, xgboost_model, visualization, optimization
        logger.info("Modules imported successfully")
    except ImportError as e:
        logger.error(f"Failed to import modules: {e}")
        logger.error("Please ensure all dependencies are installed: pip install -r requirements.txt")
        sys.exit(1)
    
    # Create output directories
    config.create_directories()
    logger.info("Output directories created")
    
    # Step 1: Data Loading
    print("\n" + "-" * 50)
    print("STEP 1: Loading Device Inventory Data")
    print("-" * 50)
    
    try:
        df_devices, summary = data_input.load_and_process_data()
        print(f"✓ Loaded {summary['total_device_types']} device types")
        print(f"✓ Total devices: {summary['total_devices']}")
        print(f"✓ Categories: {', '.join(summary['categories'])}")
    except FileNotFoundError:
        logger.error("Device inventory file not found!")
        logger.error(f"Please ensure the file exists at: {config.RAW_DATA_DIR / config.DEVICE_INVENTORY_FILE}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        sys.exit(1)
    
    # Step 2: Energy Consumption Analysis
    print("\n" + "-" * 50)
    print("STEP 2: Calculating Energy Consumption")
    print("-" * 50)
    
    total_daily_kwh = df_devices['Daily_Energy_kWh'].sum()
    total_monthly_kwh = df_devices['Monthly_Energy_kWh'].sum()
    total_annual_kwh = total_monthly_kwh * 12
    annual_cost = total_annual_kwh * config.ELECTRICITY_TARIFF
    
    print(f"✓ Daily consumption: {total_daily_kwh:,.2f} kWh")
    print(f"✓ Monthly consumption: {total_monthly_kwh:,.2f} kWh")
    print(f"✓ Annual projection: {total_annual_kwh:,.2f} kWh ({total_annual_kwh/1000:.2f} MWh)")
    print(f"✓ Annual cost: Rp {annual_cost:,.0f}")
    
    # Export consumption summary
    summary_path = energy_calculator.export_summary_tables(df_devices)
    print(f"✓ Summary tables exported to: {summary_path}")
    
    # Step 3: XGBoost Modeling
    print("\n" + "-" * 50)
    print("STEP 3: Training XGBoost Prediction Model")
    print("-" * 50)
    
    print("Generating synthetic training data...")
    results = xgboost_model.train_and_evaluate(
        df_devices, 
        num_days=config.SIMULATION_DAYS,
        save_trained_model=True
    )
    
    metrics = results['metrics']
    print(f"\n✓ Model Training Complete!")
    print(f"  - R² Score: {metrics['r2']:.4f}")
    print(f"  - RMSE: {metrics['rmse']:.4f} kWh")
    print(f"  - MAE: {metrics['mae']:.4f} kWh")
    print(f"  - MAPE: {metrics['mape']:.2f}%")
    
    if metrics['r2'] >= 0.85:
        print("✓ Model achieves target R² ≥ 0.85")
    else:
        print(f"⚠ Model R² ({metrics['r2']:.4f}) is below target (0.85)")
    
    # Step 4: Generate Visualizations
    print("\n" + "-" * 50)
    print("STEP 4: Generating Visualizations")
    print("-" * 50)
    
    try:
        saved_figures = visualization.generate_all_visualizations(
            df_devices, 
            model_results=results,
            save=True, 
            show=False
        )
        print(f"✓ Generated {len(saved_figures)} figures")
        for fig_path in saved_figures:
            print(f"  - {fig_path.name}")
    except Exception as e:
        logger.warning(f"Some visualizations failed: {e}")
    
    # Step 5: Optimization Recommendations
    print("\n" + "-" * 50)
    print("STEP 5: Generating Optimization Recommendations")
    print("-" * 50)
    
    opt_report = optimization.generate_optimization_report(df_devices)
    total_savings = opt_report['total_savings']
    
    print(f"\n✓ Optimization Analysis Complete!")
    print(f"  - Current monthly consumption: {total_savings['current_monthly_consumption_kwh']:,.2f} kWh")
    print(f"  - Potential monthly savings: {total_savings['potential_monthly_savings_kwh']:,.2f} kWh")
    print(f"  - Potential annual savings: Rp {total_savings['potential_annual_savings_rp']:,.0f}")
    print(f"  - Overall savings potential: {total_savings['overall_savings_percentage']:.2f}%")
    print(f"  - Average ROI: {total_savings['average_roi_months']:.1f} months")
    print(f"✓ Recommendations exported to: {opt_report['export_path']}")
    
    # Step 6: Export Processed Data
    print("\n" + "-" * 50)
    print("STEP 6: Exporting Processed Data")
    print("-" * 50)
    
    processed_path = data_input.export_processed_data(df_devices)
    print(f"✓ Processed data exported to: {processed_path}")
    
    # Final Summary
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    
    print(f"\n📊 RESULTS SUMMARY:")
    print(f"  - Total monthly consumption: {total_monthly_kwh:,.2f} kWh")
    print(f"  - Total monthly cost: Rp {total_monthly_kwh * config.ELECTRICITY_TARIFF:,.0f}")
    print(f"  - XGBoost model R²: {metrics['r2']:.4f}")
    print(f"  - Potential savings: {total_savings['overall_savings_percentage']:.2f}%")
    
    print(f"\n📁 OUTPUT FILES:")
    print(f"  - Figures: {config.FIGURES_DIR}")
    print(f"  - Tables: {config.TABLES_DIR}")
    print(f"  - Processed data: {config.PROCESSED_DATA_DIR}")
    print(f"  - Model: {config.OUTPUTS_DIR / config.MODEL_FILE}")
    
    print(f"\n✓ All results ready for journal publication!")
    print(f"  Analysis completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70 + "\n")
    
    return {
        'devices': df_devices,
        'summary': summary,
        'model_results': results,
        'optimization_report': opt_report
    }


if __name__ == "__main__":
    try:
        results = main()
    except KeyboardInterrupt:
        print("\n\nAnalysis interrupted by user.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
