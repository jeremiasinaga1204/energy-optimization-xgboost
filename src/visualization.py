"""
Visualization Module for Energy Consumption Analysis

This module provides functions for creating publication-ready visualizations
of energy consumption data and model results.

Functions:
    - plot_consumption_by_device: Horizontal bar chart of top consumers
    - plot_consumption_pie_chart: Pie chart by category
    - plot_prediction_vs_actual: Prediction comparison plot
    - plot_feature_importance: Feature importance bar chart
    - plot_daily_pattern: Daily consumption pattern
    - plot_consumption_distribution: Box plot by category
    - plot_monthly_consumption: Monthly consumption bar chart
    - plot_heatmap_schedule: Operation schedule heatmap
"""

import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Optional, Tuple, List, Any
from sklearn.metrics import r2_score

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette(config.CATEGORY_PALETTE)


def _setup_figure(figsize: Optional[Tuple[int, int]] = None,
                 title: Optional[str] = None) -> Tuple[plt.Figure, plt.Axes]:
    """
    Setup figure with consistent styling.
    
    Args:
        figsize: Figure size (width, height). Uses config if None.
        title: Figure title
        
    Returns:
        Tuple of (figure, axes)
    """
    if figsize is None:
        figsize = config.FIGURE_SIZE_MEDIUM
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if title:
        ax.set_title(title, fontsize=config.FONT_SIZE_TITLE, fontweight='bold', pad=15)
    
    return fig, ax


def _save_figure(fig: plt.Figure, filename: str,
                output_dir: Optional[Path] = None,
                dpi: Optional[int] = None) -> Path:
    """
    Save figure to file with high resolution.
    
    Args:
        fig: Matplotlib figure
        filename: Output filename
        output_dir: Output directory. Uses config if None.
        dpi: Resolution. Uses config if None.
        
    Returns:
        Path to saved figure
    """
    if output_dir is None:
        output_dir = config.FIGURES_DIR
    if dpi is None:
        dpi = config.FIGURE_DPI
    
    # Ensure directory exists
    output_dir.mkdir(parents=True, exist_ok=True)
    
    filepath = output_dir / filename
    fig.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor='white')
    
    logger.info(f"Figure saved to: {filepath}")
    
    return filepath


def plot_consumption_by_device(df: pd.DataFrame,
                               n_devices: int = 10,
                               energy_col: str = 'Daily_Energy_kWh',
                               save: bool = True,
                               show: bool = False) -> Tuple[plt.Figure, Path]:
    """
    Create horizontal bar chart of top N energy consuming devices.
    
    Args:
        df: Device inventory DataFrame with energy data
        n_devices: Number of top devices to show
        energy_col: Column name for energy values
        save: Whether to save figure to file
        show: Whether to display figure
        
    Returns:
        Tuple of (figure, filepath)
        
    Example:
        >>> fig, path = plot_consumption_by_device(df_processed)
    """
    logger.info(f"Creating top {n_devices} consumers bar chart...")
    
    # Get top consumers
    top_devices = df.nlargest(n_devices, energy_col)[['Device_Name', energy_col, 'Device_Category']]
    top_devices = top_devices.sort_values(energy_col, ascending=True)
    
    fig, ax = _setup_figure(figsize=config.FIGURE_SIZE_LARGE,
                           title=f'Top {n_devices} Energy Consuming Devices (Daily kWh)')
    
    # Create color mapping by category
    categories = top_devices['Device_Category'].unique()
    colors = sns.color_palette(config.CATEGORY_PALETTE, len(categories))
    color_map = dict(zip(categories, colors))
    bar_colors = [color_map[cat] for cat in top_devices['Device_Category']]
    
    # Create horizontal bar chart
    bars = ax.barh(range(len(top_devices)), top_devices[energy_col], color=bar_colors)
    
    # Add device names as y-tick labels
    ax.set_yticks(range(len(top_devices)))
    ax.set_yticklabels(top_devices['Device_Name'], fontsize=config.FONT_SIZE_TICK)
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, top_devices[energy_col])):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{value:.1f}', va='center', fontsize=config.FONT_SIZE_TICK)
    
    # Labels
    ax.set_xlabel('Daily Energy Consumption (kWh)', fontsize=config.FONT_SIZE_LABEL)
    ax.set_ylabel('Device', fontsize=config.FONT_SIZE_LABEL)
    
    # Add legend
    legend_handles = [plt.Rectangle((0,0),1,1, color=color_map[cat]) for cat in categories]
    ax.legend(legend_handles, categories, title='Category', loc='lower right')
    
    plt.tight_layout()
    
    filepath = None
    if save:
        filepath = _save_figure(fig, config.CONSUMPTION_BY_DEVICE_FIGURE)
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig, filepath


def plot_consumption_pie_chart(df: pd.DataFrame,
                               category_col: str = 'Device_Category',
                               energy_col: str = 'Daily_Energy_kWh',
                               save: bool = True,
                               show: bool = False) -> Tuple[plt.Figure, Path]:
    """
    Create pie chart of energy consumption by category.
    
    Args:
        df: Device inventory DataFrame
        category_col: Column name for categories
        energy_col: Column name for energy values
        save: Whether to save figure
        show: Whether to display figure
        
    Returns:
        Tuple of (figure, filepath)
        
    Example:
        >>> fig, path = plot_consumption_pie_chart(df_processed)
    """
    logger.info("Creating consumption pie chart by category...")
    
    # Aggregate by category
    category_consumption = df.groupby(category_col)[energy_col].sum().sort_values(ascending=False)
    
    fig, ax = _setup_figure(figsize=(10, 8),
                           title='Energy Consumption Distribution by Category')
    
    # Create pie chart
    colors = sns.color_palette(config.CATEGORY_PALETTE, len(category_consumption))
    
    wedges, texts, autotexts = ax.pie(
        category_consumption,
        labels=category_consumption.index,
        autopct='%1.1f%%',
        colors=colors,
        startangle=90,
        explode=[0.02] * len(category_consumption),
        shadow=True
    )
    
    # Style the text
    plt.setp(autotexts, size=config.FONT_SIZE_TICK, weight='bold')
    plt.setp(texts, size=config.FONT_SIZE_TICK)
    
    # Add legend with values
    legend_labels = [f'{cat}: {val:.1f} kWh' for cat, val in category_consumption.items()]
    ax.legend(wedges, legend_labels, title='Categories', loc='center left',
              bbox_to_anchor=(1, 0, 0.5, 1))
    
    plt.tight_layout()
    
    filepath = None
    if save:
        filepath = _save_figure(fig, config.CONSUMPTION_PIE_CHART_FIGURE)
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig, filepath


def plot_prediction_vs_actual(y_actual: np.ndarray,
                              y_predicted: np.ndarray,
                              save: bool = True,
                              show: bool = False) -> Tuple[plt.Figure, Path]:
    """
    Create scatter plot comparing predicted vs actual values.
    
    Args:
        y_actual: Array of actual values
        y_predicted: Array of predicted values
        save: Whether to save figure
        show: Whether to display figure
        
    Returns:
        Tuple of (figure, filepath)
        
    Example:
        >>> fig, path = plot_prediction_vs_actual(y_test, y_pred)
    """
    logger.info("Creating prediction vs actual plot...")
    
    fig, ax = _setup_figure(figsize=config.FIGURE_SIZE_MEDIUM,
                           title='XGBoost Model: Predicted vs Actual Energy Consumption')
    
    # Scatter plot
    ax.scatter(y_actual, y_predicted, alpha=0.5, s=20, c='steelblue', edgecolors='none')
    
    # Perfect prediction line
    min_val = min(y_actual.min(), y_predicted.min())
    max_val = max(y_actual.max(), y_predicted.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    # Calculate R²
    r2 = r2_score(y_actual, y_predicted)
    
    # Add R² annotation
    ax.text(0.05, 0.95, f'R² = {r2:.4f}', transform=ax.transAxes,
            fontsize=12, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Labels
    ax.set_xlabel('Actual Energy Consumption (kWh)', fontsize=config.FONT_SIZE_LABEL)
    ax.set_ylabel('Predicted Energy Consumption (kWh)', fontsize=config.FONT_SIZE_LABEL)
    ax.legend(loc='lower right')
    
    plt.tight_layout()
    
    filepath = None
    if save:
        filepath = _save_figure(fig, config.PREDICTION_VS_ACTUAL_FIGURE)
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig, filepath


def plot_feature_importance(importance_df: pd.DataFrame,
                           n_features: int = 15,
                           save: bool = True,
                           show: bool = False) -> Tuple[plt.Figure, Path]:
    """
    Create horizontal bar chart of feature importance.
    
    Args:
        importance_df: DataFrame with Feature and Importance columns
        n_features: Number of features to display
        save: Whether to save figure
        show: Whether to display figure
        
    Returns:
        Tuple of (figure, filepath)
        
    Example:
        >>> fig, path = plot_feature_importance(feature_importance_df)
    """
    logger.info(f"Creating feature importance plot (top {n_features})...")
    
    # Get top features
    top_features = importance_df.head(n_features).copy()
    top_features = top_features.sort_values('Importance', ascending=True)
    
    fig, ax = _setup_figure(figsize=config.FIGURE_SIZE_LARGE,
                           title='XGBoost Feature Importance')
    
    # Create horizontal bar chart
    colors = sns.color_palette('viridis', len(top_features))
    bars = ax.barh(range(len(top_features)), top_features['Importance'], color=colors)
    
    # Set feature names as y-tick labels
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['Feature'], fontsize=config.FONT_SIZE_TICK)
    
    # Add value labels
    for bar, pct in zip(bars, top_features['Percentage']):
        ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height()/2,
                f'{pct:.1f}%', va='center', fontsize=config.FONT_SIZE_TICK - 1)
    
    # Labels
    ax.set_xlabel('Importance Score', fontsize=config.FONT_SIZE_LABEL)
    ax.set_ylabel('Feature', fontsize=config.FONT_SIZE_LABEL)
    
    plt.tight_layout()
    
    filepath = None
    if save:
        filepath = _save_figure(fig, config.FEATURE_IMPORTANCE_FIGURE)
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig, filepath


def plot_daily_pattern(synthetic_data: pd.DataFrame,
                       save: bool = True,
                       show: bool = False) -> Tuple[plt.Figure, Path]:
    """
    Create line plot of average consumption by hour of day.
    
    Args:
        synthetic_data: DataFrame with hourly consumption data
        save: Whether to save figure
        show: Whether to display figure
        
    Returns:
        Tuple of (figure, filepath)
        
    Example:
        >>> fig, path = plot_daily_pattern(synthetic_df)
    """
    logger.info("Creating daily consumption pattern plot...")
    
    # Calculate average consumption by hour
    hourly_avg = synthetic_data.groupby('hour')['total_energy_consumption'].agg(['mean', 'std'])
    
    fig, ax = _setup_figure(figsize=config.FIGURE_SIZE_MEDIUM,
                           title='Average Hourly Energy Consumption Pattern')
    
    # Plot mean with confidence band
    hours = hourly_avg.index
    ax.plot(hours, hourly_avg['mean'], 'b-', linewidth=2, marker='o', markersize=6,
            label='Average Consumption')
    ax.fill_between(hours, 
                    hourly_avg['mean'] - hourly_avg['std'],
                    hourly_avg['mean'] + hourly_avg['std'],
                    alpha=0.2, color='blue', label='±1 Std Dev')
    
    # Add working hours shading
    ax.axvspan(config.WORKING_HOUR_START, config.WORKING_HOUR_END, 
               alpha=0.1, color='green', label='Working Hours')
    
    # Labels
    ax.set_xlabel('Hour of Day', fontsize=config.FONT_SIZE_LABEL)
    ax.set_ylabel('Energy Consumption (kWh)', fontsize=config.FONT_SIZE_LABEL)
    ax.set_xticks(range(0, 24, 2))
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    filepath = None
    if save:
        filepath = _save_figure(fig, config.DAILY_PATTERN_FIGURE)
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig, filepath


def plot_consumption_distribution(df: pd.DataFrame,
                                  category_col: str = 'Device_Category',
                                  energy_col: str = 'Daily_Energy_kWh',
                                  save: bool = True,
                                  show: bool = False) -> Tuple[plt.Figure, Path]:
    """
    Create box plot of energy consumption distribution by category.
    
    Args:
        df: Device inventory DataFrame
        category_col: Column name for categories
        energy_col: Column name for energy values
        save: Whether to save figure
        show: Whether to display figure
        
    Returns:
        Tuple of (figure, filepath)
        
    Example:
        >>> fig, path = plot_consumption_distribution(df_processed)
    """
    logger.info("Creating consumption distribution box plot...")
    
    fig, ax = _setup_figure(figsize=config.FIGURE_SIZE_LARGE,
                           title='Energy Consumption Distribution by Device Category')
    
    # Create box plot
    categories = df.groupby(category_col)[energy_col].sum().sort_values(ascending=False).index.tolist()
    
    # For box plot, we need to create data by device within each category
    box_data = []
    labels = []
    
    for cat in categories:
        cat_data = df[df[category_col] == cat][energy_col].values
        if len(cat_data) > 0:
            box_data.append(cat_data)
            labels.append(cat)
    
    bp = ax.boxplot(box_data, labels=labels, patch_artist=True)
    
    # Color the boxes
    colors = sns.color_palette(config.CATEGORY_PALETTE, len(labels))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    # Labels
    ax.set_xlabel('Device Category', fontsize=config.FONT_SIZE_LABEL)
    ax.set_ylabel('Daily Energy Consumption (kWh)', fontsize=config.FONT_SIZE_LABEL)
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    filepath = None
    if save:
        filepath = _save_figure(fig, config.CONSUMPTION_DISTRIBUTION_FIGURE)
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig, filepath


def plot_monthly_consumption(df: pd.DataFrame,
                             save: bool = True,
                             show: bool = False) -> Tuple[plt.Figure, Path]:
    """
    Create bar chart showing monthly energy consumption projection.
    
    Args:
        df: Device inventory DataFrame with monthly consumption
        save: Whether to save figure
        show: Whether to display figure
        
    Returns:
        Tuple of (figure, filepath)
        
    Example:
        >>> fig, path = plot_monthly_consumption(df_processed)
    """
    logger.info("Creating monthly consumption bar chart...")
    
    # Calculate monthly consumption by category
    monthly_by_category = df.groupby('Device_Category')['Monthly_Energy_kWh'].sum()
    monthly_by_category = monthly_by_category.sort_values(ascending=True)
    
    fig, ax = _setup_figure(figsize=config.FIGURE_SIZE_LARGE,
                           title='Monthly Energy Consumption by Category')
    
    # Create horizontal bar chart
    colors = sns.color_palette('viridis', len(monthly_by_category))
    bars = ax.barh(range(len(monthly_by_category)), monthly_by_category.values, color=colors)
    
    # Set labels
    ax.set_yticks(range(len(monthly_by_category)))
    ax.set_yticklabels(monthly_by_category.index, fontsize=config.FONT_SIZE_TICK)
    
    # Add value labels
    for bar, val in zip(bars, monthly_by_category.values):
        ax.text(bar.get_width() + 10, bar.get_y() + bar.get_height()/2,
                f'{val:,.0f} kWh', va='center', fontsize=config.FONT_SIZE_TICK)
    
    # Labels
    ax.set_xlabel('Monthly Energy Consumption (kWh)', fontsize=config.FONT_SIZE_LABEL)
    ax.set_ylabel('Device Category', fontsize=config.FONT_SIZE_LABEL)
    
    # Add total annotation
    total = monthly_by_category.sum()
    ax.text(0.95, 0.05, f'Total: {total:,.0f} kWh/month',
            transform=ax.transAxes, fontsize=12,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    filepath = None
    if save:
        filepath = _save_figure(fig, config.MONTHLY_CONSUMPTION_FIGURE)
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig, filepath


def plot_heatmap_schedule(df: pd.DataFrame,
                          save: bool = True,
                          show: bool = False) -> Tuple[plt.Figure, Path]:
    """
    Create heatmap showing device operation schedule patterns.
    
    Args:
        df: Device inventory DataFrame
        save: Whether to save figure
        show: Whether to display figure
        
    Returns:
        Tuple of (figure, filepath)
        
    Example:
        >>> fig, path = plot_heatmap_schedule(df_processed)
    """
    logger.info("Creating device operation schedule heatmap...")
    
    # Create a simulated schedule matrix
    # Rows: Device categories, Columns: Hours of day (0-23)
    categories = df['Device_Category'].unique()
    hours = range(24)
    
    # Create schedule matrix based on typical patterns
    schedule_matrix = []
    
    for cat in categories:
        cat_data = df[df['Device_Category'] == cat]
        avg_hours = cat_data['Operating_Hours_Per_Day'].mean()
        total_power = cat_data['Total_Power_Watt'].sum()
        
        # Create hourly pattern
        hourly_pattern = []
        for hour in hours:
            if cat in ['Security', 'IT_Infrastructure']:
                # 24/7 operation
                intensity = total_power * 0.8
            elif cat in ['Cooling', 'Computing', 'Office']:
                # Working hours operation
                if config.WORKING_HOUR_START <= hour < config.WORKING_HOUR_END:
                    intensity = total_power
                else:
                    intensity = total_power * 0.1
            elif cat == 'Lighting':
                # Extended hours
                if 7 <= hour < 19:
                    intensity = total_power
                else:
                    intensity = total_power * 0.2
            else:
                # Standard working hours
                if config.WORKING_HOUR_START <= hour < config.WORKING_HOUR_END:
                    intensity = total_power
                else:
                    intensity = total_power * 0.05
            
            hourly_pattern.append(intensity / 1000)  # Convert to kW
        
        schedule_matrix.append(hourly_pattern)
    
    # Create heatmap
    fig, ax = plt.subplots(figsize=(14, 8))
    
    schedule_df = pd.DataFrame(schedule_matrix, index=categories, columns=hours)
    
    sns.heatmap(schedule_df, cmap='YlOrRd', annot=False, fmt='.1f',
                ax=ax, cbar_kws={'label': 'Power Consumption (kW)'})
    
    ax.set_title('Device Operation Schedule Heatmap (Power by Hour)',
                fontsize=config.FONT_SIZE_TITLE, fontweight='bold', pad=15)
    ax.set_xlabel('Hour of Day', fontsize=config.FONT_SIZE_LABEL)
    ax.set_ylabel('Device Category', fontsize=config.FONT_SIZE_LABEL)
    
    # Add working hours markers
    ax.axvline(x=config.WORKING_HOUR_START, color='blue', linestyle='--', linewidth=2, alpha=0.5)
    ax.axvline(x=config.WORKING_HOUR_END, color='blue', linestyle='--', linewidth=2, alpha=0.5)
    
    plt.tight_layout()
    
    filepath = None
    if save:
        filepath = _save_figure(fig, config.HEATMAP_SCHEDULE_FIGURE)
    
    if show:
        plt.show()
    else:
        plt.close()
    
    return fig, filepath


def generate_all_visualizations(df_devices: pd.DataFrame,
                               model_results: Optional[dict] = None,
                               save: bool = True,
                               show: bool = False) -> List[Path]:
    """
    Generate all visualizations at once.
    
    Args:
        df_devices: Device inventory DataFrame with metrics
        model_results: Dictionary from xgboost_model.train_and_evaluate()
        save: Whether to save figures
        show: Whether to display figures
        
    Returns:
        List of paths to saved figures
        
    Example:
        >>> paths = generate_all_visualizations(df, model_results)
    """
    logger.info("Generating all visualizations...")
    
    saved_paths = []
    
    # Device consumption charts
    _, path = plot_consumption_by_device(df_devices, save=save, show=show)
    if path:
        saved_paths.append(path)
    
    _, path = plot_consumption_pie_chart(df_devices, save=save, show=show)
    if path:
        saved_paths.append(path)
    
    _, path = plot_consumption_distribution(df_devices, save=save, show=show)
    if path:
        saved_paths.append(path)
    
    _, path = plot_monthly_consumption(df_devices, save=save, show=show)
    if path:
        saved_paths.append(path)
    
    _, path = plot_heatmap_schedule(df_devices, save=save, show=show)
    if path:
        saved_paths.append(path)
    
    # Model visualizations (if results provided)
    if model_results:
        _, path = plot_prediction_vs_actual(
            model_results['actuals'],
            model_results['predictions'],
            save=save, show=show
        )
        if path:
            saved_paths.append(path)
        
        _, path = plot_feature_importance(
            model_results['feature_importance'],
            save=save, show=show
        )
        if path:
            saved_paths.append(path)
        
        _, path = plot_daily_pattern(
            model_results['synthetic_data'],
            save=save, show=show
        )
        if path:
            saved_paths.append(path)
    
    logger.info(f"Generated {len(saved_paths)} visualizations")
    
    return saved_paths


if __name__ == "__main__":
    # Test with sample data
    print("Testing visualization module...")
    print("-" * 50)
    
    # Create sample data
    sample_devices = pd.DataFrame({
        'Device_Name': ['AC Split 1 PK', 'LED Light 18W', 'Desktop PC', 'Server Rack', 'Projector'],
        'Device_Category': ['Cooling', 'Lighting', 'Computing', 'IT_Infrastructure', 'Presentation'],
        'Quantity': [20, 150, 50, 3, 10],
        'Power_Watt': [900, 18, 300, 1500, 250],
        'Operating_Hours_Per_Day': [8, 10, 8, 24, 6],
        'Total_Power_Watt': [18000, 2700, 15000, 4500, 2500],
        'Daily_Energy_kWh': [144.0, 27.0, 120.0, 108.0, 15.0],
        'Monthly_Energy_kWh': [3168.0, 594.0, 2640.0, 2376.0, 330.0]
    })
    
    print("Generating sample visualizations...")
    
    # Generate consumption charts
    plot_consumption_by_device(sample_devices, save=True, show=False)
    plot_consumption_pie_chart(sample_devices, save=True, show=False)
    
    print("\nVisualization tests complete!")
    print(f"Check {config.FIGURES_DIR} for output files")
