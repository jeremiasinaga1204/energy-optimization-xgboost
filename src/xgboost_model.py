"""
XGBoost Model Module for Energy Consumption Prediction

This module implements XGBoost regression for predicting energy consumption
based on various features like time, temperature, and occupancy.

Functions:
    - generate_synthetic_data: Create training data based on device inventory
    - create_features: Feature engineering functions
    - train_model: Train XGBoost regressor
    - evaluate_model: Calculate evaluation metrics
    - get_feature_importance: Extract feature importance
    - save_model: Save trained model to file
"""

import logging
import pickle
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Any, Tuple, List
from datetime import datetime, timedelta
import warnings

# Suppress specific XGBoost and sklearn warnings that are not actionable
warnings.filterwarnings('ignore', category=FutureWarning, module='xgboost')
warnings.filterwarnings('ignore', category=UserWarning, module='xgboost')

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import xgboost as xgb

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def generate_synthetic_data(df: pd.DataFrame,
                           num_days: Optional[int] = None,
                           random_state: Optional[int] = None) -> pd.DataFrame:
    """
    Generate synthetic training data based on device inventory.
    
    This function creates hourly energy consumption data for simulation,
    incorporating various factors like time of day, temperature, occupancy,
    and day type (weekend/weekday).
    
    Args:
        df: Device inventory DataFrame with energy consumption data
        num_days: Number of days to simulate. Uses config if None.
        random_state: Random seed for reproducibility. Uses config if None.
        
    Returns:
        DataFrame with synthetic data containing:
        - datetime: Timestamp for each hour
        - hour: Hour of day (0-23)
        - day_of_week: Day of week (0=Monday, 6=Sunday)
        - month: Month (1-12)
        - temperature: Simulated temperature (°C)
        - occupancy_rate: Simulated occupancy (0-100%)
        - is_weekend: Binary flag for weekend
        - is_working_hour: Binary flag for working hours
        - active_devices_count: Number of active devices
        - total_energy_consumption: Target variable (kWh)
        
    Example:
        >>> synthetic = generate_synthetic_data(df_processed, num_days=90)
        >>> print(f"Generated {len(synthetic)} hourly records")
    """
    if num_days is None:
        num_days = config.SIMULATION_DAYS
    if random_state is None:
        random_state = config.RANDOM_STATE
    
    np.random.seed(random_state)
    logger.info(f"Generating synthetic data for {num_days} days...")
    
    # Calculate base consumption from device inventory
    total_daily_kwh = df['Daily_Energy_kWh'].sum()
    base_hourly_kwh = total_daily_kwh / 8  # Assuming 8 working hours
    total_devices = df['Quantity'].sum()
    
    # Generate datetime range
    start_date = datetime(2024, 1, 1)
    hours = num_days * 24
    date_range = [start_date + timedelta(hours=i) for i in range(hours)]
    
    data = []
    
    for dt in date_range:
        hour = dt.hour
        day_of_week = dt.weekday()  # 0=Monday, 6=Sunday
        month = dt.month
        is_weekend = 1 if day_of_week >= 5 else 0
        is_working_hour = 1 if (config.WORKING_HOUR_START <= hour < config.WORKING_HOUR_END and not is_weekend) else 0
        
        # Simulate temperature (higher during midday)
        # Base temperature varies by time of day
        base_temp = config.MIN_TEMPERATURE + (config.MAX_TEMPERATURE - config.MIN_TEMPERATURE) * 0.5
        temp_variation = np.sin((hour - 6) * np.pi / 12) * 3  # Peak at noon
        temperature = base_temp + temp_variation + np.random.normal(0, 0.5)
        temperature = np.clip(temperature, config.MIN_TEMPERATURE, config.MAX_TEMPERATURE)
        
        # Simulate occupancy rate
        if is_weekend:
            # Lower occupancy on weekends
            occupancy_rate = np.random.uniform(5, 20)
        elif is_working_hour:
            # Higher occupancy during working hours
            if 9 <= hour <= 11 or 14 <= hour <= 16:
                occupancy_rate = np.random.uniform(70, 100)
            else:
                occupancy_rate = np.random.uniform(50, 80)
        else:
            # Low occupancy outside working hours
            occupancy_rate = np.random.uniform(5, 30)
        
        # Calculate active devices based on occupancy and time
        if is_weekend:
            active_ratio = 0.2  # 20% devices active on weekends
        elif is_working_hour:
            active_ratio = 0.6 + (occupancy_rate / 100) * 0.3
        else:
            active_ratio = 0.15  # Standby devices
        
        active_devices = int(total_devices * active_ratio)
        
        # Calculate energy consumption
        # Base consumption + temperature effect + occupancy effect + random noise
        
        if is_working_hour:
            base_consumption = base_hourly_kwh * (0.8 + np.random.uniform(0, 0.4))
        else:
            base_consumption = base_hourly_kwh * 0.15  # Standby consumption
        
        # Temperature effect (higher temp = more AC usage)
        temp_factor = 1 + (temperature - 27) * 0.05  # 5% increase per degree above 27
        
        # Occupancy effect
        occupancy_factor = 0.5 + (occupancy_rate / 100) * 0.5
        
        # Calculate total consumption with some randomness
        consumption = base_consumption * temp_factor * occupancy_factor
        consumption = consumption * (1 + np.random.normal(0, 0.1))  # 10% random variation
        consumption = max(consumption, base_hourly_kwh * 0.05)  # Minimum consumption
        
        data.append({
            'datetime': dt,
            'hour': hour,
            'day_of_week': day_of_week,
            'month': month,
            'temperature': round(temperature, 1),
            'occupancy_rate': round(occupancy_rate, 1),
            'is_weekend': is_weekend,
            'is_working_hour': is_working_hour,
            'active_devices_count': active_devices,
            'total_energy_consumption': round(consumption, 2)
        })
    
    result_df = pd.DataFrame(data)
    logger.info(f"Generated {len(result_df)} records")
    
    return result_df


def create_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Create additional features for model training.
    
    This function performs feature engineering by creating derived features
    from the raw data.
    
    Args:
        df: DataFrame with raw data
        
    Returns:
        Tuple of:
        - DataFrame with all features
        - List of feature column names
        
    Example:
        >>> X, features = create_features(synthetic_data)
        >>> print(f"Created {len(features)} features")
    """
    logger.info("Creating features for model training...")
    
    df_features = df.copy()
    
    # Time-based features
    df_features['hour_sin'] = np.sin(2 * np.pi * df_features['hour'] / 24)
    df_features['hour_cos'] = np.cos(2 * np.pi * df_features['hour'] / 24)
    df_features['day_sin'] = np.sin(2 * np.pi * df_features['day_of_week'] / 7)
    df_features['day_cos'] = np.cos(2 * np.pi * df_features['day_of_week'] / 7)
    df_features['month_sin'] = np.sin(2 * np.pi * df_features['month'] / 12)
    df_features['month_cos'] = np.cos(2 * np.pi * df_features['month'] / 12)
    
    # Interaction features
    df_features['temp_occupancy'] = df_features['temperature'] * df_features['occupancy_rate'] / 100
    df_features['working_occupancy'] = df_features['is_working_hour'] * df_features['occupancy_rate']
    
    # Peak hour indicator
    df_features['is_peak_hour'] = ((df_features['hour'] >= 9) & (df_features['hour'] <= 17)).astype(int)
    
    # Define feature columns
    feature_columns = [
        'hour', 'day_of_week', 'month',
        'temperature', 'occupancy_rate',
        'is_weekend', 'is_working_hour',
        'active_devices_count',
        'hour_sin', 'hour_cos',
        'day_sin', 'day_cos',
        'month_sin', 'month_cos',
        'temp_occupancy', 'working_occupancy',
        'is_peak_hour'
    ]
    
    logger.info(f"Created {len(feature_columns)} features")
    
    return df_features, feature_columns


def prepare_data(df: pd.DataFrame,
                 target_col: str = 'total_energy_consumption',
                 test_size: Optional[float] = None,
                 random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Prepare data for model training by creating features and splitting.
    
    Args:
        df: Synthetic data DataFrame
        target_col: Target column name
        test_size: Test set size ratio. Uses config if None.
        random_state: Random seed. Uses config if None.
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test, feature_names)
        
    Example:
        >>> X_train, X_test, y_train, y_test, features = prepare_data(synthetic)
    """
    if test_size is None:
        test_size = config.MODEL_TEST_SIZE
    if random_state is None:
        random_state = config.RANDOM_STATE
    
    logger.info("Preparing data for model training...")
    
    # Create features
    df_features, feature_columns = create_features(df)
    
    # Prepare X and y
    X = df_features[feature_columns].values
    y = df_features[target_col].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    logger.info(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test, feature_columns


def train_model(X_train: np.ndarray,
                y_train: np.ndarray,
                params: Optional[Dict[str, Any]] = None) -> xgb.XGBRegressor:
    """
    Train XGBoost regressor model.
    
    Args:
        X_train: Training features
        y_train: Training targets
        params: XGBoost hyperparameters. Uses config if None.
        
    Returns:
        Trained XGBRegressor model
        
    Example:
        >>> model = train_model(X_train, y_train)
    """
    if params is None:
        params = config.XGBOOST_PARAMS
    
    logger.info("Training XGBoost model...")
    logger.info(f"Parameters: {params}")
    
    # Create and train model
    model = xgb.XGBRegressor(**params)
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train)],
        verbose=False
    )
    
    logger.info("Model training complete")
    
    return model


def evaluate_model(model: xgb.XGBRegressor,
                   X_test: np.ndarray,
                   y_test: np.ndarray) -> Dict[str, float]:
    """
    Evaluate model performance using various metrics.
    
    Metrics calculated:
    - RMSE: Root Mean Square Error
    - MAE: Mean Absolute Error
    - MAPE: Mean Absolute Percentage Error
    - R²: Coefficient of Determination
    
    Args:
        model: Trained XGBoost model
        X_test: Test features
        y_test: Test targets
        
    Returns:
        Dictionary with evaluation metrics
        
    Example:
        >>> metrics = evaluate_model(model, X_test, y_test)
        >>> print(f"R² Score: {metrics['r2']:.4f}")
    """
    logger.info("Evaluating model performance...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Calculate MAPE (avoiding division by zero)
    mask = y_test != 0
    mape = np.mean(np.abs((y_test[mask] - y_pred[mask]) / y_test[mask])) * 100
    
    metrics = {
        'rmse': round(rmse, 4),
        'mae': round(mae, 4),
        'mape': round(mape, 2),
        'r2': round(r2, 4)
    }
    
    logger.info(f"RMSE: {metrics['rmse']}")
    logger.info(f"MAE: {metrics['mae']}")
    logger.info(f"MAPE: {metrics['mape']}%")
    logger.info(f"R²: {metrics['r2']}")
    
    return metrics


def get_feature_importance(model: xgb.XGBRegressor,
                          feature_names: List[str]) -> pd.DataFrame:
    """
    Extract feature importance from trained model.
    
    Args:
        model: Trained XGBoost model
        feature_names: List of feature names
        
    Returns:
        DataFrame with feature importances sorted by importance
        
    Example:
        >>> importance = get_feature_importance(model, feature_names)
        >>> print(importance.head(10))
    """
    logger.info("Extracting feature importance...")
    
    importance = model.feature_importances_
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    })
    
    # Sort by importance
    importance_df = importance_df.sort_values('Importance', ascending=False)
    importance_df = importance_df.reset_index(drop=True)
    
    # Calculate percentage
    total_importance = importance_df['Importance'].sum()
    importance_df['Percentage'] = (importance_df['Importance'] / total_importance * 100).round(2)
    
    logger.info(f"Top feature: {importance_df.iloc[0]['Feature']} ({importance_df.iloc[0]['Percentage']:.2f}%)")
    
    return importance_df


def predict(model: xgb.XGBRegressor,
            X: np.ndarray) -> np.ndarray:
    """
    Make predictions using trained model.
    
    Args:
        model: Trained XGBoost model
        X: Features to predict on
        
    Returns:
        Array of predictions
        
    Example:
        >>> predictions = predict(model, X_new)
    """
    return model.predict(X)


def save_model(model: xgb.XGBRegressor,
               filepath: Optional[Path] = None) -> Path:
    """
    Save trained model to file.
    
    Args:
        model: Trained XGBoost model
        filepath: Output file path. Uses default if None.
        
    Returns:
        Path to saved model file
        
    Example:
        >>> path = save_model(model)
        >>> print(f"Model saved to: {path}")
    """
    if filepath is None:
        filepath = config.OUTPUTS_DIR / config.MODEL_FILE
    
    # Ensure directory exists
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Saving model to: {filepath}")
    
    with open(filepath, 'wb') as f:
        pickle.dump(model, f)
    
    logger.info("Model saved successfully")
    
    return filepath


def load_model(filepath: Optional[Path] = None) -> xgb.XGBRegressor:
    """
    Load trained model from file.
    
    Args:
        filepath: Path to model file. Uses default if None.
        
    Returns:
        Loaded XGBoost model
        
    Example:
        >>> model = load_model()
    """
    if filepath is None:
        filepath = config.OUTPUTS_DIR / config.MODEL_FILE
    
    logger.info(f"Loading model from: {filepath}")
    
    with open(filepath, 'rb') as f:
        model = pickle.load(f)
    
    logger.info("Model loaded successfully")
    
    return model


def train_and_evaluate(df_devices: pd.DataFrame,
                       num_days: Optional[int] = None,
                       save_trained_model: bool = True) -> Dict[str, Any]:
    """
    Complete training pipeline: generate data, train, and evaluate.
    
    This is the main entry point for model training.
    
    Args:
        df_devices: Device inventory DataFrame
        num_days: Number of days to simulate
        save_trained_model: Whether to save the trained model
        
    Returns:
        Dictionary containing:
        - model: Trained XGBoost model
        - metrics: Evaluation metrics
        - feature_importance: Feature importance DataFrame
        - predictions: Test set predictions
        - actuals: Test set actual values
        - synthetic_data: Generated synthetic data
        
    Example:
        >>> results = train_and_evaluate(df_processed)
        >>> print(f"Model R²: {results['metrics']['r2']:.4f}")
    """
    logger.info("Starting complete training pipeline...")
    
    # Generate synthetic data
    synthetic_data = generate_synthetic_data(df_devices, num_days=num_days)
    
    # Prepare data
    X_train, X_test, y_train, y_test, feature_names = prepare_data(synthetic_data)
    
    # Train model
    model = train_model(X_train, y_train)
    
    # Evaluate model
    metrics = evaluate_model(model, X_test, y_test)
    
    # Get feature importance
    importance = get_feature_importance(model, feature_names)
    
    # Get predictions
    y_pred = predict(model, X_test)
    
    # Save model if requested
    if save_trained_model:
        save_model(model)
    
    results = {
        'model': model,
        'metrics': metrics,
        'feature_importance': importance,
        'predictions': y_pred,
        'actuals': y_test,
        'synthetic_data': synthetic_data,
        'feature_names': feature_names,
        'train_size': len(X_train),
        'test_size': len(X_test)
    }
    
    logger.info("Training pipeline complete")
    
    return results


if __name__ == "__main__":
    # Test the module with sample data
    print("Testing xgboost_model module...")
    print("-" * 50)
    
    # Create sample device data
    sample_devices = pd.DataFrame({
        'Device_Name': ['AC Split 1 PK', 'LED Light 18W', 'Desktop PC'],
        'Device_Category': ['Cooling', 'Lighting', 'Computing'],
        'Quantity': [20, 150, 50],
        'Power_Watt': [900, 18, 300],
        'Operating_Hours_Per_Day': [8, 10, 8],
        'Total_Power_Watt': [18000, 2700, 15000],
        'Daily_Energy_kWh': [144.0, 27.0, 120.0],
        'Monthly_Energy_kWh': [3168.0, 594.0, 2640.0]
    })
    
    # Run training pipeline
    results = train_and_evaluate(sample_devices, num_days=30, save_trained_model=False)
    
    print(f"\nModel Evaluation Metrics:")
    print(f"  RMSE: {results['metrics']['rmse']:.4f}")
    print(f"  MAE:  {results['metrics']['mae']:.4f}")
    print(f"  MAPE: {results['metrics']['mape']:.2f}%")
    print(f"  R²:   {results['metrics']['r2']:.4f}")
    
    print(f"\nTop 5 Features:")
    print(results['feature_importance'].head())
