# Energy Consumption Analysis and Optimization for BTI Campus Building

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-green.svg)](https://xgboost.readthedocs.io/)

A comprehensive system for analyzing and optimizing energy consumption at Bhineka Tunggal Ika (BTI) Campus Building, Universitas Pertahanan RI, using Extreme Gradient Boosting (XGBoost) algorithm.

---

## 📌 Table of Contents

- [English Documentation](#english-documentation)
  - [Project Overview](#project-overview)
  - [Research Objectives](#research-objectives)
  - [Key Features](#key-features)
  - [Technology Stack](#technology-stack)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Project Structure](#project-structure)
  - [Expected Outputs](#expected-outputs)
  - [Customization](#customization)
  - [Results Interpretation](#results-interpretation)
  - [Contributing](#contributing)
  - [License](#license)
- [Dokumentasi Bahasa Indonesia](#dokumentasi-bahasa-indonesia)

---

# English Documentation

## Project Overview

This project provides a complete solution for:
- Analyzing energy consumption patterns in campus buildings
- Predicting energy usage using machine learning (XGBoost)
- Generating optimization recommendations with ROI calculations
- Creating publication-ready figures and tables for academic journals

## Research Objectives

1. **Analyze** current energy consumption patterns at BTI Campus Building
2. **Develop** an accurate prediction model using XGBoost algorithm
3. **Identify** optimization opportunities with quantified potential savings
4. **Generate** actionable recommendations with ROI analysis

## Key Features

- 📊 **Complete Energy Analysis**: Daily, monthly, and annual consumption calculations
- 🤖 **XGBoost Prediction Model**: Machine learning-based energy forecasting
- 📈 **8+ Publication-Ready Visualizations**: High-resolution (300 DPI) figures
- 💰 **Cost Analysis**: Detailed cost breakdown using PLN tariff rates
- 🎯 **Optimization Recommendations**: Device-specific suggestions with ROI
- 📋 **Excel Reports**: Automated generation of analysis tables
- 🔄 **Reproducible Pipeline**: Consistent results with random seed control
- 📝 **Comprehensive Documentation**: Ready for academic publication

## Technology Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.8+ |
| ML Framework | XGBoost |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib, Seaborn |
| Excel Handling | OpenPyXL |
| Interactive Analysis | Jupyter Notebook |

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Step 1: Clone the Repository
```bash
git clone https://github.com/jeremiasinaga1204/energy-optimization-xgboost.git
cd energy-optimization-xgboost
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

## Usage

### Option 1: Run Complete Pipeline (Recommended)
```bash
python main.py
```
This will:
1. Load device inventory from Excel template
2. Calculate energy consumption metrics
3. Train XGBoost prediction model
4. Generate all visualizations
5. Create optimization recommendations
6. Export all results to Excel files

### Option 2: Use Jupyter Notebook
```bash
jupyter notebook notebooks/energy_analysis_complete.ipynb
```
Run cells sequentially for interactive analysis with explanations.

### Option 3: Use Individual Modules
```python
# Import modules
import config
from src import data_input, energy_calculator, xgboost_model, visualization, optimization

# Load and process data
df_devices, summary = data_input.load_and_process_data()

# Train model
results = xgboost_model.train_and_evaluate(df_devices)

# Generate visualizations
visualization.generate_all_visualizations(df_devices, results)

# Get optimization recommendations
report = optimization.generate_optimization_report(df_devices)
```

## Project Structure

```
energy-optimization-xgboost/
├── data/
│   ├── raw/
│   │   └── device_inventory_template.xlsx  # Input: Device inventory
│   ├── processed/                          # Output: Processed data
│   └── results/                            # Output: Analysis results
├── notebooks/
│   └── energy_analysis_complete.ipynb      # Interactive analysis notebook
├── src/
│   ├── __init__.py                         # Package initialization
│   ├── data_input.py                       # Data loading and validation
│   ├── energy_calculator.py                # Energy consumption calculations
│   ├── xgboost_model.py                    # XGBoost model implementation
│   ├── visualization.py                    # Plotting functions
│   └── optimization.py                     # Optimization recommendations
├── outputs/
│   ├── figures/                            # Generated visualization images
│   └── tables/                             # Generated Excel reports
├── config.py                               # Configuration settings
├── main.py                                 # Main pipeline script
├── requirements.txt                        # Python dependencies
└── README.md                               # This file
```

## Expected Outputs

### Figures (outputs/figures/)
| File | Description |
|------|-------------|
| `consumption_by_device.png` | Top 10 energy consumers bar chart |
| `consumption_pie_chart.png` | Category distribution pie chart |
| `prediction_vs_actual.png` | XGBoost model prediction accuracy |
| `feature_importance.png` | Model feature importance ranking |
| `daily_pattern.png` | Hourly consumption pattern |
| `consumption_distribution.png` | Box plot by category |
| `monthly_consumption.png` | Monthly consumption by category |
| `heatmap_schedule.png` | Device operation schedule heatmap |

### Tables (outputs/tables/)
| File | Description |
|------|-------------|
| `consumption_summary.xlsx` | Comprehensive energy analysis |
| `optimization_recommendations.xlsx` | Savings recommendations with ROI |

### Model
| File | Description |
|------|-------------|
| `xgboost_energy_model.pkl` | Trained prediction model |

## Customization

### 1. Replace Sample Data with Real Data
Edit `data/raw/device_inventory_template.xlsx`:
- Update device names, quantities, and power ratings
- Adjust operating hours based on actual schedules
- Ensure correct category assignments

### 2. Modify Configuration Parameters
Edit `config.py`:
```python
ELECTRICITY_TARIFF = 1467  # Change PLN tariff rate
WORKING_DAYS_PER_MONTH = 22  # Adjust working days
SIMULATION_DAYS = 90  # Change simulation period
```

### 3. Adjust XGBoost Hyperparameters
Edit `config.py`:
```python
XGBOOST_PARAMS = {
    'learning_rate': 0.1,
    'max_depth': 5,
    'n_estimators': 100,
    # ... modify as needed
}
```

## Results Interpretation

### Model Performance Metrics
- **R² Score**: >0.85 indicates good model fit
- **RMSE**: Lower is better (absolute error)
- **MAPE**: <10% is excellent prediction accuracy

### Optimization Recommendations
- **Potential Savings**: Estimated reduction in kWh and cost
- **ROI Months**: Time to recover implementation investment
- **Priority Actions**: Quick wins with shortest ROI

## Troubleshooting

### Common Issues

1. **FileNotFoundError**: Ensure Excel template exists at `data/raw/device_inventory_template.xlsx`

2. **ImportError**: Install missing packages:
   ```bash
   pip install -r requirements.txt
   ```

3. **Visualization Issues**: Install matplotlib backend:
   ```bash
   pip install matplotlib[tk]
   ```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this project in academic work, please cite:
```bibtex
@software{energy_optimization_xgboost,
  title={Energy Consumption Analysis and Optimization using XGBoost},
  author={Energy Optimization Team},
  year={2024},
  url={https://github.com/jeremiasinaga1204/energy-optimization-xgboost}
}
```

---

# Dokumentasi Bahasa Indonesia

## Deskripsi Proyek

Sistem lengkap untuk menganalisis dan mengoptimalkan konsumsi energi di Gedung Kampus BTI (Bhineka Tunggal Ika), Universitas Pertahanan RI, menggunakan algoritma Extreme Gradient Boosting (XGBoost).

### Fitur Utama
- Analisis konsumsi energi harian, bulanan, dan tahunan
- Model prediksi menggunakan machine learning XGBoost
- Visualisasi berkualitas publikasi (300 DPI)
- Rekomendasi optimasi dengan perhitungan ROI
- Laporan Excel otomatis

## Cara Instalasi

### Langkah 1: Clone Repository
```bash
git clone https://github.com/jeremiasinaga1204/energy-optimization-xgboost.git
cd energy-optimization-xgboost
```

### Langkah 2: Buat Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

### Langkah 3: Install Dependencies
```bash
pip install -r requirements.txt
```

## Cara Penggunaan

### Menjalankan Analisis Lengkap
```bash
python main.py
```

### Menggunakan Jupyter Notebook
```bash
jupyter notebook notebooks/energy_analysis_complete.ipynb
```

## Struktur Proyek

```
energy-optimization-xgboost/
├── data/raw/                    # Data input (template Excel)
├── data/processed/              # Data yang sudah diproses
├── notebooks/                   # Jupyter notebook
├── src/                         # Modul Python
├── outputs/figures/             # Gambar hasil visualisasi
├── outputs/tables/              # Tabel Excel hasil analisis
├── config.py                    # Pengaturan konfigurasi
├── main.py                      # Script utama
└── requirements.txt             # Daftar dependencies
```

## Kustomisasi Data

1. Buka `data/raw/device_inventory_template.xlsx`
2. Ganti data sampel dengan data perangkat aktual gedung Anda
3. Simpan file dan jalankan analisis

## Output yang Dihasilkan

- 8+ gambar visualisasi (PNG, 300 DPI)
- 3+ file Excel dengan hasil analisis
- Model terlatih (file .pkl)

## Kontak

Untuk pertanyaan atau masukan, silakan buat Issue di repository ini.

---

**Developed for Academic Research Publication**
