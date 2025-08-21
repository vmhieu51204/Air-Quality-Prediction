# Air-Quality-Prediction

Forecasting multi-pollutant air quality using machine learning and deep learning models

---

##  Project Overview

Air pollution poses significant threats to public health, ecosystems, and climate, making accurate air quality prediction a critical area of study. This project explores and compares approaches for predicting air pollutant concentrations—such as CO, NO₂, O₃, SO₂, PM₂.₅, and PM₁₀ from hourly weather attributes (e.g., temperature, humidity,
precipitation, wind speed...) —using both traditional and deep learning techniques.

- Stationary test and visuallization for time series data.
- Air quality forecasting with convLSTM and Random Forest.
- Demonstrated ConvLSTM performance gains: e.g., ~9% reduction in nRMSE for NO₂ and ~21% for SO₂.

---

##  Project Structure

```

├── data/                  # Raw and processed datasets
├── train/                 # Training scripts and configurations
├── models/                # Saved model weights/checkpoints
├── eval/                  # Evaluation scripts and performance metrics
├── plots/                 # Visualization outputs and diagnostic charts
├── utils/                 # Helper modules (data loading, preprocessing)
├── eda.ipynb              # Exploratory data analysis notebook
└── README.md              # Project overview and instructions

```
