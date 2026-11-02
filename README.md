# Forcasty – Solar Energy Forecasting & Analysis in Tunisia

<p align="center">
  **Data Science | Renewable Energy | Solar Forecasting | Machine Learning**
</p>

<p align="center">
  An interactive solar energy analysis platform exploring solar resource availability and photovoltaic energy potential across Tunisia.
</p>

---

## Overview

**Forcasty** is a data-driven renewable energy project designed to explore solar energy potential across multiple regions of Tunisia.

The platform combines meteorological data, solar irradiance analysis, and machine learning techniques to investigate the relationship between weather conditions and potential photovoltaic energy production.

By integrating historical and weather-related datasets with predictive modelling, Forcasty aims to provide an accessible environment for analysing solar resources and exploring data-driven approaches to renewable energy forecasting.

## Key Features

- **Multi-Location Solar Analysis:** Explore solar energy conditions across Tunis, Sfax, Sousse, Gabès, and Tozeur.
- **Meteorological Data Integration:** Retrieve weather and solar radiation data using NASA POWER and Open-Meteo.
- **Solar Irradiance Analysis:** Investigate solar resource availability and its temporal variations.
- **Machine Learning:** Implement and compare predictive models, including Linear Regression and Random Forest.
- **Model Evaluation:** Assess predictive performance using appropriate regression metrics.
- **Interactive Dashboard:** Visualize meteorological variables, solar irradiance, and estimated photovoltaic energy production.

## Technical Approach

### 1. Data Acquisition

The project integrates meteorological and solar radiation datasets from:

- [NASA POWER](https://power.larc.nasa.gov/) – Solar and meteorological data.
- [Open-Meteo](https://open-meteo.com/) – Weather data and historical meteorological information.

### 2. Data Processing

Data processing and preparation involve:

- Data cleaning and preprocessing.
- Handling missing observations.
- Feature selection and engineering.
- Temporal analysis.
- Preparation of datasets for machine learning.

### 3. Machine Learning

The project explores regression-based approaches for solar forecasting.

| Model | Purpose |
|---|---|
| Linear Regression | Establish a baseline predictive model |
| Random Forest | Capture nonlinear relationships between meteorological variables and solar irradiance |

Model performance is evaluated using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Coefficient of Determination (R²)

### 4. Solar Energy Assessment

The platform explores the relationship between solar irradiance and photovoltaic energy production.

A simplified photovoltaic energy estimation can be expressed as:

\[
E_{PV}=G \times A \times \eta \times PR
\]

Where:

- \(E_{PV}\): Estimated electrical energy production.
- \(G\): Solar irradiation over the considered period.
- \(A\): Photovoltaic panel surface area.
- \(\eta\): Module efficiency.
- \(PR\): Performance ratio.

### 5. Data Visualization

Interactive visualizations allow users to explore:

- Solar irradiance variations.
- Meteorological conditions.
- Geographic differences in solar resources.
- Predicted versus observed values.
- Estimated photovoltaic energy potential.

## Technology Stack

| Category | Technologies |
|---|---|
| Programming | Python |
| Data Analysis | Pandas, NumPy |
| Machine Learning | Scikit-learn |
| Visualization | Plotly, Matplotlib |
| Dashboard | Streamlit |
| Data Sources | NASA POWER, Open-Meteo |
| Development | Jupyter Notebook, VS Code, Git |

## Project Architecture

```text
Forcasty/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   └── exploratory_analysis.ipynb
│
├── src/
│   ├── data_acquisition/
│   ├── preprocessing/
│   ├── models/
│   └── visualization/
│
├── app/
│   └── dashboard.py
│
├── reports/
│
├── requirements.txt
├── README.md
└── .gitignore
```

*Illustrative structure; adapt it to the actual repository.*

## Project Objectives

- Apply data science techniques to renewable energy engineering.
- Investigate solar resource availability across Tunisia.
- Develop practical machine learning forecasting skills.
- Explore the use of meteorological data in photovoltaic energy assessment.
- Build an interactive tool for communicating solar energy insights.

## Current Development Status

**Status: In Development**

Planned improvements include:

- Expanded historical dataset coverage.
- Advanced feature engineering.
- Comparison of additional machine learning models.
- Improved forecasting accuracy and validation.
- More detailed photovoltaic performance estimation.
- Deployment of the interactive dashboard.

## Author

**Mouath Chouchane**

Energy Engineering Student | Renewable Energy & Data Science

Interested in sustainable energy systems, computational modelling, machine learning, and data-driven engineering.

---

*Forcasty is an independent engineering and data science portfolio project focused on exploring solar energy forecasting and renewable energy potential in Tunisia.*
