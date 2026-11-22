
# ⚡ AI-Based Electricity Demand Forecasting and Peak Load Alert System

An intelligent electricity demand forecasting system that uses Machine Learning to predict hourly electricity consumption, analyze demand patterns, and identify potential peak-load conditions.

## 📌 Project Overview

Electricity demand changes throughout the day due to usage patterns, weather conditions, and other factors. Accurate demand forecasting can help improve electricity planning and support better load management.

This project uses historical electricity consumption data and a Random Forest Regression model to forecast future demand and identify potential peak-load situations.

## 🎯 Objectives

- Forecast hourly electricity demand using historical consumption data.
- Analyze electricity usage patterns and trends.
- Identify potential peak-demand conditions.
- Display forecasts and insights through an interactive dashboard.
- Explore how machine learning can support energy management.

## ✨ Key Features

- Historical electricity demand analysis.
- Data exploration and preprocessing.
- Machine Learning-based demand forecasting.
- Random Forest Regression model.
- Next 24-hour demand forecasting.
- Peak-load alerts based on historical demand thresholds.
- Interactive Streamlit dashboard.
- Model performance evaluation.

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming |
| Pandas & NumPy | Data processing |
| Scikit-learn | Machine Learning |
| Random Forest | Demand prediction |
| Streamlit | Interactive dashboard |
| Matplotlib & Seaborn | Data visualization |
| Joblib | Model serialization |

## 📊 Model Performance

The Random Forest model was evaluated against a previous-day demand baseline.

| Metric | Random Forest | Previous-day Baseline |
|---|---:|---:|
| MAE | 154.03 MW | 921.61 MW |
| RMSE | 206.38 MW | 1221.53 MW |

The Random Forest model achieved substantially lower prediction errors than the baseline on the evaluation dataset.

## 📁 Project Structure

```text
electricity-demand-forecasting/
│
├── 01_Data_Exploration.ipynb
├── 02_Data_Preprocessing.ipynb
├── app.py
├── check_setup.py
├── inspect_data.py
├── train_model.py
├── requirements.txt
└── README.md
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/Saritap2561/Electricity-demand-forecasting.git
```

### 2. Navigate to the project folder

```bash
cd Electricity-demand-forecasting
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the dataset

Place the AEP hourly electricity consumption dataset in the project directory.

Dataset file: `AEP_hourly.csv`

Note: The dataset is not included in this repository.

### 5. Train the model

```bash
python train_model.py
```

### 6. Launch the dashboard

```bash
streamlit run app.py
```

## 📈 Future Enhancements

- Integration of weather information.
- Comparison of multiple forecasting algorithms.
- Improved peak-load prediction.
- Automated notifications for high-demand periods.
- Deployment as a cloud-based application.

## 👩‍💻 Author

**Sarita Patel**

BSc Computer Science Student

GitHub: [Saritap2561](https://github.com/Saritap2561)

---

*Developed as an academic Machine Learning project focused on electricity demand forecasting and energy management.*
