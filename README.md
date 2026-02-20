# U.S. CO₂ Emissions Simulator

## Overview
This project simulates how changes to the U.S. energy mix affect CO₂ emissions using an XGBoost machine learning model trained on historical U.S. energy, population, GDP, and policy data from 2000-2023. The interactive dashboard allows users to run counterfactual analyses, answering questions like: *"How would our emissions have changed if we had used 10% more renewable energy in 2023?"*

## Data Collection
Data was collected from the U.S. Census Bureau, U.S. Energy Information Administration, and the National Conference of State Legislatures. Key features include:

- **Energy Sources**: Coal, natural gas, petroleum, nuclear, and renewables (biomass, geothermal, hydro, solar, wind)
- **Sector Consumption**: Residential, commercial, industrial, and transportation energy use
- **Economic Indicators**: Real GDP, population, carbon intensity
- **Policy**: Renewable Portfolio Standard (RPS) requirements by state
- **Target Variable**: Total CO₂ emissions (million metric tons)

The dataset contains 1,200 observations (50 states × 24 years) with 48 features after feature engineering.

## Data Cleaning & Feature Engineering
- Dropped unnecessary columns and standardized format (one row per state per year)
- Created percentage breakdown columns for each energy source (energy source use / total energy)
- Generated lagged features for energy sources, GDP, carbon intensity, total consumption, electricity sales, and RPS
- Performed inner merge on State column across all dataframes

## Exploratory Data Analysis

### Correlation Analysis
- Identified 150 feature pairs with strong positive correlation (r > 0.9)
- No strong negative correlations found
- High multicollinearity detected between related energy metrics (e.g., energy use and electricity sales)

### Key Findings from Scatterplot Analysis
- **Strong positive relationships**: Electricity sales, coal use, transportation use, total consumption, residential/commercial/industrial use all showed strong positive correlation with CO₂ emissions
- **Renewable energy patterns**: Most renewable energy data clustered in low-use/low-emissions region, with some outliers explained by facility efficiency variations
- **Industrial use**: Showed slight nonlinearity, suggesting efficiency improvements at higher production levels

## Model Building

### Why XGBoost?
- Handles nonlinear relationships effectively
- Robust to missing data (lagged columns had NaN values for year 2000)
- Provides feature importance insights for feature selection

### Model Iterations

**Trial 1: All Features**
- Train/test split: 80/20
- GridSearchCV with 3-fold cross-validation
- Best hyperparameters: `{colsample_bytree: 0.8, learning_rate: 0.1, max_depth: 3, n_estimators: 200, subsample: 0.8}`
- MSE: 36.85

**Trial 2: Top 25 Features**
- Used only the most predictive features based on feature importance
- MSE: 22.26 (improvement by reducing noise)

**Trial 3: Top 25 Features + Removing Correlated Pairs**
- Removed one feature from each highly correlated pair
- MSE: 78.0 (significant degradation - too many features removed)

**Trial 4: Exclude Energy % Columns**
- Removed percentage columns while retaining all other numerical features
- MSE: 36.85

### Final Model Selection
The model excluding energy percentage columns performed best on test data:
- **Test R²**: 0.998 (99.8% of variance explained)
- **Test RMSE**: 3.43 MMT
- **Relative RMSE**: 3.31% of mean target value

## Dashboard

### Objective
The interactive dashboard allows users to simulate counterfactual scenarios by adjusting energy mix parameters and comparing predicted 2023 emissions to actual 2023 emissions.

### How It Works
1. User specifies % increase/decrease for selected feature(s)
2. Dashboard loads latest historical state data (2023)
3. The % change scales the corresponding 2023 rows for all states
4. Model predicts emissions for each state
5. State emissions are aggregated and compared to actual 2023 emissions

### Baseline Performance
Without any feature adjustments, the model predicts national emissions with only **1.6% error** compared to actual 2023 emissions. The map visualization shows model error by state, with red indicating overprediction.

## Tools & Technologies
- **Python**: Core programming language
- **XGBoost**: Machine learning model
- **Pandas & NumPy**: Data manipulation
- **Scikit-learn**: Model evaluation and GridSearchCV
- **Plotly**: Interactive visualizations
- **Streamlit**: Dashboard framework
- **Matplotlib**: Static visualizations
- **Jupyter Notebook**: Development and analysis

## Running the Code

### Prerequisites
```bash
pip install pandas numpy scikit-learn xgboost plotly streamlit matplotlib jupyter
```

### Running the Dashboard
1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Ensure the trained model and 2023 baseline data are in the project directory

3. Launch the Streamlit dashboard:
```bash
streamlit run dashboard.py
```

4. Open your browser to the provided local URL (typically `http://localhost:8501`)

5. Use the sidebar controls to adjust energy mix parameters and view the simulated impact on CO₂ emissions

### Running the Analysis Notebooks
```bash
jupyter notebook
```
Navigate to the notebooks directory and open the desired analysis file.

## Project Structure
```
├── data/                  # Raw and processed datasets
├── notebooks/             # Jupyter notebooks for EDA and model training
├── dashboard.py           # Streamlit dashboard application
├── model/                 # Trained XGBoost model files
└── README.md
```
