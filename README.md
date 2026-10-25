# Sector Rotation Strategy (ML)

Machine learning model that classifies the economic cycle from leading indicators and uses it to rotate between pro-cyclical and defensive sectors, backtested against the S&P 500.

Professional Application Project (PAP II) · ITESO · Aug – Nov 2024 · Team of 4
📄 [Full report (ITESO institutional repository)](https://rei.iteso.mx/items/37138611-c954-4b3c-b604-a9967ded540d)

## Problem

Portfolio managers need to decide when to be overweight, neutral or underweight in pro-cyclical assets. This project tests whether four monthly leading indicators can anticipate those phases well enough to drive a sector rotation strategy.

## Approach

1. **Data:** 291 monthly observations (Mar 2000 – May 2024) of four leading indicators (CLI, BCI, GDP and CCI) and S&P 500 prices.
2. **Labels:** each month is classified as Overweight, Neutral or Underweight depending on whether the S&P 500 moved more than ±2%.
3. **Models:** logistic regression (baseline), XGBoost tuned with Optuna, and a multilayer perceptron (scikit-learn `MLPClassifier`).
4. **Strategy:** the predicted phase drives allocation between pro-cyclical and defensive assets, classified by beta, with dynamic backtesting over random asset selections.

## Results

| Model | Accuracy | Weighted F1 | Test observations |
|---|---|---|---|
| **MLP (ReLU)** | **88.1%** | **0.88** | 59 |
| XGBoost (Optuna) | 71.6% | 0.71 | 88 |
| Logistic regression | 54.2% | 0.49 | 59 |

The MLP outperformed both alternatives, and the rotation strategy was validated with dynamic backtesting against the S&P 500.

## Limitations

- Train and test sets were split randomly (stratified), not chronologically, so results may be optimistic for a time series. A walk-forward validation would be the next step.
- The test set is small (59 months), so accuracy has a wide confidence interval.
- XGBoost was evaluated on a different split (70/30), so its comparison with the other models is indicative.

## Tech stack

Python · pandas · NumPy · scikit-learn · XGBoost · Optuna · yfinance · matplotlib · seaborn

## How to run

```bash
git clone https://github.com/diegotita4/PAP-ERS.git
cd PAP-ERS
python -m venv env
source env/bin/activate        # Windows: env\Scripts\activate
pip install -r requirements.txt
python Analysis/analysis.py
```

## Repository structure

```
Analysis/   analysis.py (main script) and utils.py (EDA, data download, models, backtesting)
Data/       indicators, S&P 500, assets and model dataset
Models/     trained models (.pkl)
```

The report and presentation are included as PDF and PPTX.

## Team

Diego Emilio Enriquez Nares ([@diegotita4](https://github.com/diegotita4)) · Juan Antonio Mugica Liparoli ([@Antonio-IF](https://github.com/Antonio-IF)) · José Alfonso Martínez Ramírez ([@JoAlfonso](https://github.com/JoAlfonso)) · Óscar Uriel Alvarado Garnica ([@Oscar148](https://github.com/Oscar148))

## License

GNU General Public License v3.0
