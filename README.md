# Stock Price Prediction with PyTorch — LSTM vs GRU

This repository follows the **One-Month ML Learning & Project Plan (Stock Price Prediction with PyTorch)** supplied for this project.

## Objective

The project demonstrates a complete machine-learning workflow:

1. Load real historical stock data.
2. Explore the closing-price series.
4. Create sliding-window sequences.
5. Split the time series chronologically into training and test sets.
6. Train an LSTM model.
7. Train a GRU model using the same general hyperparameters.
8. Evaluate both models on unseen test data.
9. Compare MSE, RMSE, and training time.
10. Visualize actual vs predicted prices.
11. Reflect on limitations and the difficulty of stock-price prediction.



## Project scope

The final notebook deliberately follows the plan's simple project scope:

- one stock
- approximately 10 years of historical daily data
- closing price as the single feature
- 20-day lookback
- next-day prediction
- PyTorch LSTM and GRU
- MSE and RMSE evaluation
- training-time comparison

It does not expand the project into a multi-stock or multi-horizon application because those are outside the requirements of this project plan.

## Data source

The default stock is **Amazon (AMZN)**, matching the example used in the project plan.

Historical prices are downloaded at runtime through `yfinance` from Yahoo Finance. No synthetic stock data is silently substituted if the download fails.

The project plan's conceptual/reference article is **Stock Price Prediction with PyTorch** by Rodolfo Saldanha (Medium).

## Repository structure

```text
stock-prediction/
├── stock_prediction_pytorch_complete.ipynb
├── README.md
└── requirements.txt
```

## Requirements

Python 3.10+ is recommended.

```bash
pip install -r requirements.txt
```

Main libraries:

- Python
- Pandas
- NumPy
- Matplotlib
- scikit-learn
- PyTorch
- yfinance
- Jupyter

## Run the project

1. Clone/download the repository.
2. Create and activate a Python virtual environment if desired.
3. Install dependencies.
4. Start Jupyter Notebook or JupyterLab.
5. Open `stock_prediction_pytorch_complete.ipynb`.
6. Run the notebook from top to bottom.

The notebook requires internet access to download real historical data from Yahoo Finance.

## Method

### Data preparation

The notebook downloads approximately 10 years of daily historical prices and extracts the `Close` series.

EDA includes descriptive statistics, missing-value checking, and a closing-price line plot.

### Train/test split

The time series is divided chronologically:

- first 80% → training
- final 20% → test

### Scaling

`MinMaxScaler` scales prices to `[-1, 1]`.

The scaler is fitted **only on training data** and then applied to the test data, preventing future-test information from leaking into preprocessing.

### Sliding window

The default sequence length is 20 trading days:

```text
20 previous trading days → next trading day's closing price
```

### Models

| Setting | Value |
|---|---:|
| Input features | 1 |
| Hidden size | 32 |
| Recurrent layers | 2 |
| Dropout | 0.20 |
| Output values | 1 |
| Epochs | 50 |
| Learning rate | 0.001 |
| Loss | MSE |
| Optimizer | Adam |

Both models use the same general settings for a fair comparison.

### Training

The training loop performs:

- forward pass
- MSE loss calculation
- `loss.backward()`
- `optimizer.step()`

Training loss is recorded and plotted. Training time is measured for both models.

### Evaluation

Both models generate predictions on unseen test data.

Metrics:

- MSE
- RMSE

Predictions are inverse-transformed to the original price scale before final evaluation.

## Hyperparameter tuning

The project plan calls for moderate tuning if needed. The notebook includes a small GRU lookback comparison using 10 vs 20 days rather than an exhaustive search.

## Overfitting and convergence

The notebook plots training MSE and provides a basic diagnostic for overall decrease, possible plateau, and instability.

Because the core experiment does not use a separate validation set, training loss alone cannot prove or rule out test-set overfitting.

## Results

LSTM MSE:  2087.7620
LSTM RMSE: 45.6920
GRU MSE:   537.7048
GRU RMSE:  23.1885
Lower test RMSE: GRU
Faster training in this run: GRU

The README does not invent metrics that have not been produced by an actual run.

## Interpretation

A lower RMSE means lower error on this particular historical test period.

GRUs can sometimes train faster because their recurrent structure is simpler and may use fewer parameters than an equivalent LSTM. However, this is not guaranteed; the actual notebook results determine the conclusion.

A lower historical test error does not establish that the model can reliably predict future stock prices.

## Limitations

- Only the closing price is used.
- Only one stock and one historical test period are evaluated.
- The model does not use news, fundamentals, macroeconomic data, or other market variables.
- Training loss is only a basic overfitting diagnostic.
- Results are sensitive to the data period, hyperparameters, seed, software, and hardware.
- Stock prices are affected by information that cannot be known when making a prediction.

## Critical thinking

Stock price prediction shows why ML metrics need careful interpretation. A model can produce a low historical error without providing dependable real-world forecasts.

The project plan recommends critical thinking through **AI Snake Oil** by Arvind Narayanan and Sayash Kapoor. A relevant lesson is to avoid overstating what a predictive model can accomplish simply because it produces a metric or visualization.

## Reflection and next steps

Possible improvements include:

- additional market features
- different lookback windows


## Disclaimer

This project is for educational purposes only. It does not constitute investment advice.
