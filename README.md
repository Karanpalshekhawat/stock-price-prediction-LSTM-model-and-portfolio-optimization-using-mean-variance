## Predictive Stock Selection and Portfolio Optimization using Machine Learning models

---
An `approach` to build a quantitative alpha trading model to identify best performing stocks from a list of stocks and
their corresponding
ideal weights in the portfolio.

Prediction is based on the multi-step forecasting using different machine learning models and comparing its performance
during the actual period.

---

### Main Features

I have implemented below ML model and portfolio optimization techniques to analyze the impact on an actual stocks
performance:-

<div align="center">

|       **Stock Prediction ML model**        | 
|:------------------------------------------:|
| Linear regression using Elastic Net (LREN) | 
|   Long-Short Term Memory Network (LSTM)    |

</div>

<div align="center">

| **Portfolio Optimization Techniques** |
|:-------------------------------------:|
|          Equal Weights (EQ)           |
|   Mean-variance Optimization (MVO)    |

</div>


### Stocks included in the analysis:-

---

<div align="center">

|        Stock Names        |
|:-------------------------:|
| State Bank of India (SBI) |
|    Reliance Industries    |
|       Bharti Airtel       |
|         Axis Bank         |
|       Tech Mahindra       |
|       Eicher Motors       |
|            TCS            |
|       Asian Paints        |
|    Hindustan Unilever     |

</div>

### Aim

---

Aim of this project is to train LSTM RNN model individually for each stock mentioned above. 
Finally, we will create a portfolio of stocks where weight of each stock is optimized using the mean-variance analysis.
It is a simulation based approach in which we minimize the variance of the portfolio for the given return. 
Return for each stock in the mean-variance model is quantified as per the multi-step forecasted prediction of the trained LSTM model for each stock while variance of each stock and covariance matrix across all stock pairs is based on their historical price trend data.

### Features and Technical momentum indicators included in the model

---

1. 1 month past historical daily returns or underlying stock price
2. 20 and 50 days moving average
3. Relative Strength index
4. Moving Average Convergence Divergence
5. Lower and upper bollinger bonds

### Steps followed to build LSTM RNN model

---

1. Computing features as explained above.
2. Train-test split. This step is critical as we are building time series model, so, we cannot directly split it based on random selection. 
3. Normalized features based on training dataset, and save the scaler information to re-use in case of validation and test set.
4. Built a basic LSTM model of 3 layers with batch normalization and dropout, check out method create_lstm_model present under `model/core/LSTM` module.
5. Define set of hyper-parameters and it's possible range of values. It includes activation function, number of neurons, initialization, optimizer etc.
6. Train LSTM model for each stock with a early stop callback feature where we will stop training if validation loss is below some certain threshold. I tried different threshold to check performance of the model.
7. Used self defined Grid Search approach to identify the best hyper-parameters individually for each stock.


### Model Performance

Notebook `LSTM_model_accuracy.ipynb` present under `notebooks` directory demonstrates the performance of LSTM model for each stock. It contains below 2 charts for each stock based on prediction from the trained LSTM model.

- `Daily Adjusted Prediction`: Predict next day stock price based on the updated new features by incorporating the effect of new day data due to each passing day in the features.
- `Multi-Step Forecasting`: Predict price based on multi-step forecasting technique where I have used the predicted values from the model as an input to predict the further next day price in future, thus, all future values of stock price are predicted at the time t=0.

As an example, please find the performance plot for Reliance Industries stock price. All other stocks performance are present in the notebook. 

![alt text](https://github.com/Karanpalshekhawat/stock-price-prediction-LSTM-model-and-portfolio-optimization-using-mean-variance/blob/main/model/output/LSTM/img1.png)
![alt text](https://github.com/Karanpalshekhawat/stock-price-prediction-LSTM-model-and-portfolio-optimization-using-mean-variance/blob/main/model/output/LSTM/img2.png)

### Documentation

---

Approach is adapted from research papers as mentioned below.

1. `Portfolio Optimization-Based Stock Prediction using Long-Short Term Memory Network in Quantitative Trading` published by Van-Dai Tai, Chuan-ming liu and Direselign Addis Tadesse.
2. `Markowitz Mean-variance Portfolio Optimization with Predictive Stock Selection using Machine Learning` published by Apichat Chaweewanchon and Rujira Chaysiri.

Pdf is saved under `literature` directory.

### Dependencies

---

- Keras
- Pandas
- sklearn
- NumPy
- MatplotLib

---
