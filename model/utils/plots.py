"""
This module is used to Plot the prediction
to demonstrate model performance. This method
is used within the notebook.
"""

import pandas as pd
import datetime as dt
import matplotlib.pyplot as plt


def plotting_trend(stock, df_one_step, df_multi_step, prediction_start_date):
    """
    Plotting the predicted values.

    Args:
        stock (str): stock name
        df_one_step (pd.DataFrame): next day prediction one day at a time
        df_multi_step (pd.DataFrame): multiple days prediction in future all at the same time, using
                                      predicted values as an input to predict further day value
        prediction_start_date (dt.datetime): prediction start date for multi-step forecasting

    """
    df_one_step = df_one_step.drop(df_one_step.index[0])
    fig, axs = plt.subplots(2, 1, figsize=(20, 23))  # 2 rows, 1 column
    axs[0].plot(df_one_step.index, df_one_step['Predicted'], label="Predicted")
    axs[0].plot(df_one_step.index, df_one_step['Actual'], label="Actual")

    axs[0].set_title(
        f"{stock} Actual trend vs Predicted trend based on updating features on a daily basis to predict next day Price",
        fontsize=20, color="darkblue")
    axs[0].set_xlabel("Date", fontsize=25, color="black")
    axs[0].set_ylabel("Benchmarked Price", fontsize=25, color="black")
    axs[0].tick_params(axis='both', labelsize=20)

    axs[0].set_frame_on(True)
    axs[0].spines['top'].set_visible(True)
    axs[0].spines['right'].set_visible(True)
    axs[0].spines['bottom'].set_visible(True)
    axs[0].spines['left'].set_visible(True)

    axs[0].grid(b=True, color="aqua", alpha=0.5, linestyle='-.')
    axs[0].legend(loc='upper left', prop={'size': 25})

    axs[1].plot(df_multi_step.index, df_multi_step['Predicted Adj Close'], label="Predicted")
    axs[1].plot(df_multi_step.index, df_multi_step['Adj Close'], label="Actual")

    axs[1].set_title(f"{stock} Actual trend comparison with Predicted trend based on Multi-Step forecasting ",
                     fontsize=20, color="darkblue")
    axs[1].set_xlabel("Date", fontsize=25, color="black")
    axs[1].set_ylabel("Stock Price", fontsize=25, color="black")
    axs[1].tick_params(axis='both', labelsize=20)

    axs[1].set_frame_on(True)
    axs[1].spines['top'].set_visible(True)
    axs[1].spines['right'].set_visible(True)
    axs[1].spines['bottom'].set_visible(True)
    axs[1].spines['left'].set_visible(True)

    axs[1].grid(b=True, color="aqua", alpha=0.5, linestyle='-.')
    axs[1].legend(loc='upper left', prop={'size': 25})
    axs[1].axvline(x=prediction_start_date, color="r", linestyle="--", label="Prediction Date")
    text_box_date = prediction_start_date - dt.timedelta(days=8)
    axs[1].text(text_box_date, 0.1, 'Prediction Date', rotation=90, transform=axs[1].get_xaxis_text1_transform(0)[0],
                color="darkblue", fontsize=20)
