# GroupNo.17CryptoCurrencyPredictionUsingPython
This project aims to analyze and predict cyrptocurrency stock prices using Long Short-Term Memory (LSTM) neural networks. The code is written in Python and utilizes various libraries such as pandas, numpy, matplotlib, TensorFlow, scikit-learn, and Plotly for data manipulation, analysis, and visualization.

# Cryptocurrency Price Prediction using LSTM

## Overview

This project utilizes LSTM (Long Short-Term Memory) neural networks to predict the closing prices of cryptocurrencies, specifically Ethereum (ETH) and Litecoin (LTC). The code involves data preprocessing, model creation, training, evaluation, and saving the models and scalers for future use.

## Table of Contents

* [Overview](#overview)
* [Installation](#installation)
* [Usage](#usage)
* [Files and Directory Structure](#files-and-directory-structure)
* [Dependencies](#dependencies)
* [License](#license)

## Overview

This repository contains Python code for predicting Ethereum and Litecoin closing prices using LSTM neural networks. The code involves the following major steps:

*Data Collection and Preprocessing:*
* Fetching cryptocurrency historical data from online sources (CSV files for ETH and LTC)
* Cleaning the data
* Preprocessing it for model training

*Model Creation:*
* Implementing LSTM neural networks using TensorFlow/Keras for predicting the closing prices of cryptocurrencies
* Applying grid search with cross-validation to find optimal hyperparameters

*Training and Evaluation:*
* Training the LSTM models on the preprocessed data
* Evaluating model performance using metrics such as RMSE (Root Mean Squared Error), MSE (Mean Squared Error), and MAE (Mean Absolute Error)

*Visualization:*
* Visualizing the original closing prices
* Visualizing predicted prices
* Visualizing model performance using Plotly and Matplotlib

*Model Saving:*
* Saving trained models and data scalers using joblib and Keras for future use

## Installation

pip install -r requirements.txt


## Usage

*Data Preparation:*

* Place your Ethereum historical data in `ETH-USD.csv` and Litecoin historical data in `LTC-USD.csv` in the `data` directory.

*Run the Code:*

Execute the Python scripts:

bash
python main.py



## Results

The code generates visualizations comparing original and predicted prices for Ethereum and Litecoin and saves trained models and scalers in the `models` directory.

## Files and Directory Structure

* `main.py`: Code for price prediction using LSTM
* `venv/`: Directory containing CSV files of Ethereum and Litecoin historical data
* `venv/`: Directory to store trained models and scalers
* `static/`: Directory for static files used in a web application (if applicable)
* `README.md`: Documentation file (you're reading it)

## Dependencies

* Python 3.x
* TensorFlow
* Keras
* Pandas
* NumPy
* Plotly
* Matplotlib

## License

This project is licensed under the MIT License.
