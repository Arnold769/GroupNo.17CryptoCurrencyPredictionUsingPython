from flask import Flask, render_template, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
import pandas as pd
import joblib

app = Flask(__name__)

# Load the trained model and scaler
btc_model = load_model('C:\\Users\\a\\OneDrive - Ashesi University\\Cryptocurrency Prediction\\venv\\btc_best_model_1.h5')
btc_scaler = joblib.load('C:\\Users\\a\\OneDrive - Ashesi University\\Cryptocurrency Prediction\\venv\\btc_scaler_1.pkl')

eth_model = load_model('C:\\Users\\a\\OneDrive - Ashesi University\\Cryptocurrency Prediction\\venv\\eth_best_model_1.h5')
eth_scaler = joblib.load('C:\\Users\\a\\OneDrive - Ashesi University\\Cryptocurrency Prediction\\venv\\eth_scaler_1.pkl')

ltc_model = load_model('C:\\Users\\a\\OneDrive - Ashesi University\\Cryptocurrency Prediction\\venv\\ltc_best_model.h5')
ltc_scaler = joblib.load('C:\\Users\\a\\OneDrive - Ashesi University\\Cryptocurrency Prediction\\venv\\ltc_scaler.pkl')

# Function to predict the future stock prices
def predict_stock_prices(sequence, model, scaler, n_steps):
    prediction_list = []
    temp_input = list(sequence)
    if len(temp_input) > n_steps:
        temp_input = temp_input[-n_steps:]  # Ensure the sequence length matches n_steps

    temp_input = [element[0] if isinstance(element, list) else element for element in temp_input]

    # Reshape the input sequence to fit the LSTM input shape
    x_input = np.array(temp_input)
    x_input = x_input.reshape((1, n_steps, 1))  # Reshape for LSTM input

    yhat = model.predict(x_input, verbose=0)
    temp_input.extend(yhat[0].tolist())
    prediction_list.extend(yhat.tolist())

    return scaler.inverse_transform(np.array(prediction_list).reshape(-1, 1)).flatten().tolist()

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        crypto = request.form['crypto'] 

        if crypto == 'BTC':
            dataset = pd.read_csv('C:\\Users\\a\\OneDrive - Ashesi University\\Cryptocurrency Prediction\\venv\\BTC-USD.csv')
            model = btc_model
            scaler = btc_scaler
        elif crypto == 'ETH':
            dataset = pd.read_csv('C:\\Users\\a\\OneDrive - Ashesi University\\Cryptocurrency Prediction\\venv\\ETH-USD.csv')
            model = eth_model
            scaler = eth_scaler
        elif crypto == 'LTC':
            dataset = pd.read_csv('C:\\Users\\a\\OneDrive - Ashesi University\\Cryptocurrency Prediction\\venv\\LTC-USD.csv')
            model = ltc_model
            scaler = ltc_scaler
        else:
            return jsonify({'error': 'Invalid cryptocurrency'})

        close_prices = dataset['Close'].values.tolist()

        # Transforming input data
        last_sequence = np.array(close_prices[-15:]).reshape(-1, 1)
        last_sequence = scaler.transform(last_sequence)
        last_sequence = last_sequence.reshape(1, -1, 1)  # Reshape for LSTM input

        predicted_data = []  # Initialize the list to hold predicted data
        normal_predicted_prices_array = []
        normal_predicted_prices = {} 

        i = 1
        # Loop to predict future stock prices for the next 30 days
        for _ in range(30):
            # Predict future stock prices for the next 15 days
            predicted_prices = predict_stock_prices(last_sequence, model, scaler, n_steps=15)

            # Normalize predicted prices before extending
            normalized_predicted_prices = scaler.transform(np.array(predicted_prices).reshape(-1, 1)).flatten()
            original_predicted_prices = scaler.inverse_transform(np.array(normalized_predicted_prices).reshape(-1, 1)).flatten()

            # Prepare a list of dictionaries with day number and normalized predicted price
            predictions = [{'Day': i, 'Price': price} for i, price in enumerate(normalized_predicted_prices, start=1)]

            # Append the predictions to the overall list
            predicted_data.extend(predictions)

            # Extend the sequence with normalized predicted values for further prediction
            extended_sequence = last_sequence[0].tolist()
            extended_sequence.extend(normalized_predicted_prices)
            # Convert the last element into an array before appending
            last_element = [extended_sequence[-1]]

            # Append the last element to the extended sequence
            extended_sequence.append(last_element)

            # Remove non-array element from the extended sequence
            extended_sequence = [elem for elem in extended_sequence if isinstance(elem, list)]

            # Update last_sequence with the last 15 values of extended_sequence
            last_sequence = np.array([extended_sequence[-15:]]).reshape(1, -1, 1)

            # Store normal predicted prices in the dictionary
            for day, price in zip(range(i, i+len(original_predicted_prices)), original_predicted_prices):
                normal_predicted_prices[day] = price

            i += len(original_predicted_prices)  # Increment the day number

            print(predicted_data)
            for day, price in normal_predicted_prices.items():
                print(f"Day {day}: Normal Predicted Price {price}")

            daily_predictions = {}
            for day, price in zip(range(i, i + len(original_predicted_prices)), original_predicted_prices):
                daily_predictions[day] = price

            normal_predicted_prices_array.append(daily_predictions)
            
        return jsonify(normal_predicted_prices)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
