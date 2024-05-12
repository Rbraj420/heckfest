import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, send_file
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import joblib
from datetime import datetime

# Load the data
data = pd.read_csv("clothes.csv")

# Sort the data by "Profit %" column in ascending order
highest_profit = data.sort_values(by="Profit %", ascending=False)
data = data.sort_values(by="Unit Sold", ascending=True)

# Initialize Flask app
app = Flask(__name__)

# Global variable for the model
model = None

# Extract sorted data for plotting
x = data['Invoice Date']
y = pd.to_numeric(data['Unit Sold'], errors='coerce')
y_sorted = y.sort_values()

# Define the route for the graph
@app.route('/graph')
def get_graph():
    # Bar graph
    plt.xlabel('Invoice Date', fontsize=18)
    plt.ylabel('Unit Sold', fontsize=16)
    plt.bar(x, y_sorted)  
    plt.savefig("graph.png")  # Save the plot as an image
    plt.close()
    return send_file("graph.png", mimetype='image/png')

# Define the route for training the model
@app.route('/train-model', methods=['GET'])
def train_model_route():
    global model
    # Define features and target
    X = data[['Invoice Date']]  
    y = data['Unit Sold']  

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Save the model to disk
    joblib.dump(model, 'model.pkl')

    # Optionally, return some success message
    return 'Model trained successfully!'

# Define the route for predicting units sold
@app.route('/predict-units-sold', methods=['POST'])
def predict_units_sold_route():
    global model
    try:
        # Check if the model is trained
        if model is None:
            return 'Model not trained yet. Please train the model first.', 400

        # Get the JSON data from the request
        request_data = request.get_json()

        # Extract the date from the JSON data
        invoice_date = request_data.get('invoice_date')

        # Check if invoice_date is provided
        if invoice_date is None:
            return 'Invoice date is missing in the request.', 400

        # Convert the input date to the same format as the input features (e.g., a DataFrame)
        user_date_df = pd.DataFrame({'Invoice Date': [invoice_date]})

        # Convert the date string to datetime object
        date_object = datetime.strptime(invoice_date, '%Y-%m-%d')

        # Extract relevant features
        year = date_object.year
        month = date_object.month
        day = date_object.day

        # Make prediction for the input date
        prediction = model.predict([[year, month, day]])
        return jsonify({'predicted_units_sold': prediction[0]})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Run the app
if __name__ == '__main__':
    app.run(debug=True)
