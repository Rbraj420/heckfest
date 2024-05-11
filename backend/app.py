# Import necessary libraries
from flask import Flask, request, jsonify
from modeltrain import train_model, predict_units_sold

# Initialize Flask app
app = Flask(__name__)

# Train the model when the app starts
model = train_model()

# Define a route for training the model
@app.route('/train_model', methods=['GET'])
def train_model_route():
    # Call the train_model function
    model = train_model()
    
    # Return a success message
    return 'Model trained successfully!'

# Define a route for predicting units sold
@app.route('/predict_units_sold', methods=['POST'])
def predict_units_sold_route():
    # Get the JSON data from the request
    request_data = request.get_json()

    # Extract the date from the JSON data
    invoice_date = request_data['invoice_date']

    # Predict units sold
    prediction = predict_units_sold(model, invoice_date)

    # Return the prediction as JSON response
    return jsonify({'predicted_units_sold': prediction})

# Run the app
if __name__ == '__main__':
    app.run(debug=True)

