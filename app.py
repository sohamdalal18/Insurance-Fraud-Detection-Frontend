import numpy as np
import os
import pickle
import warnings
from dotenv import load_dotenv
from flask import Flask, render_template, request, redirect, url_for
import google.generativeai as genai

warnings.filterwarnings("ignore")

# Load environment variables from the .env file
load_dotenv()

# Get the Google API key from the environment variable
google_api_key = os.getenv("GOOGLE_API_KEY")

# Configure the Generative AI API with the API key
genai.configure(api_key=google_api_key)

# Initialize the model
model = genai.GenerativeModel("gemini-pro")

# Start the chat
chat = model.start_chat(history=[])

# Get the base directory of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Construct paths to the pickle files using raw strings
model_path = os.path.join(BASE_DIR, "notebooks", "trained_model.sav")
scaler_path = os.path.join(BASE_DIR, "notebooks", "standardized_1.pkl")

# Load the pickle files
loaded_model = pickle.load(open(model_path, 'rb'))
scaler = pickle.load(open(scaler_path, 'rb'))


app = Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/bot", methods=["GET", "POST"])
def bot():
    response_text = ""
    if request.method == "POST":
        user_input = request.form.get("query")
        if user_input:
            response = get_gemini_response(user_input)
            response_text = "".join(chunk.text for chunk in response)
    return render_template("bot.html", response_text=response_text)


@app.route("/model", methods=["GET"])
def model():
    return render_template("model.html")


@app.route("/prediction",methods=["POST"])
def prediction():
    try:
        # Collect input data from the form
        input_data = [
            request.form.get("months"),
            request.form.get("deductible"),
            request.form.get("umbrella"),
            request.form.get("gains"),
            request.form.get("loss"),
            request.form.get("hour"),
            request.form.get("vehicles"),
            request.form.get("injuries"),
            request.form.get("witnesses"),
            request.form.get("injury_claim"),
            request.form.get("property_claim"),
            request.form.get("vehicle_claim"),
        ]

        def validate_input_data(input_data):
            # Convert inputs to appropriate types and handle missing/invalid data
            try:
                return [float(i) for i in input_data]
            except ValueError:
                return None  # Handle invalid data
            
        input_data = validate_input_data(input_data)

        if input_data is None:
            return render_template("model.html", error="Invalid input data. Please provide numerical values.")

        def predict_insurance_fraud(input_data):
            """Function to make predictions based on input data."""
            input_data_as_numpy_array = np.asarray(input_data).reshape(1, -1)
            standardized_data = scaler.transform(input_data_as_numpy_array)
            prediction = loaded_model.predict(standardized_data)
            return "Insurance Claim is Legitimate" if prediction[0] == 0 else "Insurance Claim is Fraudulent"
        
        prediction_result = predict_insurance_fraud(input_data)
        return render_template("result.html", Prediction=prediction_result)
    
    except Exception as e:
        return f"Error: {str(e)}"
    
# Function to get Gemini responses
def get_gemini_response(question):
    response = chat.send_message(question, stream=True)
    return response
    
if __name__ == "__main__":
    app.run(debug=True)
