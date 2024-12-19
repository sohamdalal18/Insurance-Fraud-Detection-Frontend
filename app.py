import numpy as np
import os
import pickle
import warnings
from dotenv import load_dotenv
from flask import Flask, render_template, request, redirect, url_for
from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

warnings.filterwarnings("ignore")

# Load environment variables from the .env file
load_dotenv()

def chatbot_response(user_input):
    import os
    from dotenv import load_dotenv
    from langchain_ollama import OllamaLLM
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser

    # Load environment variables
    load_dotenv()
    os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT")

    # Define the prompt
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """
                You are an expert in Indian insurance with comprehensive knowledge of all aspects of vehicle insurance, for all types of vechiles, and general insurance. You are familiar with Indian regulations, insurance providers, tax benefits under the Income Tax Act, claim processes, and market trends. Your responses should be clear, accurate, and tailored to the user's needs.
                Strictly follow this : If a user asks a question that is unrelated to insurance, politely respond with:'I am here to assist with insurance-related queries only. Please ask me a question related to insurance.'
                

                Examples of Queries You Can Handle:
                - Recommendations for different types of vehicle insurances based on user requirements.
                - Explanation of policies and their benefits.
                - Claim filing and settlement processes.
                - Tax benefits of insurance under Indian laws (e.g., Section 80C, 80D).
                - Comparison of policies from major Indian insurance providers.

                Tone and Style:
                - Be professional, yet approachable.
                - Provide detailed and actionable advice.
                - Simplify complex insurance terms for better understanding.

                Constraints:
                - Ensure all recommendations are general and unbiased; avoid endorsing specific companies unless explicitly asked.
                - Avoid giving financial advice outside the scope of insurance.
                - Avoid using raw Markdown or asterisks in the text directly.
                """
            ),
            ("user", f"Question: {user_input}")
        ]
    )

    # Initialize the LLM
    llm = OllamaLLM(model="gemma:2b")
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser

    # Example questions
    example_questions = [
        "What are the tax benefits of life insurance in India?",
        "How does motor insurance coverage work?",
        "What are the steps to file a vehicle insurance claim?",
        "What is the difference between term insurance and ULIPs?",
        "Which policies are best for senior citizens in India?"
    ]

    try:
        response = chain.invoke({"question": user_input})
        return response
    except Exception as e:
        return f"Error processing responsepyt: {e}"

# Get the base directory of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Construct paths to the pickle files using raw strings
model_path = os.path.join(BASE_DIR, r"notebooks\trained_model.sav")
scaler_path = os.path.join(BASE_DIR, r"notebooks\standardized_1.pkl")

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
            response = chatbot_response(user_input)
            response_text = response
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
    
if __name__ == "__main__":
    app.run(debug=False)
