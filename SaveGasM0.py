# app.py
import os
import requests
from flask import Flask, request, jsonify
from firebase_admin import credentials, firestore, initialize_app
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Initialize Firebase
cred = credentials.Certificate("FIREBASE_CREDENTIALS_PATH")  # Update with your Firebase credentials path
initialize_app(cred)
db = firestore.client()

# Hugging Face API settings
HF_API_TOKEN = os.getenv("HF_API_TOKEN")  # Your Hugging Face API token from .env
HF_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"  # Using a stronger model for better responses
HF_API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
headers = {"Authorization": f"Bearer {HF_API_TOKEN}"}

# Test Firebase connection
@app.route('/ping', methods=['GET'])
def ping():
    return jsonify({"status": "Firebase connected!"})

# Add an expense to Firestore
@app.route('/add-expense', methods=['POST'])
def add_expense():
    try:
        data = request.json
        user_id = data.get('user_id', 'demo_user')
        amount = data.get('amount')
        category = data.get('category')

        if not all([amount, category]):
            return jsonify({"error": "Missing 'amount' or 'category'"}), 400

        expense_ref = db.collection('expenses').document()
        expense_ref.set({
            'user_id': user_id,
            'amount': float(amount),
            'category': category,
            'timestamp': firestore.SERVER_TIMESTAMP
        })

        return jsonify({"status": "success", "expense_id": expense_ref.id})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Generate a financial tip using Hugging Face API
@app.route('/generate-tip-hf', methods=['GET'])
def generate_tip_hf():
    try:
        user_id = request.args.get('user_id', 'demo_user')

        # Fetch user's expenses from Firestore
        expenses = db.collection('expenses').where('user_id', '==', user_id).stream()
        expenses_list = [{"amount": e.get('amount'), "category": e.get('category')} for e in expenses]

        # Construct a detailed prompt
        prompt = (
            "You are a financial expert. Analyze the user's expenses and provide a practical, detailed, and creative savings tip. "
            "The tip should be personalized based on their spending habits. Make it engaging and helpful.\n\n"
            f"User's Expenses: {expenses_list}\n\n"
            "Financial Advice:"
        )

        # Hugging Face API request payload
        payload = {
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 256,  # Generate more detailed responses
                "temperature": 0.7,  # Balance randomness for creativity
                "top_p": 0.9  # Enable top-p sampling for diversity
            }
        }

        # Make API request to Hugging Face
        response = requests.post(HF_API_URL, headers=headers, json=payload)
        if response.status_code != 200:
            return jsonify({"error": f"Hugging Face API error: {response.text}"}), response.status_code

        result = response.json()

        # Extract the AI-generated tip
        tip = result[0]['generated_text'] if isinstance(result, list) and 'generated_text' in result[0] else "No tip generated."

        # Clean up and format the response
        tip = tip.split("Financial Advice:")[-1].strip()

        return jsonify({"tip": tip})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
